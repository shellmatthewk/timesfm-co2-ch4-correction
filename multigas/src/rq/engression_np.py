"""NumPy re-implementation of the frozen-TimesFM Engression head, plus residual variants.

Why NumPy: PyTorch cannot be installed in the environments reachable for this run,
and the TimesFM backbone is frozen, so only a small MLP on cached features is trained.
The control reproduces the protocol of the earlier ``z1280_c12`` case (1280-dim noise,
lambda=1/2, AdamW lr=2e-4, wd=0.01, clip 1.0, batch 32, 16/256/512 samples, patience 12,
max 80 epochs, validation-ES selection with epoch 0 eligible). Arithmetic is float64.

Three heads share one MLP class g([h, eps]) : 1280+Z -> 64 -> 64 -> 14 (SiLU):

* ``control``   sample = center + scale * g([h, eps]),  g warm-started to native raw q50 + A*eps
* ``residual``  sample = q50_native + scale * g([h, eps]),  g warm-started to 0 + A*eps
* ``residual_centered``  sample = q50_native + scale * (g - median_over_samples(g))

``q50_native`` is the sorted native TimesFM median in ppm (the reported native point
forecast). In the centered head the sample median equals q50_native exactly, so the
point forecast cannot drift; Engression learns only the shape/spread/dependence.
One noise vector per trajectory is shared by the four stations, as before.
"""
from __future__ import annotations

import numpy as np

HORIZON, STATIONS, NQ, MEDIAN = 14, 4, 9, 4
HEADS = ("control", "residual", "residual_centered")


def silu(a):
    s = 1.0 / (1.0 + np.exp(-a))
    return a * s, s


class Head:
    """Parameters and manual forward/backward for the shared station MLP."""

    names = ("Wh", "We", "b1", "W2", "b2", "W3", "b3")

    def __init__(self, feature_dim, noise_dim, hidden, rng):
        # torch.nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias.
        def lin(fan_out, fan_in):
            k = 1 / np.sqrt(fan_in)
            return rng.uniform(-k, k, (fan_out, fan_in)), rng.uniform(-k, k, fan_out)
        W1, self.b1 = lin(hidden, feature_dim + noise_dim)
        self.Wh, self.We = W1[:, :feature_dim].copy(), W1[:, feature_dim:].copy()
        self.W2, self.b2 = lin(hidden, hidden)
        self.W3, self.b3 = lin(HORIZON, hidden)
        self.feature_dim, self.noise_dim, self.hidden = feature_dim, noise_dim, hidden

    def params(self):
        return {n: getattr(self, n) for n in self.names}

    def set_params(self, p):
        for n in self.names:
            setattr(self, n, np.array(p[n], dtype=np.float64, copy=True))

    def warm_start(self, weight, bias, noise_scale):
        """Same paired-SiLU identity as uq14.initialization: g = W h + b + A eps."""
        H = HORIZON
        A = np.zeros((H, self.noise_dim))
        A[:, :H] = np.diag(np.full(H, noise_scale))
        self.Wh[:H], self.We[:H], self.b1[:H] = weight, A, bias
        self.Wh[H:2 * H], self.We[H:2 * H], self.b1[H:2 * H] = -weight, -A, -bias
        I = np.eye(H)
        self.W2[:2 * H] = 0
        self.W2[:H, :H], self.W2[:H, H:2 * H] = I, -I
        self.W2[H:2 * H, :H], self.W2[H:2 * H, H:2 * H] = -I, I
        self.b2[:2 * H] = 0
        self.W3[:] = 0
        self.W3[:, :H], self.W3[:, H:2 * H] = I, -I
        self.b3[:] = 0

    def forward(self, h, eps):
        """h [B,4,D], eps [B,S,Z] -> g [B,S,4,14] and a cache for backward."""
        a1 = (h @ self.Wh.T)[:, None] + (eps @ self.We.T)[:, :, None] + self.b1
        z1, s1 = silu(a1)
        a2 = z1 @ self.W2.T + self.b2
        z2, s2 = silu(a2)
        out = z2 @ self.W3.T + self.b3
        return out, (h, eps, a1, z1, s1, a2, z2, s2)

    def backward(self, dout, cache):
        h, eps, a1, z1, s1, a2, z2, s2 = cache
        g = {}
        g["W3"] = np.einsum("bsco,bsch->oh", dout, z2)
        g["b3"] = dout.sum((0, 1, 2))
        dz2 = dout @ self.W3
        da2 = dz2 * (s2 * (1 + a2 * (1 - s2)))
        g["W2"] = np.einsum("bsco,bsch->oh", da2, z1)
        g["b2"] = da2.sum((0, 1, 2))
        dz1 = da2 @ self.W2
        da1 = dz1 * (s1 * (1 + a1 * (1 - s1)))
        g["b1"] = da1.sum((0, 1, 2))
        g["Wh"] = np.einsum("bch,bcd->hd", da1.sum(1), h)
        g["We"] = np.einsum("bsh,bsz->hz", da1.sum(2), eps)
        return g


def native_q50_weights(output_head_weight, output_head_bias):
    rows = np.arange(HORIZON) * NQ + MEDIAN
    return output_head_weight[rows].astype(np.float64), output_head_bias[rows].astype(np.float64)


class Model:
    def __init__(self, kind, head, q50_weight, q50_bias, noise_scale):
        if kind not in HEADS:
            raise ValueError(kind)
        self.kind, self.head = kind, head
        if kind == "control":
            head.warm_start(q50_weight, q50_bias, noise_scale)
        else:  # residual g starts at 0 + A*eps: identical initial spread to the control.
            head.warm_start(np.zeros_like(q50_weight), np.zeros_like(q50_bias), noise_scale)

    def base(self, enc):
        return enc["center"] if self.kind == "control" else enc["q50"]

    def sample(self, enc, eps, need_grad=False):
        """ppm samples [B,S,4,14]; cache for backward if requested."""
        g, cache = self.head.forward(enc["features"], eps)
        if self.kind == "residual_centered":
            med = np.median(g, axis=1, keepdims=True)
            g_used = g - med
        else:
            g_used = g
        x = self.base(enc)[:, None] + enc["scale"][:, None] * g_used
        return (x, (g, cache)) if need_grad else x

    def backward(self, dx, enc, state):
        g, cache = state
        dg = dx * enc["scale"][:, None]
        if self.kind == "residual_centered":
            dg = dg - median_vjp(g, dg.sum(1, keepdims=True))
        return self.head.backward(dg, cache)


def median_vjp(g, upstream):
    """Vector-Jacobian product of np.median over axis 1 (keepdims) w.r.t. g.

    For even S the median is the mean of the two middle order statistics; for odd S
    it is the single middle one. Ties receive the subgradient of the selected index.
    """
    S = g.shape[1]
    order = np.argsort(g, axis=1, kind="stable")
    out = np.zeros_like(g)
    picks = [S // 2 - 1, S // 2] if S % 2 == 0 else [S // 2]
    w = 1.0 / len(picks)
    for p in picks:
        idx = np.take(order, [p], axis=1)
        np.put_along_axis(out, idx, np.take_along_axis(out, idx, 1) + w * upstream, axis=1)
    return out


def energy_score_and_grad(x, y, mask, pair_weight=0.5, need_grad=True):
    """Masked, dimension-normalized sample ES (window-mean), matching uq14.losses.

    x [B,S,4,14] ppm samples; y, mask [B,4,14]. Returns (loss, dL/dx).
    """
    B, S = x.shape[:2]
    m = mask[:, None]
    xs = np.where(m, x, 0.0).reshape(B, S, -1)
    ys = np.where(mask, y, 0.0).reshape(B, 1, -1)
    n = mask.reshape(B, -1).sum(1).astype(np.float64)
    diff = xs - ys
    dist = np.linalg.norm(diff, axis=-1)                      # [B,S]
    first = dist.mean(1)
    pd = xs[:, :, None] - xs[:, None]                          # [B,S,S,D]
    pdist = np.linalg.norm(pd, axis=-1)                        # [B,S,S]
    pairs = pdist.sum((1, 2)) / 2                              # unordered pairs
    per = (first - 2 * pair_weight * pairs / (S * (S - 1))) / np.sqrt(n)
    loss = per.mean()
    if not need_grad:
        return loss, None
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = np.where(dist[..., None] > 0, diff / dist[..., None], 0.0)
        u2 = np.where(pdist[..., None] > 0, pd / pdist[..., None], 0.0)
    coef = 1.0 / (np.sqrt(n) * B)
    grad = (u1 / S - 2 * pair_weight * u2.sum(2) / (S * (S - 1))) * coef[:, None, None]
    grad = grad.reshape(x.shape) * m
    return loss, grad


class AdamW:
    """torch.optim.AdamW (defaults betas=(.9,.999), eps=1e-8, decoupled decay)."""

    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, weight_decay, *betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, p in params.items():
            p *= 1 - self.lr * self.wd
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            p -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


def clip_grad_norm(grads, max_norm):
    total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if not np.isfinite(total):
        raise FloatingPointError("Nonfinite gradient norm.")
    coef = min(1.0, max_norm / (total + 1e-6))
    for g in grads.values():
        g *= coef
    return total
