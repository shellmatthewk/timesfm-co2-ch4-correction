"""Step 3, round 2: an explicit CO2-CH4 linkage strength r (no conformal calibration, no empirical copula).

The main network is round-1 A's (rq.engression_np Model 'residual_centered' with a StationHead;
same inputs, loss, hyperparameters, folds, seeds and test windows). Only the noise coupling changes.
Every slot gets a full 1280-dim standard normal vector, drawn exactly as B's noise, and at station s

    CH4 noise = r_s * CO2 noise + sqrt(1 - r_s^2) * the CH4 slot's own noise.

Each slot's noise stays standard normal, so the single-cell distributions do not depend on r;
r_s only sets how the two gases move together (at the warm start the correlation of the two
gases' samples on every day equals r_s exactly). r_s = tanh(u_s) comes from a small head:

  E : z_s = [sin(month), cos(month), 14-day history co-movement of s, is_BRW, is_MLO]
  F : z_s = [is_BRW, is_MLO]                         (one learned r per station)

The head's output layer starts at 0, so r = 0 and the model starts identical to B.

Round 3 (G, H) = E, F trained with L = ES56 + lambda * ES2, where ES2 is the mean 2-dimensional
Energy Score of the normalized (CO2, CH4) pairs at the same station and day (divided by sqrt 2).

Usage (PYTHONPATH=src): python -m mg.linkage train {E,F,G,H} YEAR SEED
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np

from . import multigas as mg
from .data import ROOT, sha, write_json

sys.path.insert(0, str(ROOT / "src"))
from rq.engression_np import Model, AdamW, clip_grad_norm, energy_score_and_grad, silu     # noqa: E402
from rq.station_noise import StationHead                                                  # noqa: E402

NAMES = {"E": "E_explicit_conditional", "F": "F_explicit_constant",
         "G": "G_explicit_conditional_pairloss", "H": "H_explicit_constant_pairloss"}
Z_OF = {"E": "E", "F": "F", "G": "E", "H": "F"}      # which r-head inputs
PAIR_LOSS = {"G", "H"}
LAMBDA = 1.0
R_CFG = {"hidden": 16, "lr": 2e-3, "weight_decay": 0.01, "init_stream_offset": 7_000_000}
PROBE_MONTHS = (1, 4, 7, 10)


class RHead:
    """r = tanh(w2 . silu(W1 z + b1) + b2) for each window and station; z [B,2,in] -> r [B,2]."""

    names = ("W1", "b1", "w2", "b2")

    def __init__(self, in_dim, hidden, rng):
        k = 1 / np.sqrt(in_dim)
        self.W1, self.b1 = rng.uniform(-k, k, (hidden, in_dim)), rng.uniform(-k, k, hidden)
        self.w2, self.b2 = np.zeros(hidden), np.zeros(1)

    def params(self):
        return {n: getattr(self, n) for n in self.names}

    def set_params(self, p):
        for n in self.names:
            setattr(self, n, np.array(p[n], dtype=np.float64, copy=True))

    def forward(self, z):
        a = z @ self.W1.T + self.b1
        h, s = silu(a)
        return np.tanh(h @ self.w2 + self.b2[0]), (z, a, h, s)

    def backward(self, du, cache):
        """du = dL/du [B,2] (u the pre-tanh value)."""
        z, a, h, s = cache
        g = {"w2": np.einsum("bs,bsh->h", du, h), "b2": np.array([du.sum()])}
        da = (du[..., None] * self.w2) * (s * (1 + a * (1 - s)))
        g["W1"] = np.einsum("bsh,bsi->hi", da, z)
        g["b1"] = da.sum((0, 1))
        return g


class CouplingStationHead(StationHead):
    """StationHead whose backward also keeps dL/d(noise) of the two CH4 slots (for the r gradient)."""

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
        g["We"] = np.einsum("bsch,bscz->hz", da1, eps)
        self.d_eps_ch4 = da1[:, :, 2:] @ self.We
        return g


def build(seed, feature_dim):
    """Same initialisation as round-1 A/B (same class hierarchy, same random stream)."""
    head = CouplingStationHead(feature_dim, mg.CFG["own_noise"] + mg.CFG["second_noise"], mg.CFG["hidden_dim"],
                               np.random.default_rng(seed))
    return Model("residual_centered", head, np.zeros((14, feature_dim)), np.zeros(14), mg.CFG["noise_initial_std_normalized"])


def z_inputs(variant, idx):
    d = mg.data()
    n = len(idx)
    onehot = np.broadcast_to(np.eye(2)[None], (n, 2, 2))
    if variant == "F":
        return np.array(onehot, dtype=np.float64)
    angle = 2 * np.pi * (d["months"][idx] - 0.5) / 12
    season = np.broadcast_to(np.stack([np.sin(angle), np.cos(angle)], -1)[:, None], (n, 2, 2))
    hist = d["history_comovement"][idx][..., None]
    return np.concatenate([season, hist, onehot], -1).astype(np.float64)


def coupled(base, r):
    """base [B,S,4,Z] independent standard normal (slots BRW-CO2, MLO-CO2, BRW-CH4, MLO-CH4); r [B,2]."""
    eps = base.copy()
    eps[:, :, 2:] = r[:, None, :, None] * base[:, :, :2] + np.sqrt(1 - r ** 2)[:, None, :, None] * base[:, :, 2:]
    return eps


def du_from_deps(d_eps_ch4, base, r):
    """dL/du [B,2] for r = tanh(u), given dL/d(CH4 noise) [B,S,2,Z]."""
    one_minus = 1 - r ** 2
    deps_du = one_minus[:, None, :, None] * base[:, :, :2] - (r * np.sqrt(one_minus))[:, None, :, None] * base[:, :, 2:]
    return (d_eps_ch4 * deps_du).sum((1, 3))


def pair_es(xn, yn, mask, need_grad=True):
    """Mean 2-D Energy Score of the normalized (CO2, CH4) pairs at the same station and day with both
    labels (each divided by sqrt 2, pair weight 1/2, as in ES56). xn [B,S,4,14], yn and mask [B,4,14]."""
    b_i, s_i, t_i = np.nonzero(mask[:, :2] & mask[:, 2:])
    if not len(b_i):
        return 0.0, (np.zeros_like(xn) if need_grad else None)
    x2 = np.stack([xn[b_i, :, s_i, t_i], xn[b_i, :, s_i + 2, t_i]], -1)[..., None]      # [P,S,2,1]
    y2 = np.stack([yn[b_i, s_i, t_i], yn[b_i, s_i + 2, t_i]], -1)[..., None]            # [P,2,1]
    loss, g2 = energy_score_and_grad(x2, y2, np.ones(y2.shape, bool), 0.5, need_grad)
    if not need_grad:
        return loss, None
    dx = np.zeros_like(xn)
    dx[b_i, :, s_i, t_i] = g2[:, :, 0, 0]
    dx[b_i, :, s_i + 2, t_i] = g2[:, :, 1, 0]
    return loss, dx


def combined_loss(x, y, mask, c, chunk=8):
    """Validation value of L = ES56 + lambda * ES2 (ES56 window mean, ES2 mean over all pairs), in chunks."""
    es56, num2, cnt2 = 0.0, 0.0, 0
    for i in range(0, len(x), chunk):
        xn, yn, m = x[i:i + chunk] / c[None, None, :, None], y[i:i + chunk] / c[None, :, None], mask[i:i + chunk]
        es56 += energy_score_and_grad(xn, yn, m, 0.5, need_grad=False)[0] * len(xn)
        p = int((m[:, :2] & m[:, 2:]).sum())
        if p:
            num2 += pair_es(xn, yn, m, need_grad=False)[0] * p
            cnt2 += p
    return es56 / len(x) + LAMBDA * num2 / max(cnt2, 1)


def predict(model, rhead, e, z, n_samples, seed, chunk=4):
    rng = np.random.default_rng(seed)
    r, _ = rhead.forward(z)
    out = []
    for i in range(0, len(e["features"]), chunk):
        part = {k: v[i:i + chunk] for k, v in e.items()}
        base = mg.noise(rng, len(part["features"]), n_samples, "B")
        out.append(model.sample(part, coupled(base, r[i:i + chunk])))
    return np.concatenate(out), r


def probe(variant, rhead, hist_median):
    """r by station at a few months (history co-movement at its training median), for the log."""
    if Z_OF[variant] == "F":
        r, _ = rhead.forward(np.eye(2)[None])
        return {st: [float(r[0, i])] for i, st in enumerate(("BRW", "MLO"))}
    out = {}
    for i, st in enumerate(("BRW", "MLO")):
        rows = []
        for m in PROBE_MONTHS:
            angle = 2 * np.pi * (m - 0.5) / 12
            z = np.zeros((1, 2, 5))
            z[0, :, :2] = np.sin(angle), np.cos(angle)
            z[0, :, 2] = hist_median
            z[0, :, 3:] = np.eye(2)
            rows.append(float(rhead.forward(z)[0][0, i]))
        out[st] = rows
    return out


def train(variant, year, seed):
    name = NAMES[variant]
    out = ROOT / "04_训练" / f"multigas_{name}" / f"fold{year}_seed{seed}"
    f = mg.fold(year)
    c = mg.norm_constants(f["train"])
    d = mg.data()
    tr, va_idx = f["train"], f["validation"][::mg.CFG["validation_stride_days"]]
    e_tr, e_va = mg.enc("A", tr), mg.enc("A", va_idx)
    z_tr, z_va = z_inputs(Z_OF[variant], tr), z_inputs(Z_OF[variant], va_idx)
    pair = variant in PAIR_LOSS
    y, mask = d["y"][tr], d["mask"][tr]
    hist_median = np.median(d["history_comovement"][tr], axis=0)
    model = build(seed, e_tr["features"].shape[-1])
    rhead = RHead(z_tr.shape[-1], R_CFG["hidden"], np.random.default_rng(seed + R_CFG["init_stream_offset"]))
    params, rparams = model.head.params(), rhead.params()
    if (out / "complete.json").exists():
        done = json.loads((out / "complete.json").read_text())
        model.head.set_params(np.load(out / "best_head.npz"))
        rhead.set_params(np.load(out / "best_rhead.npz"))
    else:
        out.mkdir(parents=True, exist_ok=True)
        opt = AdamW(params, mg.CFG["head_lr"], mg.CFG["weight_decay"])
        ropt = AdamW(rparams, R_CFG["lr"], R_CFG["weight_decay"])
        scale4 = c[None, None, :, None]

        def validate():
            x, _ = predict(model, rhead, e_va, z_va, mg.CFG["validation_samples"], seed + mg.OFFSETS["validation"])
            if pair:
                return float(combined_loss(x, d["y"][va_idx], d["mask"][va_idx], c))
            return float(mg.normalized_es(x, d["y"][va_idx], d["mask"][va_idx], c))

        def r_text(p):
            return " ".join(f"{st} " + " ".join(f"{v:+.2f}" for v in vals) for st, vals in p.items())
        best = validate()
        best_epoch, bad = 0, 0
        hist = [{"epoch": 0, "validation_es": best, "r_probe": probe(variant, rhead, hist_median)}]
        best_params = {k: p.copy() for k, p in params.items()}
        best_rparams = {k: p.copy() for k, p in rparams.items()}
        t0 = time.time()
        for epoch in range(1, mg.CFG["epochs"] + 1):
            order = np.random.default_rng(seed + epoch).permutation(len(tr))
            rng = np.random.default_rng(seed * 10000 + epoch)
            total = 0.0
            for i in range(0, len(order), mg.CFG["batch_size"]):
                ids = order[i:i + mg.CFG["batch_size"]]
                part = {k: v[ids] for k, v in e_tr.items()}
                r, rcache = rhead.forward(z_tr[ids])
                base = mg.noise(rng, len(ids), mg.CFG["train_samples"], "B")
                x, state = model.sample(part, coupled(base, r), need_grad=True)
                loss, dxn = energy_score_and_grad(x / scale4, y[ids] / c[None, :, None], mask[ids], 0.5)
                if pair:
                    loss2, dx2 = pair_es(x / scale4, y[ids] / c[None, :, None], mask[ids])
                    loss, dxn = loss + LAMBDA * loss2, dxn + LAMBDA * dx2
                grads = model.backward(dxn / scale4, part, state)
                rgrads = rhead.backward(du_from_deps(model.head.d_eps_ch4, base, r), rcache)
                clip_grad_norm({**{f"m.{k}": v for k, v in grads.items()}, **{f"r.{k}": v for k, v in rgrads.items()}},
                               mg.CFG["grad_clip"])
                opt.step(params, grads)
                ropt.step(rparams, rgrads)
                total += loss * len(ids)
            val = validate()
            if val < best - 1e-9:
                best, best_epoch, bad = val, epoch, 0
                best_params = {k: p.copy() for k, p in params.items()}
                best_rparams = {k: p.copy() for k, p in rparams.items()}
            else:
                bad += 1
            p = probe(variant, rhead, hist_median)
            hist.append({"epoch": epoch, "train_es": total / len(tr), "validation_es": val, "r_probe": p})
            print(f"multigas_{name} fold{year} seed{seed} ep {epoch} train {total/len(tr):.4f} val {val:.4f} "
                  f"best {best_epoch} ({time.time()-t0:.0f}s) | r {r_text(p)}", flush=True)
            if bad >= mg.CFG["patience"]:
                break
        np.savez_compressed(out / "best_head.npz", **best_params)
        np.savez_compressed(out / "best_rhead.npz", **best_rparams)
        write_json(out / "history.json", hist)
        done = {"variant": variant, "fold": year, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                "best_validation_es": best, "seconds": time.time() - t0, "config": mg.CFG, "r_config": R_CFG,
                "loss": "ES56 + %g * ES2" % LAMBDA if pair else "ES56",
                "training_windows": int(len(tr)), "validation_windows": int(len(va_idx)),
                "input_dim": int(e_tr["features"].shape[-1]), "r_input_dim": int(z_tr.shape[-1]),
                "checkpoint_sha256": sha(out / "best_head.npz"), "r_checkpoint_sha256": sha(out / "best_rhead.npz"),
                "periods_read_for_training": ["train", "validation"]}
        write_json(out / "complete.json", done)
        model.head.set_params(best_params)
        rhead.set_params(best_rparams)
    te = f["test"]
    samples, r_test = predict(model, rhead, mg.enc("A", te), z_inputs(Z_OF[variant], te), mg.CFG["eval_samples"],
                              seed + mg.OFFSETS["test"])
    point_gap = float(np.abs(np.median(samples, 1) - d["q50"][te]).max())
    mg.evaluate(name, year, seed, samples, te, c, {"selected_epoch": done["best_epoch"], "training": done,
                                                   "point_minus_native_median_max_abs": point_gap})
    np.savez_compressed(ROOT / "06_结果" / f"multigas_{name}" / f"fold{year}_seed{seed}" / "linkage_r.npz",
                        r=r_test, origins=d["origins"][te], months=d["months"][te],
                        history_comovement=d["history_comovement"][te])


if __name__ == "__main__":
    if sys.argv[1] != "train":
        raise SystemExit("unknown stage")
    train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
