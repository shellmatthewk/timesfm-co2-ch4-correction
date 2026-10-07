"""The CO2 experiment's five forecast families, generalized to C channels and H steps."""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch import nn

from .config import ROOT

sys.path.insert(0, str(ROOT / "multigas/src"))
from cop.marginal import icdf_from_z  # noqa: E402


def native_samples(quantiles, samples, generator):
    b, c, h, _ = quantiles.shape
    z = torch.randn(b, samples, c, h, generator=generator, dtype=quantiles.dtype)
    return icdf_from_z(quantiles, z)


class ForecastHead(nn.Module):
    def __init__(self, method, feature_dim, context_dim, horizon, cfg, seed, native_weight=None, native_bias=None):
        super().__init__()
        torch.manual_seed(seed)
        self.method, self.horizon = method, horizon
        self.noise_dim = cfg["noise_dim"]
        if method in ("basic", "residual", "N1"):
            # Paired SiLU identity needs 2*H units. Increase capacity when H grows.
            hidden = max(cfg["hidden_dim"], 2 * horizon)
            self.hidden_dim = hidden
            self.l1 = nn.Linear(feature_dim + self.noise_dim, hidden)
            self.l2 = nn.Linear(hidden, hidden)
            self.l3 = nn.Linear(hidden, horizon)
            self.feature_dim = feature_dim
            weight = torch.as_tensor(native_weight) if method == "basic" else torch.zeros(horizon, feature_dim)
            bias = torch.as_tensor(native_bias) if method == "basic" else torch.zeros(horizon)
            with torch.no_grad():
                a = torch.zeros(horizon, self.noise_dim)
                a[:, :horizon] = torch.eye(horizon) * cfg["initial_noise_scale"]
                self.l1.weight[:horizon] = torch.cat([weight, a], -1)
                self.l1.weight[horizon:2 * horizon] = -self.l1.weight[:horizon]
                self.l1.bias[:horizon], self.l1.bias[horizon:2 * horizon] = bias, -bias
                self.l2.weight[:2 * horizon] = 0
                eye = torch.eye(horizon)
                self.l2.weight[:horizon, :horizon], self.l2.weight[:horizon, horizon:2 * horizon] = eye, -eye
                self.l2.weight[horizon:2 * horizon, :horizon], self.l2.weight[horizon:2 * horizon, horizon:2 * horizon] = -eye, eye
                self.l2.bias[:2 * horizon] = 0
                self.l3.weight.zero_()
                self.l3.weight[:, :horizon], self.l3.weight[:, horizon:2 * horizon] = eye, -eye
                self.l3.bias.zero_()
        if method in ("T1", "N1"):
            self.rule = nn.Sequential(nn.Linear(context_dim, cfg["rule_hidden_dim"]), nn.SiLU(), nn.Linear(cfg["rule_hidden_dim"], 1))
            nn.init.zeros_(self.rule[-1].weight)
            nn.init.zeros_(self.rule[-1].bias)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("rule.")]

    def rule_parameters(self):
        return list(self.rule.parameters()) if hasattr(self, "rule") else []

    def forward(self, e, context, samples, generator):
        q = e["quantiles"]
        median = q[..., 4]
        if self.method == "T1":
            amplitude = self.rule(context)[..., 0].clamp(-3, 3).exp()
            return median[:, None] + amplitude[:, None, :, None] * (native_samples(q, samples, generator) - median[:, None])
        h = e["features"]
        b, c, _ = h.shape
        # The basic CO2 head uses shared noise; centered heads use one vector per channel.
        shape = (b, samples, 1 if self.method == "basic" else c, self.noise_dim)
        noise = torch.randn(shape, generator=generator, dtype=h.dtype).expand(b, samples, c, self.noise_dim)
        if self.method == "N1":
            amplitude = self.rule(context)[..., 0].clamp(-3, 3).exp()
            noise = amplitude[:, None, :, None] * noise
        w = self.l1.weight
        a = (h @ w[:, :self.feature_dim].T)[:, None] + noise @ w[:, self.feature_dim:].T + self.l1.bias
        g = self.l3(torch.nn.functional.silu(self.l2(torch.nn.functional.silu(a))))
        if self.method != "basic":
            g = g - torch.quantile(g, 0.5, dim=1, keepdim=True)
        centre = e["center"] if self.method == "basic" else median
        return centre[:, None] + e["scale"][:, None] * g


def energy_score(x, y, mask, scale):
    n = mask.flatten(1).sum(1)
    safe_y = torch.where(mask, y, torch.zeros_like(y))
    xx = (x / scale[None, None, :, None] * mask[:, None]).flatten(2)
    yy = (safe_y / scale[None, :, None]).flatten(1)[:, None]
    first = torch.linalg.vector_norm(xx - yy, dim=-1).mean(1)
    distances = torch.cdist(xx, xx, compute_mode="donot_use_mm_for_euclid_dist")
    s = x.shape[1]
    score = (first - distances.sum((1, 2)) / (2 * s * (s - 1))) / n.clamp_min(1).sqrt()
    return score[n > 0].mean()


def crps_loss(x, y, mask, scale):
    safe_y = torch.where(mask, y, torch.zeros_like(y))
    xx = x / scale[None, None, :, None]
    yy = safe_y / scale[None, :, None]
    s = x.shape[1]
    ordered = xx.sort(dim=1).values
    weights = (2 * torch.arange(1, s + 1, dtype=x.dtype) - s - 1)[None, :, None, None]
    score = (xx - yy[:, None]).abs().mean(1) - (ordered * weights).sum(1) / (s * (s - 1))
    return score[mask].mean()


def training_scale(y, mask, median):
    values = []
    for j in range(y.shape[1]):
        errors = (y[:, j] - median[:, j])[mask[:, j]]
        if not len(errors):
            raise ValueError(f"Channel {j} has no observed training targets")
        values.append(max(float(np.sqrt(np.mean(errors ** 2))), 1e-6))
    return np.asarray(values, dtype=np.float32)


def raw_context(windows, encoded, calendar=False):
    q = encoded["quantiles"]
    width = np.log(np.maximum((q[..., 8] - q[..., 0]).mean(-1), 1e-6))
    scale = np.log(np.maximum(encoded["scale"][..., 0], 1e-6))
    dx = np.diff(windows["x"], axis=-1)
    good = ~(windows["imputed"][..., 1:] | windows["imputed"][..., :-1])
    vol = np.full(width.shape, np.nan)
    for b, c in np.ndindex(width.shape):
        if good[b, c].sum() >= 3:
            vol[b, c] = np.log(max(float(dx[b, c][good[b, c]].std()), 1e-6))
    pieces = [width[..., None], scale[..., None], vol[..., None]]
    if calendar:
        dates = windows["target_times"][:, 0].astype("datetime64[D]")
        day = (dates - dates.astype("datetime64[Y]")).astype(int)
        angle = 2 * np.pi * day / 365.25
        seasonal = np.stack([np.sin(angle), np.cos(angle)], -1)
        pieces.append(np.broadcast_to(seasonal[:, None], (*width.shape, 2)))
    return np.concatenate(pieces, -1).astype(np.float32)


def fit_context(raw, training_idx):
    flat = raw[training_idx].reshape(-1, raw.shape[-1])
    fill = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in flat.T])
    filled = np.where(np.isfinite(flat), flat, fill)
    return {"fill": fill, "mean": filled.mean(0), "std": np.maximum(filled.std(0), 1e-6)}


def apply_context(raw, stats):
    filled = np.where(np.isfinite(raw), raw, stats["fill"])
    standardized = (filled - stats["mean"]) / stats["std"]
    b, c, _ = raw.shape
    return np.concatenate([standardized, np.broadcast_to(np.eye(c)[None], (b, c, c))], -1).astype(np.float32)
