"""Scores for step 3 (slots 0 BRW-CO2, 1 MLO-CO2, 2 BRW-CH4, 3 MLO-CH4).

Joint quantities use normalized deviations from the TimesFM median,
z = (value - median) / c_slot, with c_slot a fixed per-slot constant from the
training years, so that CO2 (ppm) and CH4 (ppb) are comparable and no station
dominates. Only cells with an observed label enter. Each metric is returned per
window as (num, den); the reported value is num.sum() / den.sum(), so the paired
bootstrap is a plain resampling of windows (or of blocks of windows).

Cross-gas quantities use the same station and day:
  gas_difference  d = z_CO2 - z_CH4: CRPS, 80% interval coverage / width / MIS
  gas_sum         s = z_CO2 + z_CH4: same (supplementary)
  both_above_brier  Brier score of "both gases above their TimesFM median"
  quadrant_brier  Brier score over the four sign patterns (supplementary)
  variogram_cross_gas  variogram score (p = 0.5) over the (CO2, CH4) pairs
  band_two_gas    simultaneous 80% band over the two gases: coverage and width
Overall: energy_score of the normalized observed vector (sqrt(dimension)-normalized)
and variogram_within_series (same slot, different days).
Single cells (original units, per slot): cell_{crps, mis80, coverage80, width80, abs_error, n} [B,4].
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cop.joint_metrics import crps_units, interval_units, simultaneous_band      # noqa: E402

GAS_PAIRS = ((0, 2), (1, 3))
CROSS = ("variogram_cross_gas",
         "gas_difference_crps", "gas_difference_coverage80", "gas_difference_width80", "gas_difference_mis80",
         "gas_sum_crps", "gas_sum_coverage80", "gas_sum_width80", "gas_sum_mis80",
         "both_above_brier", "quadrant_brier", "band_two_gas_coverage80", "band_two_gas_width80")
JOINT = ("energy_score", "variogram_within_series") + CROSS + tuple(f"{s}_{k}" for s in ("BRW", "MLO") for k in CROSS)


def normalize(samples, y, q50, c):
    cc = np.asarray(c, dtype=np.float64)
    zs = (samples - q50[:, None]) / cc[None, None, :, None]
    zy = (y - q50) / cc[None, :, None]
    return zs, zy


def cell_per_window(samples, y, mask):
    """Single-cell scores per window and slot, original units: {name: [B,4]}."""
    n_b, s, n_k, n_h = samples.shape
    b_i, k_i, h_i = np.nonzero(mask)
    x = samples[b_i, :, k_i, h_i]                          # [N,S]
    yy = y[b_i, k_i, h_i]
    covered, width, score = interval_units(x, yy)
    values = {"crps": crps_units(x, yy), "mis80": score, "coverage80": covered, "width80": width,
              "abs_error": np.abs(np.median(x, axis=1) - yy), "n": np.ones(len(yy))}
    out = {}
    for name, v in values.items():
        a = np.zeros((n_b, n_k))
        np.add.at(a, (b_i, k_i), v)
        out[f"cell_{name}"] = a
    return out


def all_multigas_per_window(samples, y, mask, q50, c, p=0.5):
    zs, zy = normalize(np.asarray(samples, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(q50), c)
    n_b = len(zy)
    num = {k: np.zeros(n_b) for k in JOINT}
    den = {k: np.zeros(n_b) for k in JOINT}
    for b in range(n_b):
        m = mask[b]
        x = zs[b]                                    # [S,4,14]
        s = len(x)
        flat_m = m.ravel()
        xf = x.reshape(s, -1)[:, flat_m]
        yf = zy[b].ravel()[flat_m]
        first = np.linalg.norm(xf - yf, axis=1).mean()
        num["energy_score"][b] = (first - pdist(xf).sum() / (s * (s - 1))) / np.sqrt(flat_m.sum())
        den["energy_score"][b] = 1.0
        vals = []
        for k in range(4):                           # same slot, different days
            days = np.flatnonzero(m[k])
            if len(days) < 2:
                continue
            i, j = np.triu_indices(len(days), 1)
            fc = (np.abs(x[:, k, days[i]] - x[:, k, days[j]]) ** p).mean(0)
            ob = np.abs(zy[b, k, days[i]] - zy[b, k, days[j]]) ** p
            vals.append((ob - fc) ** 2)
        if vals:
            num["variogram_within_series"][b], den["variogram_within_series"][b] = np.concatenate(vals).mean(), 1.0
        for prefix, pairs in (("", GAS_PAIRS), ("BRW_", GAS_PAIRS[:1]), ("MLO_", GAS_PAIRS[1:])):
            units = [(k1, k2, t) for k1, k2 in pairs for t in np.flatnonzero(m[k1] & m[k2])]
            if units:
                for key, (nu, de) in _cross(x, zy[b], units, p).items():
                    num[prefix + key][b], den[prefix + key][b] = nu, de
    return {k: (num[k], den[k]) for k in JOINT}


def _cross(x, zy, units, p):
    """Cross-gas (num, den) of one window over its (CO2 slot, CH4 slot, day) units."""
    a = np.stack([x[:, k1, t] for k1, _, t in units], 1)          # [S,U] CO2
    c2 = np.stack([x[:, k2, t] for _, k2, t in units], 1)         # [S,U] CH4
    ya = np.array([zy[k1, t] for k1, _, t in units])
    yb = np.array([zy[k2, t] for _, k2, t in units])
    n = len(units)
    out = {}
    for key, xs, obs in (("gas_difference", a - c2, ya - yb), ("gas_sum", a + c2, ya + yb)):
        crps = crps_units(xs.T, obs)
        cov, width, score = interval_units(xs.T, obs)
        for suffix, v in (("crps", crps), ("coverage80", cov), ("width80", width), ("mis80", score)):
            out[f"{key}_{suffix}"] = (float(v.sum()), n)
    pa, pb = a > 0, c2 > 0
    probs = np.stack([(pa & pb).mean(0), (pa & ~pb).mean(0), (~pa & pb).mean(0), (~pa & ~pb).mean(0)], 1)
    oa, ob_ = ya > 0, yb > 0
    obs = np.stack([oa & ob_, oa & ~ob_, ~oa & ob_, ~oa & ~ob_], 1).astype(float)
    out["both_above_brier"] = (float(((probs[:, 0] - obs[:, 0]) ** 2).sum()), n)
    out["quadrant_brier"] = (float(((probs - obs) ** 2).sum()), n)
    vc = (np.abs(ya - yb) ** p - (np.abs(a - c2) ** p).mean(0)) ** 2
    out["variogram_cross_gas"] = (float(vc.mean()), 1.0)
    cov_b = wid_b = 0.0
    for u in range(n):
        lo, hi = simultaneous_band(np.stack([a[:, u], c2[:, u]], 1))
        cov_b += float((lo[0] <= ya[u] <= hi[0]) and (lo[1] <= yb[u] <= hi[1]))
        wid_b += float((hi - lo).mean())
    out["band_two_gas_coverage80"], out["band_two_gas_width80"] = (cov_b, n), (wid_b, n)
    return out


def _normal_scores(x, axis):
    r = rankdata(x, method="average", axis=axis)
    return norm.ppf(r / (x.shape[axis] + 1))


def implied_and_observed(samples, y, mask):
    """implied_correlation [B,2,14]: CO2-CH4 correlation of the normal scores of the sample
    ranks (per window, station and day; NaN if a gas has no spread);
    observed_pit_score [B,4,14]: normal score of the observed value within the samples (NaN if missing)."""
    x = np.asarray(samples, dtype=np.float64)
    n_b, s = x.shape[:2]
    implied = np.full((n_b, 2, 14), np.nan)
    pit = np.full((n_b, 4, 14), np.nan)
    for b in range(n_b):
        below = (x[b] < y[b][None]).sum(0) + 0.5 * (x[b] == y[b][None]).sum(0)        # [4,14]
        z = norm.ppf((below + 0.5) / (s + 1))
        pit[b][mask[b]] = z[mask[b]]
        ns = _normal_scores(x[b], axis=0)                                           # [S,4,14]
        for st, (k1, k2) in enumerate(GAS_PAIRS):
            u, v = ns[:, k1] - ns[:, k1].mean(0), ns[:, k2] - ns[:, k2].mean(0)
            den = np.sqrt((u ** 2).sum(0) * (v ** 2).sum(0))
            with np.errstate(invalid="ignore", divide="ignore"):
                implied[b, st] = np.where(den > 1e-12, (u * v).sum(0) / den, np.nan)
    return {"implied_correlation": implied, "observed_pit_score": pit}
