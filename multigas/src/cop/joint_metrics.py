"""Joint (multi-cell) scores on the observed sub-vector of each test window.

No test window has all 56 labels, so every score uses only observed cells and
nothing is imputed. Each metric is returned per window as a pair (num, den);
the reported value is num.sum() / den.sum(). Window-mean scores (ES, variogram)
use den = 1 per window; pooled scores use den = number of units in the window.
This layout makes the paired window bootstrap a plain resampling of rows.

samples [B,S,C,H], target [B,C,H] (NaN where missing), mask [B,C,H] bool.

Variogram score: Scheuerer and Hamill (2015), order p, unit weights, reported
as the mean over valid pairs. Simultaneous bands: the 80% of trajectories with
the largest rank depth (minimum over cells of the two-sided rank) define the
band, so at least 80% of the sampled trajectories lie inside it on all cells.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.spatial.distance import pdist

LEVEL = 0.8


def crps_units(x, y):
    """Unbiased sample CRPS. x [U,S], y [U] -> [U]."""
    s = x.shape[1]
    ordered = np.sort(x, axis=1)
    weights = 2 * np.arange(1, s + 1) - s - 1
    return np.abs(x - y[:, None]).mean(1) - (ordered * weights).sum(1) / (s * (s - 1))


def interval_units(x, y, level=LEVEL):
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2], axis=1)
    width = hi - lo
    score = width + (2 / (1 - level)) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return ((lo <= y) & (y <= hi)).astype(np.float64), width, score


def energy_per_window(samples, target, mask):
    """sqrt(observed dimension)-normalized ES, same definition as uq14.metrics."""
    out = np.empty(len(target))
    for b in range(len(target)):
        m = mask[b].ravel()
        x = samples[b].reshape(samples.shape[1], -1)[:, m]
        y = target[b].ravel()[m]
        s = len(x)
        first = np.linalg.norm(x - y, axis=1).mean()
        out[b] = (first - pdist(x).sum() / (s * (s - 1))) / np.sqrt(m.sum())
    return out


def _pairs(mask_b, kind):
    """Index pairs (flat station-major cells) with both labels observed."""
    c_n, h_n = mask_b.shape
    found = []
    if kind == "within_station":
        for c in range(c_n):
            days = np.flatnonzero(mask_b[c])
            found += [(c * h_n + i, c * h_n + j) for i, j in combinations(days, 2)]
    else:
        for h in range(h_n):
            stations = np.flatnonzero(mask_b[:, h])
            found += [(i * h_n + h, j * h_n + h) for i, j in combinations(stations, 2)]
    return np.asarray(found, dtype=np.int64).reshape(-1, 2)


def variogram_per_window(samples, target, mask, kind, p=0.5):
    num, den = np.zeros(len(target)), np.zeros(len(target))
    for b in range(len(target)):
        pairs = _pairs(mask[b], kind)
        if not len(pairs):
            continue
        x = samples[b].reshape(samples.shape[1], -1)
        y = target[b].ravel()
        forecast = (np.abs(x[:, pairs[:, 0]] - x[:, pairs[:, 1]]) ** p).mean(0)
        observed = np.abs(y[pairs[:, 0]] - y[pairs[:, 1]]) ** p
        num[b], den[b] = np.mean((observed - forecast) ** 2), 1.0
    return num, den


def _derived_units(samples, target, mask, kind):
    """Yield (window, x[S], y) for one derived scalar per unit."""
    n_b, _, c_n, h_n = samples.shape
    for b in range(n_b):
        if kind == "station_mean":            # mean over the observed days of one station
            for c in range(c_n):
                days = mask[b, c]
                if days.sum() >= 2:
                    yield b, samples[b][:, c][:, days].mean(1), target[b, c, days].mean()
        elif kind == "station_difference":    # same day, two stations
            for h in range(h_n):
                for i, j in combinations(np.flatnonzero(mask[b, :, h]), 2):
                    yield b, samples[b, :, i, h] - samples[b, :, j, h], target[b, i, h] - target[b, j, h]
        elif kind == "day_mean4":             # same day, mean of all four stations
            for h in range(h_n):
                if mask[b, :, h].all():
                    yield b, samples[b, :, :, h].mean(1), target[b, :, h].mean()
        else:
            raise ValueError(kind)


def derived_per_window(samples, target, mask, kind):
    units = list(_derived_units(samples, target, mask, kind))
    n_b = len(target)
    out = {k: np.zeros(n_b) for k in ("crps", "coverage80", "width80", "mis80", "n")}
    if not units:
        return out
    window = np.array([u[0] for u in units])
    x = np.stack([u[1] for u in units])
    y = np.array([u[2] for u in units])
    crps = crps_units(x, y)
    covered, width, score = interval_units(x, y)
    for name, value in (("crps", crps), ("coverage80", covered), ("width80", width),
                        ("mis80", score), ("n", np.ones(len(y)))):
        np.add.at(out[name], window, value)
    return out


def simultaneous_band(x, level=LEVEL):
    """x [S,K] -> (lower[K], upper[K]) containing >= level of the trajectories on all K cells."""
    s = len(x)
    rank = x.argsort(0, kind="stable").argsort(0, kind="stable")           # 0-based
    depth = np.minimum(rank, s - 1 - rank).min(1)
    k = int(np.sort(depth)[::-1][int(np.ceil(level * s)) - 1])
    ordered = np.sort(x, axis=0)
    return ordered[k], ordered[s - 1 - k]


def band_per_window(samples, target, mask, kind):
    """kind='station_14days': one band per (window, station) over its observed days;
    kind='day_4stations': one band per (window, day) over its observed stations."""
    n_b, _, c_n, h_n = samples.shape
    out = {k: np.zeros(n_b) for k in ("coverage80", "width80", "sample_inside", "n")}
    for b in range(n_b):
        if kind == "station_14days":
            groups = [(samples[b][:, c][:, mask[b, c]], target[b, c, mask[b, c]]) for c in range(c_n)]
        else:
            groups = [(samples[b][:, :, h][:, mask[b, :, h]], target[b, mask[b, :, h], h]) for h in range(h_n)]
        for x, y in groups:
            if x.shape[1] < 2:
                continue
            lo, hi = simultaneous_band(x)
            out["coverage80"][b] += float(np.all((lo <= y) & (y <= hi)))
            out["width80"][b] += float((hi - lo).mean())
            out["sample_inside"][b] += float(np.all((lo <= x) & (x <= hi), axis=1).mean())
            out["n"][b] += 1
    return out


def cell_per_window(samples, target, mask):
    """Single-cell scores in the same (num, den) layout, for the paired bootstrap."""
    n_b = len(target)
    out = {k: np.zeros(n_b) for k in ("abs_error", "crps", "coverage80", "width80", "mis80", "n")}
    for b in range(n_b):
        m = mask[b].ravel()
        x = samples[b].reshape(samples.shape[1], -1)[:, m].T
        y = target[b].ravel()[m]
        covered, width, score = interval_units(x, y)
        out["abs_error"][b] = np.abs(np.median(x, axis=1) - y).sum()
        out["crps"][b] = crps_units(x, y).sum()
        out["coverage80"][b], out["width80"][b], out["mis80"][b] = covered.sum(), width.sum(), score.sum()
        out["n"][b] = len(y)
    return out


def all_per_window(samples, target, mask, center=None, p=0.5):
    """Every joint metric as {name: (num[B], den[B])}.

    ``center`` [B,C,H] (the native TimesFM median, a function of the inputs
    only) gives the supplementary centered variogram scores, computed on
    y - center and x - center.
    """
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    ones = np.ones(len(target))
    out = {"energy_score": (energy_per_window(samples, target, mask), ones)}
    for kind in ("within_station", "across_station"):
        out[f"variogram_{kind}"] = variogram_per_window(samples, target, mask, kind, p)
        if center is not None:
            c = np.asarray(center, dtype=np.float64)
            out[f"variogram_centered_{kind}"] = variogram_per_window(
                samples - c[:, None], target - c, mask, kind, p)
    for kind in ("station_mean", "station_difference", "day_mean4"):
        d = derived_per_window(samples, target, mask, kind)
        for name in ("crps", "coverage80", "width80", "mis80"):
            out[f"{kind}_{name}"] = (d[name], d["n"])
    for kind in ("station_14days", "day_4stations"):
        d = band_per_window(samples, target, mask, kind)
        for name in ("coverage80", "width80", "sample_inside"):
            out[f"band_{kind}_{name}"] = (d[name], d["n"])
    d = cell_per_window(samples, target, mask)
    for name in ("abs_error", "crps", "coverage80", "width80", "mis80"):
        out[f"cell_{name}"] = (d[name], d["n"])
    return out


def value(pair, index=None):
    num, den = pair
    if index is not None:
        num, den = num[index], den[index]
    return float(num.sum() / den.sum()) if den.sum() > 0 else None
