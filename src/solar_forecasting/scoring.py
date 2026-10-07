"""Observed-cell scores and the CO2 workflow's paired month-block bootstrap."""
from __future__ import annotations

import numpy as np

METRICS = ("mis80", "coverage80", "width80", "crps", "mae", "squared_error")


def per_window(samples, target, mask):
    # [window, sample, channel, lead] -> score sums [window, channel].
    b, s, c, _ = samples.shape
    result = {name: np.zeros((b, c), dtype=np.float64) for name in METRICS}
    result["n"] = mask.sum(-1).astype(np.float64)
    for channel in range(c):
        w, lead = np.nonzero(mask[:, channel])
        if not len(w):
            continue
        x, y = samples[w, :, channel, lead], target[w, channel, lead]
        low, median, high = np.quantile(x, [0.1, 0.5, 0.9], axis=1)
        width = high - low
        ordered = np.sort(x, axis=1)
        weights = 2 * np.arange(1, s + 1) - s - 1
        crps = np.abs(x - y[:, None]).mean(1) - (ordered * weights).sum(1) / (s * (s - 1))
        values = {"mis80": width + 10 * np.maximum(low - y, 0) + 10 * np.maximum(y - high, 0),
                  "coverage80": ((y >= low) & (y <= high)).astype(float), "width80": width,
                  "crps": crps, "mae": np.abs(median - y), "squared_error": (median - y) ** 2}
        for name, val in values.items():
            np.add.at(result[name][:, channel], w, val)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Nonfinite scores")
    return result


def pooled(scores, channel):
    n = scores["n"][:, channel].sum()
    if not n:
        return {"n_cells": 0, **{m: None for m in (*METRICS[:-1], "rmse")}}
    values = {m: float(scores[m][:, channel].sum() / n) for m in METRICS[:-1]}
    values.update({"rmse": float(np.sqrt(scores["squared_error"][:, channel].sum() / n)), "n_cells": int(n)})
    return values


def bootstrap_difference(a, b, n, origins, draws, seed):
    labels = origins.astype("datetime64[M]")
    blocks, block_of = np.unique(labels, return_inverse=True)
    difference = float((a.sum() - b.sum()) / n.sum()) if n.sum() else None
    if len(blocks) < 2 or not n.sum():
        return {"difference": difference, "ci95": None, "month_blocks": len(blocks)}
    numerator = np.bincount(block_of, weights=a - b, minlength=len(blocks))
    denominator = np.bincount(block_of, weights=n, minlength=len(blocks))
    rng = np.random.default_rng(seed)
    reps = []
    # Bound memory even when users increase the data period or bootstrap count.
    for start in range(0, draws, 256):
        take = rng.integers(0, len(blocks), (min(256, draws - start), len(blocks)))
        den = denominator[take].sum(1)
        reps.extend((numerator[take].sum(1)[den > 0] / den[den > 0]).tolist())
    interval = np.percentile(reps, [2.5, 97.5]).tolist() if reps else None
    return {"difference": difference, "ci95": interval, "month_blocks": len(blocks)}
