"""Original-unit metrics with explicit missing-future-label masks.

Forecasts are [B,S,C,H], targets/masks [B,C,H], and quantiles [B,C,H,9].
Ordinary marginal metrics pool observed cells; ES averages dimension-normalized
observed-subvector scores across windows. Empty station/lead groups return None
and n=0. Overall all-missing windows are rejected. No predictions or labels are
silently treated as zero observations.

CRPS uses an unbiased sample U-statistic, evaluated by the sorted-sample
identity. Nine-quantile WIS is 2*mean(pinball), not a full-distribution CRPS.
Interval-only conformal results do not identify CRPS or nine-quantile WIS.

Sources: https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

DEFAULT_LEVELS = np.arange(1, 10, dtype=np.float64) / 10


def _targets(target, mask, *, require_each_window=True):
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask)
    if target.ndim != 3 or not all(target.shape):
        raise ValueError("target must have nonempty shape [B,C,H].")
    if mask.dtype != np.bool_ or mask.shape != target.shape:
        raise ValueError("mask must be bool with the target shape.")
    if not np.isfinite(target[mask]).all():
        raise ValueError("Observed targets must be finite.")
    if require_each_window and np.any(mask.reshape(len(mask), -1).sum(1) == 0):
        raise ValueError("Every window must contain an observed future cell.")
    return target, mask


def _forecast(value, target, mask, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != target.shape or not np.isfinite(value[mask]).all():
        raise ValueError(f"{name} must have target shape and be finite on observed cells.")
    return value


def point_metrics(target, point, mask, *, _allow_empty=False):
    target, mask = _targets(target, mask, require_each_window=not _allow_empty)
    point = _forecast(point, target, mask, "point")
    error = point[mask] - target[mask]
    n = error.size
    return {"n": int(n), "mae": float(np.abs(error).mean()) if n else None,
            "rmse": float(np.sqrt(np.square(error).mean())) if n else None,
            "bias": float(error.mean()) if n else None}


def interval_metrics(target, lower, upper, mask, *, alpha=0.2,
                     include_joint=True, _allow_empty=False):
    """Evaluate an interval, permitting -inf lower and +inf upper endpoints.

    Infinite widths/scores remain +inf and inf_count reports observed cells
    with an unbounded interval. No inf-inf arithmetic is used. joint_full is
    evaluated only on windows with every C*H label observed, or None otherwise.
    Joint coverage is descriptive and has no claim of nominal 80% coverage.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    target, mask = _targets(target, mask, require_each_window=not _allow_empty)
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lower.shape != target.shape or upper.shape != target.shape:
        raise ValueError("Interval endpoints must have the target shape.")
    lo, hi, y = lower[mask], upper[mask], target[mask]
    if (np.isnan(lo).any() or np.isnan(hi).any() or np.isposinf(lo).any()
            or np.isneginf(hi).any() or np.any(lo > hi)):
        raise ValueError("Observed intervals must be ordered; only outward infinities are allowed.")
    n = y.size
    width = hi - lo
    score = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    suffix = str(int(round(100 * (1 - alpha))))
    coverage = float(np.mean((lo <= y) & (y <= hi))) if n else None
    mean_width, mean_score = (float(width.mean()), float(score.mean())) if n else (None, None)
    result = {"n": int(n), f"coverage{suffix}": coverage,
              f"mean_width{suffix}": mean_width, f"width{suffix}": mean_width,
              f"mis{suffix}": mean_score, f"interval_score{suffix}": mean_score,
              "inf_count": int(np.isinf(width).sum()), "nominal_coverage": 1 - alpha}
    if include_joint:
        observed_count = mask.reshape(len(mask), -1).sum(1)
        eligible = observed_count > 0
        safe_y = np.where(mask, target, 0.0)
        inside = (~mask) | ((lower <= safe_y) & (safe_y <= upper))
        covered = inside.reshape(len(mask), -1).all(1)
        complete = mask.reshape(len(mask), -1).all(1)
        result.update(joint_observed_coverage=float(covered[eligible].mean()) if eligible.any() else None,
                      joint_observed_n=int(eligible.sum()),
                      joint_full=float(covered[complete].mean()) if complete.any() else None,
                      joint_full_n=int(complete.sum()),
                      joint_full_dimensions=int(np.prod(target.shape[1:])))
    return result


def _quantiles(quantiles, target, mask, levels):
    levels = np.asarray(levels, dtype=np.float64)
    if levels.shape != (9,) or not np.allclose(levels, DEFAULT_LEVELS, rtol=0, atol=1e-12):
        raise ValueError("This WIS implementation requires levels 0.1,0.2,...,0.9.")
    q = np.asarray(quantiles, dtype=np.float64)
    if q.shape != target.shape + (9,):
        raise ValueError("quantiles must have shape [B,C,H,9].")
    if not np.isfinite(q[mask]).all() or np.any(np.diff(q[mask], axis=-1) < 0):
        raise ValueError("Observed quantiles must be finite and nondecreasing.")
    return q


def quantile_metrics(target, quantiles, mask, *, levels=DEFAULT_LEVELS,
                     point=None, include_joint=True, _allow_empty=False):
    target, mask = _targets(target, mask, require_each_window=not _allow_empty)
    q = _quantiles(quantiles, target, mask, levels)
    result = point_metrics(target, q[..., 4] if point is None else point, mask, _allow_empty=_allow_empty)
    result.update(interval_metrics(target, q[..., 0], q[..., 8], mask,
                                   include_joint=include_joint, _allow_empty=_allow_empty))
    error = target[mask, None] - q[mask]
    pinball = np.maximum(DEFAULT_LEVELS * error, (DEFAULT_LEVELS - 1) * error)
    value = float(pinball.mean()) if error.size else None
    result.update(mean_pinball=value, wis=2 * value if value is not None else None,
                  wis9=2 * value if value is not None else None)
    return result


def _samples(samples, target, mask):
    x = np.asarray(samples, dtype=np.float64)
    if (x.ndim != 4 or x.shape[0] != target.shape[0] or x.shape[2:] != target.shape[1:]
            or x.shape[1] < 2):
        raise ValueError("samples must have shape [B,S,C,H], S>=2.")
    visible = np.broadcast_to(mask[:, None], x.shape)
    if not np.isfinite(x[visible]).all():
        raise ValueError("Samples at observed cells must be finite.")
    return np.where(visible, x, 0.0)


def masked_crps(target, samples, mask, *, _allow_empty=False):
    """Marginal unbiased sample CRPS, pooled over observed station/lead cells."""
    target, mask = _targets(target, mask, require_each_window=not _allow_empty)
    x = _samples(samples, target, mask)
    n_samples = x.shape[1]
    ordered = np.sort(x, axis=1)
    weights = (2 * np.arange(1, n_samples + 1) - n_samples - 1)[None, :, None, None]
    pair_sum = (ordered * weights).sum(axis=1)
    safe_y = np.where(mask, target, 0.0)
    values = np.abs(x - safe_y[:, None]).mean(1) - pair_sum / (n_samples * (n_samples - 1))
    return float(values[mask].mean()) if mask.any() else None


def masked_energy_score(target, samples, mask):
    """Original-unit, sqrt(observed-dimension)-normalized ES; window mean."""
    target, mask = _targets(target, mask)
    x = _samples(samples, target, mask).reshape(len(target), np.asarray(samples).shape[1], -1)
    safe_y = np.where(mask, target, 0.0).reshape(len(target), 1, -1)
    count = mask.reshape(len(mask), -1).sum(1)
    first = np.linalg.norm(x - safe_y, axis=-1).mean(1)
    pairs = np.zeros(len(target), dtype=np.float64)
    s = x.shape[1]
    for index in range(s - 1):
        pairs += np.linalg.norm(x[:, index + 1:] - x[:, index:index + 1], axis=-1).sum(1)
    return float(((first - pairs / (s * (s - 1))) / np.sqrt(count)).mean())


def sample_metrics(target, samples, mask, *, include_energy=True,
                   include_joint=True, _allow_empty=False):
    target, mask = _targets(target, mask, require_each_window=not _allow_empty)
    x = _samples(samples, target, mask)
    q = np.moveaxis(np.quantile(x, DEFAULT_LEVELS, axis=1), 0, -1)
    result = quantile_metrics(target, q, mask, include_joint=include_joint, _allow_empty=_allow_empty)
    crps = masked_crps(target, x, mask, _allow_empty=_allow_empty)
    result.update(crps=crps, crps_unbiased=crps, n_samples=int(x.shape[1]))
    if include_energy:
        result["energy_score_observed_normalized"] = masked_energy_score(target, x, mask)
    return result


def evaluate_predictions(target, mask, *, point=None, samples=None, quantiles=None,
                         lower=None, upper=None, station_names: Sequence[str] | None = None,
                         include_energy=True, include_joint=True):
    """Return overall, by_station and by_lead metrics in original ppm units.

    Pass samples OR quantiles OR a deterministic point. Intervals can accompany
    a point forecast (e.g. conformal output); this path deliberately does not
    produce CRPS or WIS. Group metrics pool observed cells without ES; the one
    overall ES evaluates the actual observed multivariate subvector.
    """
    target, mask = _targets(target, mask)
    if sum(value is not None for value in (samples, quantiles)) > 1:
        raise ValueError("Pass either samples or quantiles, not both.")
    if samples is not None and point is not None:
        raise ValueError("Sample forecasts use their empirical median; do not override point.")
    if (lower is None) != (upper is None):
        raise ValueError("Provide both lower and upper endpoints.")
    if lower is not None and (samples is not None or quantiles is not None):
        raise ValueError("Interval-only calibration cannot identify a new CRPS/WIS distribution.")
    if samples is None and quantiles is None and point is None:
        raise ValueError("Provide a point forecast, samples, or nine quantiles.")
    values = {name: np.asarray(value, dtype=np.float64) for name, value in
              (("point", point), ("samples", samples), ("quantiles", quantiles),
               ("lower", lower), ("upper", upper)) if value is not None}
    names = list(station_names) if station_names is not None else [str(i) for i in range(target.shape[1])]
    if len(names) != target.shape[1] or len(set(names)) != len(names):
        raise ValueError("station_names must contain one unique name per station.")

    def compute(yy, mm, vv, grouped=False):
        if "samples" in vv:
            return sample_metrics(yy, vv["samples"], mm, include_energy=include_energy and not grouped,
                                  include_joint=include_joint and not grouped, _allow_empty=grouped)
        if "quantiles" in vv:
            return quantile_metrics(yy, vv["quantiles"], mm, point=vv.get("point"),
                                    include_joint=include_joint and not grouped, _allow_empty=grouped)
        result = point_metrics(yy, vv["point"], mm, _allow_empty=grouped)
        if "lower" in vv:
            result.update(interval_metrics(yy, vv["lower"], vv["upper"], mm,
                                           include_joint=include_joint and not grouped, _allow_empty=grouped))
        return result

    def sliced(cs, hs):
        return {name: value[:, :, cs, hs] if name == "samples" else value[:, cs, hs]
                for name, value in values.items()}
    return {"unit": "ppm", "aggregation": "marginals: observed-cell mean; energy: equal-window mean",
            "overall": compute(target, mask, values),
            "by_station": {name: compute(target[:, i:i+1], mask[:, i:i+1],
                                           sliced(slice(i, i+1), slice(None)), True)
                           for i, name in enumerate(names)},
            "by_lead": {str(h+1): compute(target[:, :, h:h+1], mask[:, :, h:h+1],
                                           sliced(slice(None), slice(h, h+1)), True)
                        for h in range(target.shape[2])}}
