"""Per-station/per-lead 80% split conformal intervals; expansion-only CQR.

Romano, Patterson, Candes, Conformalized Quantile Regression (2019), equations
9--11 and Appendix A Lemma 2: https://arxiv.org/pdf/1905.03222
Official implementation: https://github.com/yromano/cqr

For n available calibration labels in one station/lead, the order-statistic
rank is ceil((n+1)*(1-alpha)). A rank beyond n yields +infinity. The standard
score max(lower-y,y-upper) can be negative; our deliberate conservative variant
uses max(selected_score,0), so intervals only expand and never cross.

The finite-sample statement is MARGINAL for one cell, conditional on a fitted
model independent of calibration labels, when calibration and future scores
are exchangeable. Time dependence and selection by label availability may
violate that premise. Neither iid/exchangeability nor simultaneous 56-cell
coverage is established here. There is no distribution-free conditional or
missing-cell guarantee. Expansion-only intervals can be more conservative than
ordinary CQR, so its near-exact upper coverage bound is not asserted.

Fit functions consume only caller-supplied calibration labels. Apply functions
have no label argument, preserve the original point, and only change interval
endpoints. They do not fabricate CRPS or a nine-quantile WIS after calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    cal_n: np.ndarray
    rank: np.ndarray
    raw_q: np.ndarray
    q: np.ndarray
    method: str = "cqr_expansion_only"

    def to_dict(self):
        """JSON-safe provenance: explicit '+inf' instead of nonstandard Infinity."""
        def thresholds(value):
            return [["+inf" if np.isposinf(v) else float(v) for v in row] for row in value]
        return {"method": self.method, "alpha": self.alpha, "coverage": 1-self.alpha,
                "cal_n": self.cal_n.tolist(), "rank": self.rank.tolist(),
                "raw_q": thresholds(self.raw_q), "q": thresholds(self.q),
                "infinite_threshold_count": int(np.isinf(self.q).sum()),
                "rank_rule": "ceil((cal_n+1)*(1-alpha)); +inf when rank>cal_n",
                "expansion_only": True,
                "scope": "Each station and lead separately; no simultaneous 56-cell guarantee.",
                "validity_conditions": "Marginal finite-sample lower coverage bound requires exchangeable calibration/future scores and a model fixed independently of calibration labels. Time dependence and missingness may violate this; the experiment does not establish iid/exchangeability.",
                "source": "https://arxiv.org/pdf/1905.03222"}

    @classmethod
    def from_dict(cls, value):
        def thresholds(rows):
            return np.asarray([[np.inf if v == "+inf" else float(v) for v in row] for row in rows])
        result = cls(float(value["alpha"]), np.asarray(value["cal_n"], dtype=np.int64),
                     np.asarray(value["rank"], dtype=np.int64), thresholds(value["raw_q"]),
                     thresholds(value["q"]), value.get("method", "cqr_expansion_only"))
        _check_calibration(result)
        return result


def _check_calibration(calibration):
    if not isinstance(calibration, ConformalCalibration) or not 0 < calibration.alpha < 1:
        raise ValueError("A valid ConformalCalibration is required.")
    shape = calibration.q.shape
    if len(shape) != 2 or not all(shape):
        raise ValueError("Calibration arrays must be nonempty [station,lead].")
    if any(value.shape != shape for value in (calibration.cal_n, calibration.rank, calibration.raw_q)):
        raise ValueError("All calibration arrays must share a shape.")
    if (np.isnan(calibration.q).any() or np.isnan(calibration.raw_q).any()
            or np.any(calibration.q < 0) or np.isneginf(calibration.raw_q).any()
            or np.any(calibration.cal_n < 0)):
        raise ValueError("Invalid conformal threshold or calibration count.")
    expected_rank = np.array([[_rank(int(n), calibration.alpha) for n in row] for row in calibration.cal_n])
    if (not np.array_equal(calibration.rank, expected_rank)
            or not np.array_equal(calibration.q, np.maximum(calibration.raw_q, 0))
            or not np.array_equal(np.isposinf(calibration.q), calibration.rank > calibration.cal_n)):
        raise ValueError("Calibration ranks/thresholds violate the expansion-only finite-sample rule.")


def _rank(n, alpha):
    # Decimal avoids turning a mathematically integer rank into k+1 via float error.
    return int((Decimal(n+1) * (Decimal(1)-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))


def fit_cqr(lower, upper, target, mask, *, alpha=0.2):
    """Fit each C,H cell from its own available calibration labels only.

    Zero-label cells are retained with cal_n=0, rank=1, threshold=+inf. This is
    transparent lack of information, not a fabricated calibration observation.
    Calibration rows with no observations contribute no scores to any cell.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    target, lower, upper = (np.asarray(v, dtype=np.float64) for v in (target, lower, upper))
    mask = np.asarray(mask)
    if target.ndim != 3 or any(d == 0 for d in target.shape[1:]):
        raise ValueError("Calibration target must have shape [B,C,H].")
    if lower.shape != target.shape or upper.shape != target.shape or mask.shape != target.shape or mask.dtype != np.bool_:
        raise ValueError("Calibration predictions/targets/bool mask must have matching shapes.")
    if any(not np.isfinite(v[mask]).all() for v in (target, lower, upper)) or np.any(lower[mask] > upper[mask]):
        raise ValueError("Observed calibration labels and ordered initial intervals must be finite.")
    n = mask.sum(axis=0, dtype=np.int64)
    rank = np.empty(n.shape, dtype=np.int64)
    raw_q = np.full(n.shape, np.inf, dtype=np.float64)
    for c, h in np.ndindex(n.shape):
        rank[c,h] = _rank(int(n[c,h]), alpha)
        if rank[c,h] <= n[c,h]:
            keep = mask[:,c,h]
            y, lo, hi = target[keep,c,h], lower[keep,c,h], upper[keep,c,h]
            scores = np.maximum(lo-y, y-hi)
            raw_q[c,h] = np.partition(scores, int(rank[c,h])-1)[int(rank[c,h])-1]
    result = ConformalCalibration(float(alpha), n, rank, raw_q, np.maximum(raw_q, 0))
    _check_calibration(result)
    return result


def apply_cqr(calibration, lower, upper, *, point=None):
    """Return expanded endpoints and an unchanged copy of the optional point."""
    _check_calibration(calibration)
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if lower.ndim != 3 or lower.shape != upper.shape or lower.shape[1:] != calibration.q.shape:
        raise ValueError("Prediction endpoints must have shape [B,C,H] matching calibration.")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("Initial prediction intervals must be finite and ordered.")
    result = {"lower": lower-calibration.q[None], "upper": upper+calibration.q[None]}
    if point is not None:
        point = np.asarray(point)
        if point.shape != lower.shape or not np.isfinite(point).all():
            raise ValueError("Point forecasts must be finite and match the interval shape.")
        result["point"] = point.copy()
    return result


def fit_absolute_residual(point, target, mask, *, alpha=0.2):
    """Symmetric split conformal interval using absolute point residuals."""
    fit = fit_cqr(point, point, target, mask, alpha=alpha)
    return ConformalCalibration(fit.alpha, fit.cal_n, fit.rank, fit.raw_q, fit.q,
                                "absolute_residual_split_conformal")


def apply_absolute_residual(calibration, point):
    return apply_cqr(calibration, point, point, point=point)


fit_cqr80 = fit_cqr
fit_absolute_residual80 = fit_absolute_residual
