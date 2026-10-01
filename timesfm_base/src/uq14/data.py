"""Four station histories -> four station futures, with observed-only targets.

No global CO2 series or alternate station-trend series is read. Historical
interpolation is local to each input window. Future targets are never filled.
"""
from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

STATIONS = ("brw", "mlo", "smo", "spo")
STATION_COLUMNS = tuple(f"{s}_daily_archive_ppm" for s in STATIONS)
DEFAULT_SPLITS = {
    "train": {"start": "2017-01-01", "end": "2022-12-31"},
    "validation": {"start": "2023-01-01", "end": "2023-12-31"},
    "calibration": {"start": "2024-01-01", "end": "2024-12-31"},
    "test": {"start": "2025-01-01", "end": "2025-12-31"},
}
MAX_BOUNDARY_GAP = 7


class UnfillableContext(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(str(issues))


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _read_csv(path, required):
    path = Path(path)
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if not set(required).issubset(reader.fieldnames or []):
        raise ValueError(f"Missing columns in {path.name}: {sorted(set(required)-set(reader.fieldnames or []))}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"Empty source: {path}")
    dates = [r["timestamp"] for r in rows]
    try:
        parsed = [date.fromisoformat(d) for d in dates]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid timestamp in {path}") from exc
    if any(d.isoformat() != raw for d, raw in zip(parsed, dates)):
        raise ValueError("Dates must use canonical YYYY-MM-DD.")
    if any(b-a != timedelta(days=1) for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Dates must be unique, increasing, consecutive calendar days.")
    provenance = {"path": str(path.absolute()), "resolved_path": str(path.resolve()),
                  "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    return rows, dates, provenance


def _read_sources(panel_path, quality_path):
    panel, dates, panel_source = _read_csv(panel_path, ("timestamp", *STATION_COLUMNS))
    quality, qdates, quality_source = _read_csv(quality_path, ("timestamp", *(f"{s}_archive_valid" for s in STATIONS)))
    if dates != qdates:
        raise ValueError("Panel and quality dates must match exactly.")
    values = np.empty((4, len(dates)), dtype=np.float64)
    qvalid = np.empty(values.shape, dtype=bool)
    for j, (station, column) in enumerate(zip(STATIONS, STATION_COLUMNS)):
        for i, (row, qrow) in enumerate(zip(panel, quality)):
            raw = row[column]
            try:
                value = np.nan if raw in (None, "") else float(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Non-numeric station value: {column} {dates[i]}") from exc
            if np.isinf(value):
                raise ValueError(f"Infinite station value: {column} {dates[i]}")
            values[j, i] = value
            flag = qrow[f"{station}_archive_valid"]
            if flag not in ("0", "1"):
                raise ValueError(f"Quality validity must be 0/1: {station} {dates[i]}")
            qvalid[j, i] = flag == "1"
    valid = np.isfinite(values)
    mismatch = np.argwhere(valid != qvalid)
    if len(mismatch):
        examples = [(STATIONS[j], dates[i]) for j, i in mismatch[:10]]
        raise ValueError(f"Station finite values and quality valid flags disagree at {len(mismatch)} cells: {examples}")
    if not valid.any(axis=1).all():
        raise ValueError("Every station needs at least one real observation in the source.")
    last_ids = [int(np.flatnonzero(v)[-1]) for v in valid]
    return dates, values, min(last_ids), {"panel": panel_source, "quality": quality_source}


def fill_context(raw, max_boundary_gap=MAX_BOUNDARY_GAP):
    """Return float64 filled context and separate internal/edge boolean masks."""
    _positive_int(max_boundary_gap, "max_boundary_gap", 0)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] < 2:
        raise ValueError("History must have shape [4, context>=2].")
    if np.isinf(values).any():
        raise ValueError("Infinite historical values are invalid.")
    observed, issues = [], []
    for station, channel in zip(STATIONS, values):
        ids = np.flatnonzero(np.isfinite(channel))
        observed.append(ids)
        if not len(ids):
            issues.append({"station": station, "reason": "no_observed_values"})
            continue
        for edge, gap in (("leading", int(ids[0])), ("trailing", int(len(channel)-1-ids[-1]))):
            if gap > max_boundary_gap:
                issues.append({"station": station, "reason": edge+"_gap_too_long", "days": gap})
    if issues:
        raise UnfillableContext(issues)
    filled = values.copy()
    internal, boundary = np.zeros(values.shape, bool), np.zeros(values.shape, bool)
    for j, ids in enumerate(observed):
        missing = ~np.isfinite(values[j])
        internal[j, ids[0]:ids[-1]+1] = missing[ids[0]:ids[-1]+1]
        boundary[j] = missing & ~internal[j]
        loc = np.flatnonzero(missing)
        filled[j, loc] = np.interp(loc, ids, values[j, ids])
    return filled, internal, boundary


def _missing_runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [(int(a), int(b-1)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _stats(values):
    a = np.asarray(values)
    return {"min": int(a.min()), "max": int(a.max()), "median": float(np.median(a)),
            "mean": float(a.mean())} if a.size else {"min": None, "max": None, "median": None, "mean": None}


def _latest(dates, values, support, context, horizon, as_of=None):
    limit = dates[support] if as_of is None else min(dates[support], date.fromisoformat(as_of).isoformat())
    for end in range(support, context-2, -1):
        if dates[end] > limit:
            continue
        try:
            x, internal, boundary = fill_context(values[:, end-context+1:end+1])
        except UnfillableContext:
            continue
        return {"x": x.astype(np.float32), "origin": dates[end],
                "forecast_dates": np.asarray([(date.fromisoformat(dates[end])+timedelta(days=h)).isoformat()
                                               for h in range(1, horizon+1)], dtype="U10"),
                "imputed_mask": internal|boundary, "interpolation_mask": internal, "boundary_mask": boundary}
    return None


def load_latest_input(panel_path, quality_path, context=14, horizon=14, *, as_of=None):
    """Return one unbatched input dictionary without consulting future labels.

    The latest origin is capped by the earliest last real station observation.
    ``as_of`` is an upper bound; this API never extends station histories.
    """
    _positive_int(context, "context", 2)
    _positive_int(horizon, "horizon")
    dates, values, support, _ = _read_sources(panel_path, quality_path)
    result = _latest(dates, values, support, context, horizon, as_of)
    if result is None:
        raise ValueError("No eligible historical input exists on/before as_of.")
    return result


def _pack(records, context, horizon):
    if not records:
        return {"x": np.empty((0, 4, context), np.float32),
                "y": np.empty((0, 4, horizon), np.float64),
                "mask": np.empty((0, 4, horizon), bool),
                "origins": np.empty(0, dtype="U10"),
                "forecast_dates": np.empty((0, horizon), dtype="U10"),
                **{k: np.empty((0, 4, context), bool) for k in
                   ("imputed_mask", "interpolation_mask", "boundary_mask")}}
    return {key: np.stack([r[key] for r in records]) for key in
            ("x", "y", "mask", "origins", "forecast_dates", "imputed_mask", "interpolation_mask", "boundary_mask")}


def load_windows(panel_path, quality_path, context=14, horizon=14, stride=14, *, splits=None):
    """Return (split -> array dictionary, JSON-serializable audit).

    A split's grid is anchored to its first target date. Retain an input iff
    historical interpolation meets the declared rules and at least one future
    real station cell exists. Missing targets remain NaN. The optional splits
    argument supports explicit alternative protocols and synthetic regression
    tests; defaults are the fixed 2017--2025 protocol.
    """
    for value, name, minimum in ((context, "context", 2), (horizon, "horizon", 1), (stride, "stride", 1)):
        _positive_int(value, name, minimum)
    dates, values, support, sources = _read_sources(panel_path, quality_path)
    split_config = DEFAULT_SPLITS if splits is None else splits
    spans = [(key, date.fromisoformat(span["start"]), date.fromisoformat(span["end"]))
             for key, span in split_config.items()]
    if not spans or any(a > b for _, a, b in spans) or any(b >= c for (_, _, b), (_, c, _) in zip(spans, spans[1:])):
        raise ValueError("Splits must be nonempty, disjoint and in chronological order.")
    index, arrays, reports = {d: i for i, d in enumerate(dates)}, {}, {}
    for split, first, last in spans:
        records, excluded, candidates, input_eligible = [], [], 0, 0
        candidate_complete, candidate_any = 0, 0
        start = first
        while start+timedelta(days=horizon-1) <= last:
            first_target = start.isoformat()
            candidates += 1
            start += timedelta(days=stride)
            i = index.get(first_target)
            if i is None or i < context or i+horizon > len(dates):
                excluded.append({"forecast_start": first_target, "reason": "insufficient_source_history_or_labels"})
                continue
            y = values[:, i:i+horizon].copy()
            mask = np.isfinite(y)
            candidate_complete += int(mask.all())
            candidate_any += int(mask.any())
            if i-1 > support:
                excluded.append({"forecast_start": first_target, "reason": "origin_after_four_station_actual_support"})
                continue
            try:
                x, internal, boundary = fill_context(values[:, i-context:i])
            except UnfillableContext as exc:
                excluded.append({"forecast_start": first_target, "origin": dates[i-1],
                                 "reason": "unresolved_history", "issues": exc.issues})
                continue
            input_eligible += 1
            if not mask.any():
                excluded.append({"forecast_start": first_target, "origin": dates[i-1], "reason": "all_future_station_labels_missing"})
                continue
            records.append({"x": x.astype(np.float32), "y": y, "mask": mask,
                "origins": np.asarray(dates[i-1], dtype="U10"),
                "forecast_dates": np.asarray(dates[i:i+horizon], dtype="U10"),
                "imputed_mask": internal|boundary, "interpolation_mask": internal, "boundary_mask": boundary})
        a = _pack(records, context, horizon)
        arrays[split] = a
        m = a["mask"]
        valid_per_window = m.sum(axis=(1, 2))
        per_station_lead = m.sum(axis=0)
        per_station = {}
        for j, station in enumerate(STATIONS):
            imputed = a["imputed_mask"][:, j]
            internal_runs = [b-c+1 for row in a["interpolation_mask"][:, j] for c, b in _missing_runs(row)]
            boundary_runs = [b-c+1 for row in a["boundary_mask"][:, j] for c, b in _missing_runs(row)]
            per_station[station] = {"observed_label_cells": int(m[:, j].sum()),
                "possible_label_cells": int(m[:, j].size),
                "label_observed_fraction": float(m[:, j].mean()) if m[:, j].size else None,
                "observed_labels_per_lead": per_station_lead[j].tolist(),
                "observed_labels_per_window": _stats(m[:, j].sum(axis=1)),
                "windows_with_no_label": int((~m[:, j].any(axis=1)).sum()),
                "real_history_per_window": _stats((~imputed).sum(axis=1)),
                "history_imputed_fraction": float(imputed.mean()) if imputed.size else None,
                "internal_filled_cells": int(a["interpolation_mask"][:, j].sum()),
                "boundary_filled_cells": int(a["boundary_mask"][:, j].sum()),
                "max_internal_gap_days": max(internal_runs, default=0),
                "max_boundary_gap_days": max(boundary_runs, default=0)}
        reports[split] = {**split_config[split], "candidate_windows": candidates,
            "candidate_complete_target_windows": candidate_complete,
            "candidate_any_observed_target_windows": candidate_any,
            "input_eligible_windows": input_eligible, "retained_windows": len(records),
            "excluded_windows": len(excluded), "complete_56_windows": int(m.all(axis=(1, 2)).sum()),
            "complete_target_windows": int(m.all(axis=(1, 2)).sum()),
            "any_observed_target_windows": int(m.any(axis=(1, 2)).sum()),
            "observed_target_cells": int(m.sum()), "possible_target_cells": int(m.size),
            "observed_labels_per_window": _stats(valid_per_window),
            "observed_labels_per_station_lead": per_station_lead.tolist(),
            "observed_target_fraction": float(m.mean()) if m.size else None,
            "historical_imputed_fraction": float(a["imputed_mask"].mean()) if a["imputed_mask"].size else None,
            "first_forecast_date": str(a["forecast_dates"][0, 0]) if records else None,
            "last_forecast_date": str(a["forecast_dates"][-1, -1]) if records else None,
            "first_origin": str(a["origins"][0]) if records else None,
            "last_origin": str(a["origins"][-1]) if records else None,
            "per_station": per_station,
            "excluded_reason_counts": dict(Counter(r["reason"] for r in excluded)),
            "excluded_details": excluded}
    latest = _latest(dates, values, support, context, horizon)
    availability = {}
    for j, station in enumerate(STATIONS):
        valid = np.isfinite(values[j])
        ids = np.flatnonzero(valid)
        long_gaps = [{"start": dates[a], "end": dates[b], "days": b-a+1}
                     for a, b in _missing_runs(~valid) if b-a+1 >= 14]
        availability[station] = {"first_real_date": dates[int(ids[0])], "last_real_date": dates[int(ids[-1])],
            "observed_days": int(valid.sum()), "missing_days": int((~valid).sum()), "missing_runs_ge_14_days": long_gaps}
    report = {"protocol": "four_station_14_history_14_future_observed_targets_v1",
        "context": context, "horizon": horizon, "stride": stride, "input_shape": [4, context],
        "target_shape": [4, horizon], "station_order": list(STATIONS), "station_columns": list(STATION_COLUMNS),
        "unit": "ppm", "sources": sources, "source_rows": len(dates),
        "source_start": dates[0], "source_end": dates[-1], "quality_validity_mismatch_count": 0,
        "global_or_trend_columns_used": False, "target_imputation": "none",
        "label_policy": "retain at least one observed cell; missing y stays NaN; target mask is for loss/evaluation only",
        "historical_fill": "internal linear within each origin's history, boundary nearest within same window <=7 days",
        "max_boundary_gap_days": MAX_BOUNDARY_GAP, "internal_gap_cap_days": None,
        "four_station_actual_support_end": dates[support],
        "latest_forecast_origin": latest["origin"] if latest else None,
        "latest_forecast_dates": latest["forecast_dates"].tolist() if latest else [],
        "station_availability": availability, "splits": reports,
        "limitations": ["Labels are only scored where valid station observations exist; missingness may be informative.",
            "Target validity is unavailable at forecast time and must never be supplied as an input feature.",
            "Masked energy score assesses observed subvectors, not verified calibration of every complete 56-dimensional path.",
            "Historical fills are assumptions; retrospective archive revisions and pretraining overlap are not ruled out.",
            "Only retained fixed-grid windows are evaluated, not every calendar date."]}
    return arrays, report
