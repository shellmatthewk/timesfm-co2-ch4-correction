"""Daily panel from the raw NOAA files, and forecast windows for any group of series.

The validity rule and the window rules are those of the base project
(01_实验项目/四站14天预测14天_20260920 and uq1/修改后/data1/prepare_dataset.py);
`check_against_project` proves that the CO2 part reproduces the project exactly.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / "timesfm_base"   # TimesFM loading code (src/uq14), vendored TimesFM source and the downloaded weights
CO2_RAW = ROOT.parent / "data" / "raw"
CH4_RAW = ROOT.parent / "data" / "raw"
FILES = {f"co2_{s}": CO2_RAW / f"co2_{s}_surface-insitu_1_ccgg_DailyData.txt" for s in ("brw", "mlo", "smo", "spo")}
FILES.update({f"ch4_{s}": CH4_RAW / f"ch4_{s}_surface-insitu_1_ccgg_DailyData.txt" for s in ("brw", "mlo", "smo")})
GROUPS = {"co2": ("co2_brw", "co2_mlo", "co2_smo", "co2_spo"), "ch4": ("ch4_brw", "ch4_mlo")}
MAX_BOUNDARY_GAP = 7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    return json.loads((ROOT / "configs/experiment.json").read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float))
    tmp.replace(path)


def read_noaa_daily(path):
    """{date: value or None}. Valid iff value>0, nvalue>0, qcflag[:2]=='..', qcflag[2]!='P'."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or line.startswith("site_code") or not line.strip():
            continue
        p = line.split()
        d = date(int(p[1]), int(p[2]), int(p[3])).isoformat()
        value, nvalue, flag = float(p[10]), int(p[12]), p[-1]
        valid = np.isfinite(value) and value > 0 and nvalue > 0 and flag[:2] == ".." and flag[2] != "P"
        if d in out:
            raise ValueError(f"duplicate date {d} in {path}")
        out[d] = value if valid else None
    return out


def build_panel(start="1990-01-01", end="2025-12-31"):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    dates = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
    values = {}
    for name, path in FILES.items():
        raw = read_noaa_daily(path)
        values[name] = np.array([np.nan if raw.get(d) is None else raw[d] for d in dates], dtype=np.float64)
    provenance = {name: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for name, path in FILES.items()}
    return dates, values, provenance


def check_against_project(dates, values):
    """The CO2 series must equal the project's daily panel wherever both exist."""
    import csv
    rows = list(csv.DictReader(open(BASE / "data/daily_panel.csv")))
    index = {d: i for i, d in enumerate(dates)}
    worst, compared, disagree = 0.0, 0, 0
    for r in rows:
        i = index.get(r["timestamp"])
        if i is None or r["timestamp"] > dates[-1]:
            continue
        for s in ("brw", "mlo", "smo", "spo"):
            mine, theirs = values[f"co2_{s}"][i], r[f"{s}_daily_archive_ppm"]
            theirs = np.nan if theirs == "" else float(theirs)
            compared += 1
            if np.isnan(mine) != np.isnan(theirs):
                disagree += 1
            elif not np.isnan(mine):
                worst = max(worst, abs(mine - theirs))
    return {"cells_compared": compared, "validity_disagreements": disagree, "max_abs_value_difference": worst}


def fill_context(raw, max_boundary_gap=MAX_BOUNDARY_GAP):
    """Project rule for k series. Returns (filled, internal, boundary) or raises ValueError."""
    values = np.asarray(raw, dtype=np.float64)
    filled = values.copy()
    internal, boundary = np.zeros(values.shape, bool), np.zeros(values.shape, bool)
    for j, channel in enumerate(values):
        ids = np.flatnonzero(np.isfinite(channel))
        if not len(ids):
            raise ValueError("no_observed_values")
        if ids[0] > max_boundary_gap or len(channel) - 1 - ids[-1] > max_boundary_gap:
            raise ValueError("edge_gap_too_long")
        missing = ~np.isfinite(channel)
        internal[j, ids[0]:ids[-1] + 1] = missing[ids[0]:ids[-1] + 1]
        boundary[j] = missing & ~internal[j]
        loc = np.flatnonzero(missing)
        filled[j, loc] = np.interp(loc, ids, channel[ids])
    return filled, internal, boundary


def build_windows(dates, values, group, first_target, last_target, stride, context=14, horizon=14):
    """Windows whose 14 targets lie in [first_target, last_target], grid anchored at first_target."""
    series = np.stack([values[name] for name in GROUPS[group]]) if isinstance(group, str) else \
        np.stack([values[name] for name in group])
    index = {d: i for i, d in enumerate(dates)}
    support = min(int(np.flatnonzero(np.isfinite(s))[-1]) for s in series)
    records, reasons = [], {}
    start, last = date.fromisoformat(first_target), date.fromisoformat(last_target)
    while start + timedelta(days=horizon - 1) <= last:
        i = index.get(start.isoformat())
        start += timedelta(days=stride)
        if i is None or i < context or i + horizon > len(dates) or i - 1 > support:
            reasons["outside_source"] = reasons.get("outside_source", 0) + 1
            continue
        y = series[:, i:i + horizon]
        mask = np.isfinite(y)
        try:
            x, internal, boundary = fill_context(series[:, i - context:i])
        except ValueError as exc:
            reasons[str(exc)] = reasons.get(str(exc), 0) + 1
            continue
        if not mask.any():
            reasons["no_future_label"] = reasons.get("no_future_label", 0) + 1
            continue
        records.append({"x": x.astype(np.float32), "y": y.copy(), "mask": mask, "origins": dates[i - 1],
                        "forecast_dates": np.array(dates[i:i + horizon], dtype="U10"),
                        "imputed_mask": internal | boundary, "interpolation_mask": internal, "boundary_mask": boundary})
    keys = ("x", "y", "mask", "origins", "forecast_dates", "imputed_mask", "interpolation_mask", "boundary_mask")
    arrays = {k: np.stack([np.asarray(r[k]) for r in records]) for k in keys} if records else {}
    return arrays, reasons
