"""Regular numerical time series; history-only filling and chronological folds."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import file_sha, resolve, run_directory, write_json


def channel_names(cfg):
    return [c["name"] for c in cfg["data"]["channels"]]


def _fixed_offset(value):
    try:
        delta = pd.Timedelta(pd.tseries.frequencies.to_offset(value).nanos, unit="ns")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("cadence must be a fixed interval such as 1D, 1h or 15min") from exc
    if delta <= pd.Timedelta(0):
        raise ValueError("cadence must be positive")
    return delta


def read_panel(cfg):
    spec = cfg["data"]
    path = resolve(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV missing: {path}. Use fetch for GOES, or supply a CSV.")
    frame = pd.read_csv(path)
    names = channel_names(cfg)
    needed = [spec["timestamp_column"], *names]
    absent = set(needed) - set(frame)
    if absent:
        raise ValueError(f"CSV missing columns: {sorted(absent)}")
    timestamps = pd.to_datetime(frame[spec["timestamp_column"]], utc=True, errors="raise")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Input timestamps must be present and unique")
    frame.index = pd.DatetimeIndex(timestamps)
    frame = frame.sort_index()
    start, end = pd.Timestamp(spec["start"], tz="UTC"), pd.Timestamp(spec["end"], tz="UTC") + pd.Timedelta(days=1)
    frame = frame.loc[(frame.index >= start) & (frame.index < end)].copy()
    if len(frame) < 2:
        raise ValueError("At least two input rows are needed in the selected data period")
    delta = _fixed_offset(spec["cadence"])
    if spec["native_cadence"] is None:
        gaps = frame.index.to_series().diff().dropna()
        native = gaps.value_counts().index[0]
    else:
        native = _fixed_offset(spec["native_cadence"])
    if native <= pd.Timedelta(0) or native > delta:
        raise ValueError("Input cadence exceeds output cadence; upsampling observations is not supported")
    for c in spec["channels"]:
        frame[c["name"]] = pd.to_numeric(frame[c["name"]], errors="coerce")
        quality = c.get("quality_column")
        if quality:
            if quality not in frame:
                raise ValueError(f"Required quality column {quality!r} is missing")
            frame.loc[frame[quality] != c.get("good_quality", 0), c["name"]] = np.nan
        frame.loc[~np.isfinite(frame[c["name"]]), c["name"]] = np.nan
        if c.get("transform", "identity") == "log10":
            frame.loc[frame[c["name"]] <= 0, c["name"]] = np.nan
        if c.get("transform", "identity") == "log1p":
            frame.loc[frame[c["name"]] < 0, c["name"]] = np.nan
        if "minimum" in c:
            frame.loc[frame[c["name"]] < c["minimum"], c["name"]] = np.nan
    # A bin is labelled by its END, when the complete aggregate becomes available.
    # This prevents the first minutes of a day from seeing that day's final mean.
    resampler = frame[names].resample(spec["cadence"], closed="left", label="right", origin="start_day")
    panel = resampler.aggregate(spec["aggregation"])
    counts = resampler.count()
    expected = float(delta / native)
    panel = panel.where(counts >= expected * spec["min_bin_coverage"])
    panel = panel.loc[(panel.index > start) & (panel.index <= end)]
    if panel.empty:
        raise ValueError("No data bins in the selected period")
    physical = panel.to_numpy(dtype=np.float64)
    model = physical.copy()
    for j, c in enumerate(spec["channels"]):
        if c.get("transform", "identity") == "log10":
            model[:, j] = np.log10(physical[:, j])
        elif c.get("transform", "identity") == "log1p":
            model[:, j] = np.log1p(physical[:, j])
    digest = file_sha(path)
    provenance = path.with_suffix(".provenance.json")
    source_record = {}
    if provenance.is_file():
        metadata = json.loads(provenance.read_text(encoding="utf-8"))
        if metadata.get("input_sha256") != digest:
            raise ValueError("Input CSV does not match its source provenance")
        source_record = {"source_provenance": metadata, "provenance_sha256": file_sha(provenance)}
    audit = {"input_sha256": digest, "input_rows": len(frame), "bins": len(panel),
             "native_cadence": str(native), "cadence": spec["cadence"], "timestamp_convention": "bin end, UTC",
             "channels": names, "observed_bins": dict(zip(names, np.isfinite(model).sum(0).tolist())), **source_record}
    return panel.index.tz_localize(None).to_numpy(dtype="datetime64[ns]"), model, physical, audit


def fill_history(raw, spec):
    out = np.asarray(raw, dtype=np.float64).copy()
    imputed = ~np.isfinite(out)
    for j, row in enumerate(out):
        ids = np.flatnonzero(np.isfinite(row))
        if not len(ids) or len(ids) / len(row) < spec["min_observed_history_fraction"]:
            raise ValueError("insufficient_history")
        if ids[0] > spec["max_edge_gap"] or len(row) - 1 - ids[-1] > spec["max_edge_gap"]:
            raise ValueError("edge_gap")
        if np.max(np.diff(ids) - 1, initial=0) > spec["max_internal_gap"]:
            raise ValueError("internal_gap")
        missing = np.flatnonzero(imputed[j])
        row[missing] = np.interp(missing, ids, row[ids])
    return out, imputed


def build_windows(times, model, physical, spec):
    context, horizon = spec["context"], spec["horizon"]
    rows, excluded = [], {}
    for first in range(context, len(times) - horizon + 1, spec["stride"]):
        try:
            x, imputed = fill_history(model[first - context:first].T, spec)
        except ValueError as exc:
            reason = str(exc)
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        y = model[first:first + horizon].T.copy()
        mask = np.isfinite(y)
        if not mask.any():
            excluded["no_target"] = excluded.get("no_target", 0) + 1
            continue
        rows.append({"x": x, "imputed": imputed, "y": y, "mask": mask,
                     "physical_y": physical[first:first + horizon].T.copy(),
                     "origin": times[first - 1], "target_times": times[first:first + horizon]})
    if not rows:
        raise ValueError("No usable windows: reduce the lengths, adjust history-gap rules, or supply more data")
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}, excluded


def _cap(idx, maximum):
    # Even coverage through time, deterministic and independent of target values.
    if maximum is not None and len(idx) > maximum:
        return idx[np.linspace(0, len(idx) - 1, maximum, dtype=int)]
    return idx


def evaluation_times(windows, cfg):
    """Select measurement-bin starts or completed-bin ends for folds/reporting."""
    target, origin = windows["target_times"], windows.get("origin")
    if cfg["evaluation"]["date_basis"] == "bin_start":
        delta = _fixed_offset(cfg["data"]["cadence"]).to_timedelta64()
        target = target - delta
        origin = origin - delta if origin is not None else None
    return target, origin


def fold_indices(windows, year, cfg):
    e = cfg["evaluation"]
    target_times, _ = evaluation_times(windows, cfg)
    first = target_times[:, 0]
    last = target_times[:, -1]
    train_start = e["train_start"]
    if e["train_years"] is not None:
        train_start = max(train_start, f"{year - 1 - e['train_years']}-01-01")
    spans = {"train": (train_start, f"{year - 1}-01-01"),
             "validation": (f"{year - 1}-01-01", f"{year}-01-01"),
             "test": (f"{year}-01-01", f"{year + 1}-01-01")}
    result = {}
    for part, (a, b) in spans.items():
        idx = np.flatnonzero((first >= np.datetime64(a)) & (last < np.datetime64(b)))
        if part == "validation":
            idx = idx[::e["validation_stride"]]
        result[part] = _cap(idx, e[f"max_{part}_windows"])
    return result


def inverse(values, channels, channel_axis):
    out = np.asarray(values, dtype=np.float64).copy()
    for j, c in enumerate(channels):
        sl = [slice(None)] * out.ndim
        sl[channel_axis] = j
        if c.get("transform", "identity") == "log10":
            with np.errstate(over="raise", invalid="raise"):
                out[tuple(sl)] = np.power(10.0, out[tuple(sl)])
        elif c.get("transform", "identity") == "log1p":
            with np.errstate(over="raise", invalid="raise"):
                out[tuple(sl)] = np.expm1(out[tuple(sl)])
        if "minimum" in c:
            out[tuple(sl)] = np.maximum(out[tuple(sl)], c["minimum"])
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite prediction after inverse transformation")
    return out


def prepare(cfg):
    times, model, physical, audit = read_panel(cfg)
    windows, excluded = build_windows(times, model, physical, cfg["windows"])
    out = run_directory(cfg)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "windows.npz", **windows)
    audit.update({"windows": len(windows["x"]), "excluded": excluded,
                  "evaluation_date_basis": cfg["evaluation"]["date_basis"],
                  "windows_sha256": file_sha(out / "windows.npz"),
                  "folds": {str(y): {p: len(i) for p, i in fold_indices(windows, y, cfg).items()}
                            for y in cfg["evaluation"]["test_years"]}})
    write_json(out / "config.json", cfg)
    write_json(out / "data_audit.json", audit)
    print(f"Prepared {audit['windows']} windows; context={cfg['windows']['context']}, horizon={cfg['windows']['horizon']}")
    print(f"Saved to {out}")
    print("Fold sizes:", audit["folds"])
    return windows, audit


def load_windows(cfg):
    out = run_directory(cfg)
    if not (out / "windows.npz").is_file():
        return prepare(cfg)
    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))
    if audit["input_sha256"] != file_sha(resolve(cfg["data"]["path"])):
        raise ValueError("Input CSV changed. Run prepare and encode again before using cached results.")
    if "provenance_sha256" in audit:
        provenance = resolve(cfg["data"]["path"]).with_suffix(".provenance.json")
        if not provenance.is_file() or file_sha(provenance) != audit["provenance_sha256"]:
            raise ValueError("Source provenance changed. Run prepare and encode again.")
    if audit["windows_sha256"] != file_sha(out / "windows.npz"):
        raise ValueError("Window cache changed. Run prepare and encode again.")
    with np.load(out / "windows.npz", allow_pickle=False) as saved:
        return dict(saved), audit
