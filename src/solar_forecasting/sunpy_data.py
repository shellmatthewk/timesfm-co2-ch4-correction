"""Optional SunPy GOES XRS ingestion. No downloads occur during model training."""
from __future__ import annotations

import json
import os

from .config import file_sha, resolve, run_directory, write_json


def fetch_goes(cfg):
    if cfg["data"]["source"] != "goes_xrs":
        raise ValueError("fetch requires data.source=goes_xrs; CSV inputs are supplied locally")
    try:
        import pandas as pd
        import sunpy
        import sunpy.timeseries as ts
        from sunpy.net import Fido, attrs as a
    except ImportError as exc:
        raise RuntimeError('SunPy ingestion needs: python -m pip install "sunpy[net,timeseries,visualization]"') from exc
    spec = cfg["data"]
    names = [c["name"] for c in spec["channels"]]
    quality = [c.get("quality_column") for c in spec["channels"] if c.get("quality_column")]
    if set(names) - {"xrsa", "xrsb"}:
        raise ValueError("GOES XRS channels must be xrsa and/or xrsb")
    start = pd.Timestamp(spec["start"], tz="UTC")
    end = pd.Timestamp(spec["end"], tz="UTC") + pd.Timedelta(days=1)
    raw = run_directory(cfg) / "sunpy_downloads"
    raw.mkdir(parents=True, exist_ok=True)
    path = resolve(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    written, records = 0, []
    # Month-sized requests bound memory; the final CSV is replaced only after success.
    cursor = start
    try:
        while cursor < end:
            stop = min(cursor + pd.offsets.MonthBegin(1), end)
            query = Fido.search(a.Time(cursor.to_pydatetime(), (stop - pd.Timedelta(microseconds=1)).to_pydatetime()),
                                a.Instrument.xrs, a.goes.SatelliteNumber(spec["satellite"]), a.Resolution("avg1m"))
            if sum(len(block) for block in query) == 0:
                raise ValueError(f"No GOES-{spec['satellite']} XRS files for {cursor.date()}..{stop.date()}; check satellite coverage")
            fetched = Fido.fetch(query, path=str(raw / "{file}"))
            if fetched.errors:
                raise RuntimeError(f"{len(fetched.errors)} SunPy downloads failed; no replacement CSV was published")
            series = ts.TimeSeries(list(fetched), concatenate=True)
            frame = series.to_dataframe().sort_index()
            frame.index = pd.to_datetime(frame.index, utc=True)
            frame = frame.loc[(frame.index >= cursor) & (frame.index < stop)]
            if frame.index.duplicated().any():
                raise ValueError("Duplicate GOES times; select one satellite/product explicitly")
            absent = set(names + quality) - set(frame)
            if absent:
                raise ValueError(f"GOES data missing required columns: {sorted(absent)}")
            # Normalize flux units using the TimeSeries unit metadata.
            import astropy.units as u
            for channel in spec["channels"]:
                unit = u.Unit(channel.get("unit", "W m-2"))
                factor = series.units[channel["name"]].to(unit)
                frame[channel["name"]] *= factor
            frame = frame[names + quality]
            frame.index.name = spec["timestamp_column"]
            frame.to_csv(temporary, mode="w" if written == 0 else "a", header=written == 0)
            written += len(frame)
            for item in fetched:
                from pathlib import Path
                p = Path(item)
                records.append({"file": p.name, "sha256": file_sha(p)})
            print(f"Fetched {cursor.date()}..{stop.date()}: {len(frame)} rows", flush=True)
            cursor = stop
        if not written:
            raise ValueError("No rows downloaded")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    write_json(path.with_suffix(".provenance.json"), {"source": "SunPy Fido / NOAA GOES XRS avg1m", "sunpy_version": sunpy.__version__,
               "satellite": spec["satellite"], "start": spec["start"], "end": spec["end"], "rows": written,
               "channels": spec["channels"], "input_sha256": file_sha(path), "downloaded_files": records})
    print(f"Saved {path}. Run prepare next.")
