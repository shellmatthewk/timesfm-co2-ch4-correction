"""Small numerical OISST/ERA5 subsets, source records, and CO2-matched presets."""
from __future__ import annotations

import ast
import concurrent.futures
import datetime as dt
import json
import hashlib
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .config import DEFAULTS, ROOT, file_sha, resolve, write_json

CLIMATE = ROOT / "data/climate"


def validate_sources(spec):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec["start"]) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec["end"]):
        raise ValueError("Source dates must be YYYY-MM-DD")
    if pd.Timestamp(spec["start"]) > pd.Timestamp(spec["end"]):
        raise ValueError("Source start date must not follow the end date")
    for source in ("sst", "era5"):
        points = spec[source]["points"]
        if not points or len({p["name"] for p in points}) != len(points):
            raise ValueError(f"{source} needs at least one point and unique names")
        for p in points:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", p["name"]):
                raise ValueError("Point names must use letters, numbers, underscores or hyphens")
            for coordinate, bound in (("latitude", 90), ("longitude", 180)):
                value = p[coordinate]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not -bound <= value <= bound:
                    raise ValueError(f"Invalid {coordinate} for {p['name']}")
    return spec


def _get_text(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(10, 45))
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _parse_ascii_grid(text):
    """Decode NOAA's DAP2 ASCII Grid; reject truncated or duplicate rows."""
    header = re.search(r"^sst\.sst\[(\d+)\]\[(\d+)\]\[(\d+)\]$", text, re.M)
    if header is None:
        raise ValueError("Unexpected NOAA ASCII grid header")
    shape = tuple(map(int, header.groups()))
    values = np.empty(shape, dtype=np.float32)
    seen = set()
    for match in re.finditer(r"^\[(\d+)\]\[(\d+)\], (.+)$", text, re.M):
        row = tuple(map(int, match.groups()[:2]))
        if row in seen or row[0] >= shape[0] or row[1] >= shape[1]:
            raise ValueError("Invalid or duplicate NOAA grid row")
        a = np.array([float(v) for v in match.group(3).split(",")], dtype=np.float32)
        if len(a) != shape[2]:
            raise ValueError("Incomplete NOAA grid longitude row")
        values[row] = a
        seen.add(row)
    if len(seen) != shape[0] * shape[1]:
        raise ValueError("Incomplete NOAA grid data")
    axes = {}
    for axis, count in zip(("time", "lat", "lon"), shape):
        match = re.search(rf"^sst\.{axis}\[{count}\]\n([^\n]+)", text, re.M)
        if match is None:
            raise ValueError(f"Missing NOAA {axis} coordinates")
        axes[axis] = np.array([float(v) for v in match.group(1).split(",")])
        if len(axes[axis]) != count:
            raise ValueError(f"Incomplete NOAA {axis} coordinates")
    return values, axes


def _sst_year(url, days, points, year):
    metadata = _get_text(url + ".das")
    title_match = re.search(r'String title "([^"]*)";', metadata)
    title = title_match.group(1) if title_match else ""
    historical_title = "NOAA High-resolution Blended Analysis: Daily Values using AVHRR only"
    if "Version 2.1" not in title and not (year < 2016 and title == historical_title):
        raise ValueError(f"Unexpected OISST source version: {title}")
    sst_attrs = re.search(r"\bsst \{(.*?)\n\s*\}", metadata, re.S)
    time_attrs = re.search(r"\btime \{(.*?)\n\s*\}", metadata, re.S)
    if sst_attrs is None or 'String units "degC";' not in sst_attrs.group(1):
        raise ValueError("Unexpected NOAA SST units")
    units = re.search(r'String units "days since ([^"]+)";', time_attrs.group(1) if time_attrs else "")
    if units is None:
        raise ValueError("Unexpected NOAA time units")
    epoch = pd.Timestamp(units.group(1))
    indices = [(int(np.clip(np.floor((p["latitude"] + 89.875) / .25 + .5), 0, 719)),
                int(np.clip(np.floor((p["longitude"] % 360 - .125) / .25 + .5), 0, 1439))) for p in points]

    def span(ids):
        ids = sorted(set(ids))
        step = math.gcd(*np.diff(ids).tolist()) if len(ids) > 1 else 1
        return ids[0], step, ids[-1]

    latitude, longitude = span([i[0] for i in indices]), span([i[1] for i in indices])
    cells = ((latitude[2] - latitude[0]) // latitude[1] + 1) * ((longitude[2] - longitude[0]) // longitude[1] + 1)
    # One sparse rectangle avoids rereading the full annual file once per point.
    # Arbitrary points that make a large rectangle use separate point requests.
    rectangles = [(latitude, longitude, list(range(len(points))))] if cells <= 1024 else [
        ((a, 1, a), (b, 1, b), [j]) for j, (a, b) in enumerate(indices)]
    chunks = []
    for month in sorted(set(days.month)):
        part = days[days.month == month]
        first = (part[0] - pd.Timestamp(year=year, month=1, day=1)).days
        last = first + len(part) - 1
        for lat, lon, sites in rectangles:
            query = f"sst[{first}:1:{last}][{lat[0]}:{lat[1]}:{lat[2]}][{lon[0]}:{lon[1]}:{lon[2]}]"
            chunks.append((part, sites, url + ".ascii?" + query))

    def chunk_data(chunk):
        part, sites, query_url = chunk
        payload = _get_text(query_url).replace("\r\n", "\n")
        data, axes = _parse_ascii_grid(payload)
        actual_dates = pd.DatetimeIndex(epoch + pd.to_timedelta(axes["time"], unit="D"))
        if not actual_dates.equals(part):
            raise ValueError("OISST response dates differ from requested dates")
        chosen, output = [], []
        for j in sites:
            a, b = indices[j]
            expected_lat, expected_lon = -89.875 + .25 * a, .125 + .25 * b
            lat_idx, lon_idx = np.flatnonzero(axes["lat"] == expected_lat), np.flatnonzero(axes["lon"] == expected_lon)
            if len(lat_idx) != 1 or len(lon_idx) != 1:
                raise ValueError("OISST response grid cells differ from requested cells")
            value = data[:, lat_idx[0], lon_idx[0]]
            if not np.isfinite(value).all() or np.any((value < -5) | (value > 50)):
                raise ValueError(f"Invalid ocean-point SST values for {points[j]['name']} in {year}")
            output.append(value)
            chosen.append({"name": points[j]["name"], "requested_latitude": points[j]["latitude"],
                           "requested_longitude": points[j]["longitude"], "latitude": expected_lat,
                           "longitude": ((expected_lon + 180) % 360) - 180})
        return pd.DataFrame(np.stack(output, -1), index=part, columns=[points[j]["name"] for j in sites]), chosen, {
            "url": query_url, "response_sha256": hashlib.sha256(payload.encode()).hexdigest(), "days": len(part)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(chunk_data, chunks))
    frame = pd.concat([r[0] for r in results]).groupby(level=0).first().reindex(days)
    names = [p["name"] for p in points]
    chosen = {p["name"]: p for r in results for p in r[1]}
    subset = xr.Dataset({"sst": (("time", "site"), frame[names].to_numpy(dtype=np.float32))},
                       coords={"time": days, "site": names,
                               "latitude": ("site", [chosen[n]["latitude"] for n in names]),
                               "longitude": ("site", [chosen[n]["longitude"] for n in names])},
                       attrs={"source_title": title, "requested_and_selected_points": json.dumps([chosen[n] for n in names], sort_keys=True),
                              "subset_requests": json.dumps([r[2] for r in results], sort_keys=True),
                              "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest()})
    subset.sst.attrs["units"] = "degC"
    return subset


def co2_protocol():
    """Read constants from the current CO2 source without importing its runtime dependencies."""
    path = ROOT / "multigas/src/mg/multigas.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"FOLDS", "TRAIN_START", "CONFIRM_FOLDS", "CONFIRM_TRAIN_START"}
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in wanted:
                    result[name.id] = ast.literal_eval(node.value)
    if set(result) != wanted:
        raise ValueError("CO2 protocol constants changed; update the climate setup explicitly")
    window_source = ROOT / "multigas/src/mg/data.py"
    functions = [node for node in ast.parse(window_source.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and node.name == "build_windows"]
    if len(functions) != 1:
        raise ValueError("CO2 window builder changed; update the climate setup explicitly")
    args = functions[0].args
    defaults = dict(zip([a.arg for a in args.args][-len(args.defaults):], args.defaults))
    sizes = {name: ast.literal_eval(defaults[name]) for name in ("context", "horizon")}
    return {"main": {"test_years": list(result["FOLDS"]), "train_start": result["TRAIN_START"]},
            "confirm": {"test_years": list(result["CONFIRM_FOLDS"]), "train_start": result["CONFIRM_TRAIN_START"]},
            "source_file": str(path), "source_sha256": file_sha(path), **sizes,
            "window_source_file": str(window_source), "window_source_sha256": file_sha(window_source),
            "validation": "Y-1", "training_end": "end of Y-2"}


def _dates(spec):
    return pd.date_range(spec["start"], spec["end"], freq="1D")


def _publish_csv(path, frame, metadata, expected):
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate dates for {path.name}")
    frame = frame.sort_index().reindex(expected)
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Missing or nonfinite daily values for {path.name}; refusing an incomplete dataset")
    frame.index.name = "timestamp"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, float_format="%.9g")
    temporary.replace(path)
    metadata.update({"input_sha256": file_sha(path), "rows": len(frame), "channels": list(frame),
                     "measurement_dates": [str(frame.index[0].date()), str(frame.index[-1].date())],
                     "input_timestamp_convention": "measurement-day start, UTC; runner labels completed bins by end time",
                     "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    write_json(path.with_suffix(".provenance.json"), metadata)
    return frame, metadata


def copy_ccai(spec):
    source = resolve(spec["ccai_directory"]) / "data"
    destination = CLIMATE / "ccai_reference"
    destination.mkdir(parents=True, exist_ok=True)
    record = {}
    for name in ("era5_daily_t2m_2010_2021.nc", "era5_daily_tp_mm_2010_2021.nc"):
        original, copied = source / name, destination / name
        if not original.is_file():
            raise FileNotFoundError(f"CCAI reference missing: {original}")
        digest = file_sha(original)
        if copied.exists() and file_sha(copied) != digest:
            raise ValueError(f"A different reference already exists at {copied}; preserve it separately")
        if not copied.exists():
            shutil.copy2(original, copied)
        if file_sha(copied) != digest:
            raise ValueError(f"Copied CCAI checksum mismatch: {name}")
        with xr.open_dataset(copied) as dataset:
            record[name] = {"source": str(original), "copy": str(copied), "sha256": digest,
                            "bytes": copied.stat().st_size, "dimensions": dict(dataset.sizes),
                            "date_span": [str(dataset.time.values[0]), str(dataset.time.values[-1])],
                            "variables": {k: {a: str(dataset[k].attrs[a]) for a in ("units", "long_name", "source_units", "aggregation")
                                              if a in dataset[k].attrs} for k in dataset.data_vars}}
        print(f"Copied/verified CCAI reference: {name}", flush=True)
    write_json(destination / "provenance.json", record)
    return record


def download_sst(spec):
    expected = _dates(spec)
    raw = CLIMATE / "raw/oisst"
    raw.mkdir(parents=True, exist_ok=True)
    frames, records, locations = [], [], None
    names = [p["name"] for p in spec["sst"]["points"]]
    for year in range(expected[0].year, expected[-1].year + 1):
        days = expected[expected.year == year]
        path = raw / f"sst_points_{year}.nc"
        url = f"{spec['sst']['base_url']}/sst.day.mean.{year}.nc"
        request = json.dumps({"dates": [str(days[0].date()), str(days[-1].date())], "points": spec["sst"]["points"]}, sort_keys=True)
        if path.is_file():
            with xr.open_dataset(path) as saved:
                if saved.attrs.get("request") != request or saved.attrs.get("source_url") != url:
                    digest = hashlib.sha256((url + request).encode()).hexdigest()[:12]
                    path = raw / f"sst_points_{year}_{digest}.nc"
        if not path.is_file():
            print(f"SST {year}: retrieving bounded monthly subsets", flush=True)
            subset = _sst_year(url, days, spec["sst"]["points"], year)
            subset.attrs.update(source_url=url, request=request)
            temporary = path.with_suffix(".part.nc")
            subset.to_netcdf(temporary, engine="netcdf4")
            temporary.replace(path)
        with xr.open_dataset(path) as saved:
            if saved.attrs.get("request") != request or saved.attrs.get("source_url") != url:
                raise ValueError(f"SST subset cache request mismatch: {path}")
            if not pd.DatetimeIndex(saved.time.values).equals(days) or list(saved.site.values) != names or saved.sst.attrs.get("units") != "degC":
                raise ValueError(f"SST subset cache dates, sites or units differ: {path}")
            if not np.isfinite(saved.sst.values).all() or np.any((saved.sst.values < -5) | (saved.sst.values > 50)):
                raise ValueError(f"Invalid cached ocean-point SST values: {path}")
            frames.append(pd.DataFrame(saved.sst.values, index=pd.DatetimeIndex(saved.time.values), columns=names))
            selected_locations = json.loads(saved.attrs["requested_and_selected_points"])
            if locations is not None and selected_locations != locations:
                raise ValueError("SST grid locations changed between years")
            locations = selected_locations
            records.append({"year": year, "file": str(path.relative_to(ROOT)), "sha256": file_sha(path),
                            "source_url": url, "source_title": saved.attrs["source_title"], "days": len(days),
                            "subset_requests": json.loads(saved.attrs.get("subset_requests", "[]"))})
        print(f"SST {year}: verified {len(days)} days at {len(names)} ocean points", flush=True)
    return _publish_csv(CLIMATE / "oisst_daily.csv", pd.concat(frames),
               {"dataset": "NOAA OISST v2.1, daily SST, NOAA PSL distribution", "unit": "degC",
                "dataset_doi": "10.25921/RE9P-PT57", "product_url": "https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
                "locations": locations, "subset_files": records, "interpretation": "Retrospective analysed SST, not direct station observations"}, expected)


def download_era5(spec):
    expected = _dates(spec)
    raw = CLIMATE / "raw/era5_open_meteo"
    raw.mkdir(parents=True, exist_ok=True)

    def point_data(point):
        params = {"latitude": point["latitude"], "longitude": point["longitude"],
                  "start_date": spec["start"], "end_date": spec["end"], "timezone": "UTC", "models": "era5",
                  "elevation": "nan", "cell_selection": "nearest",
                  "daily": "temperature_2m_mean,precipitation_sum"}
        path = raw / f"{point['name']}.json"
        meta_path = raw / f"{point['name']}.request.json"
        if path.is_file() and meta_path.is_file():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing.get("params") != params or existing.get("base_url") != spec["era5"]["base_url"]:
                digest = hashlib.sha256(json.dumps([spec["era5"]["base_url"], params], sort_keys=True).encode()).hexdigest()[:12]
                path = raw / f"{point['name']}_{digest}.json"
                meta_path = path.with_suffix(".request.json")
        if path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("params") != params or meta.get("base_url") != spec["era5"]["base_url"]:
                raise ValueError(f"ERA5 cache settings changed: {path}; use a separate cache directory")
            if meta["sha256"] != file_sha(path):
                raise ValueError(f"ERA5 cache checksum mismatch: {path}")
            response = json.loads(path.read_text(encoding="utf-8"))
        else:
            for attempt in range(3):
                try:
                    result = requests.get(spec["era5"]["base_url"], params=params, timeout=(15, 180))
                    result.raise_for_status()
                    response = result.json()
                    if response.get("error"):
                        raise ValueError(str(response.get("reason")))
                    write_json(path, response)
                    write_json(meta_path, {"base_url": spec["era5"]["base_url"], "params": params,
                                          "response_url": result.url, "sha256": file_sha(path)})
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        if response["utc_offset_seconds"] != 0:
            raise ValueError("ERA5 daily statistics must use UTC")
        if response["daily_units"]["temperature_2m_mean"] != "°C" or response["daily_units"]["precipitation_sum"] != "mm":
            raise ValueError("Unexpected ERA5 API units")
        index = pd.DatetimeIndex(pd.to_datetime(response["daily"]["time"]))
        if not index.equals(expected):
            raise ValueError(f"Incomplete daily ERA5 date coverage for {point['name']}")
        frame = pd.DataFrame({"temperature": response["daily"]["temperature_2m_mean"],
                              "precipitation": response["daily"]["precipitation_sum"]}, index=index, dtype=float)
        if not np.isfinite(frame.values).all() or (frame.precipitation < 0).any():
            raise ValueError(f"Invalid ERA5 values for {point['name']}")
        if abs(response["latitude"] - point["latitude"]) > 0.13 or abs(response["longitude"] - point["longitude"]) > 0.13:
            raise ValueError("ERA5 API selected a different grid cell than requested")
        record = {"name": point["name"], "requested_latitude": point["latitude"], "requested_longitude": point["longitude"],
                  "latitude": response["latitude"], "longitude": response["longitude"],
                  "source_file": str(path.relative_to(ROOT)), "sha256": file_sha(path), "params": params}
        print(f"ERA5 {point['name']}: verified {len(index)} daily values per variable", flush=True)
        return point["name"], frame, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(point_data, spec["era5"]["points"]))
    records = [r[2] for r in results]
    metadata = {"dataset": "ERA5, redistributed by Open-Meteo Historical Weather API", "model": "era5",
                "source_url": "https://open-meteo.com/en/docs/historical-weather-api",
                "elevation_downscaling": False, "grid_selection": "nearest", "locations": records,
                "interpretation": "Retrospective reanalysis; API precision/processing may differ from the copied CCAI fields",
                "source_consistency": "Entire date range retrieved from one API/model; CCAI and API values are not stitched"}
    outputs = {}
    for variable, filename, unit in (("temperature", "era5_temperature_daily.csv", "degC"),
                                     ("precipitation", "era5_precipitation_daily.csv", "mm day-1")):
        frame = pd.DataFrame({name: data[variable] for name, data, _ in results})
        outputs[variable] = _publish_csv(CLIMATE / filename, frame, {**metadata, "variable": variable, "unit": unit}, expected)
    return outputs


def compare_ccai(spec, outputs):
    """Quantify source differences; never silently replace copied reference values."""
    overlap = {}
    for variable, filename, field, offset in (("temperature", "era5_daily_t2m_2010_2021.nc", "t2m", -273.15),
                                              ("precipitation", "era5_daily_tp_mm_2010_2021.nc", "tp", 0.0)):
        frame = outputs[variable][0].loc["2010-01-01":"2021-12-31"]
        locations = {p["name"]: p for p in outputs[variable][1]["locations"]}
        exact = {}
        with xr.open_dataset(CLIMATE / "ccai_reference" / filename) as dataset:
            expected_unit = "K" if variable == "temperature" else "mm day-1"
            if dataset[field].attrs.get("units") != expected_unit:
                raise ValueError(f"Unexpected copied CCAI {variable} units")
            for point in spec["era5"]["points"]:
                chosen = locations[point["name"]]
                values = dataset[field].sel(latitude=chosen["latitude"], longitude=chosen["longitude"], method="nearest")
                if not np.isclose(float(values.latitude), chosen["latitude"], atol=1e-6, rtol=0) or not np.isclose(float(values.longitude), chosen["longitude"], atol=1e-6, rtol=0):
                    overlap.setdefault(variable, {})[point["name"]] = {"comparable": False, "reason": "API grid cell is outside the copied CCAI grid"}
                    continue
                if frame.empty:
                    overlap.setdefault(variable, {})[point["name"]] = {"comparable": False, "reason": "No dates overlap 2010-2021"}
                    continue
                values = values.sel(time=slice("2010-01-01", "2021-12-31")).load()
                series = pd.Series(values.values.astype(np.float64) + offset, index=pd.DatetimeIndex(values.time.values))
                exact[point["name"]] = series
                diff = frame[point["name"]] - series.reindex(frame.index)
                if not np.isfinite(diff.values).all():
                    raise ValueError("CCAI overlap contains missing values")
                overlap.setdefault(variable, {})[point["name"]] = {"days": len(diff), "mean_difference": float(diff.mean()),
                      "mae": float(diff.abs().mean()), "rmse": float(np.sqrt((diff ** 2).mean())), "max_abs_difference": float(diff.abs().max())}
        if exact:
            _publish_csv(CLIMATE / f"era5_{variable}_ccai_daily.csv", pd.DataFrame(exact),
                  {"dataset": "Copied CCAI ERA5 daily reference", "source_file": filename,
                   "source_sha256": file_sha(CLIMATE / "ccai_reference" / filename), "variable": variable,
                   "unit": "degC" if variable == "temperature" else "mm day-1", "locations": [locations[n] for n in exact]},
                  pd.date_range("2010-01-01", "2021-12-31"))
    write_json(CLIMATE / "era5_ccai_overlap_comparison.json", {"notes": "API values are compared with copied CCAI grid cells; differences are not forecast errors.", "variables": overlap})
    return overlap


def make_configs(spec, overwrite=False):
    protocol = co2_protocol()
    datasets = {"oisst": ("oisst_daily.csv", spec["sst"]["points"], "degC", "identity", None),
                "era5_temperature": ("era5_temperature_daily.csv", spec["era5"]["points"], "degC", "identity", None),
                "era5_precipitation": ("era5_precipitation_daily.csv", spec["era5"]["points"], "mm day-1", "log1p", 0.0)}
    paths = []
    for tag, (filename, points, unit, transform, minimum) in datasets.items():
        for period in ("main", "confirm"):
            channels = [{"name": p["name"], "unit": unit, "transform": transform,
                         **({"minimum": minimum} if minimum is not None else {})} for p in points]
            cfg = {"name": f"{tag}_{period}", "output_root": "climate_runs",
                   "data": {"source": "csv", "path": f"data/climate/{filename}", "start": spec["start"], "end": spec["end"],
                            "cadence": "1D", "native_cadence": "1D", "min_bin_coverage": 1.0, "channels": channels},
                   "windows": {**DEFAULTS["windows"], "context": protocol["context"], "horizon": protocol["horizon"], "stride": 1},
                   "evaluation": {**DEFAULTS["evaluation"], "train_start": protocol[period]["train_start"], "test_years": protocol[period]["test_years"], "date_basis": "bin_start"},
                   "model": dict(DEFAULTS["model"]), "training": dict(DEFAULTS["training"])}
            path = ROOT / "configs" / f"{tag}_{period}.json"
            if overwrite or not path.exists():
                write_json(path, cfg)
            paths.append(path)
    write_json(CLIMATE / "co2_date_protocol.json", protocol)
    return paths
