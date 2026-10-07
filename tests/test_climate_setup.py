"""Source integrity and climate-specific preprocessing checks; no remote calls."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_forecasting import climate_data
from solar_forecasting.config import DEFAULTS, file_sha, load_config, validate
from solar_forecasting.data import build_windows, evaluation_times, fold_indices, inverse, load_windows, prepare, read_panel


def rain_config(tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["output_root"] = str(tmp_path / "runs")
    cfg["data"].update({"path": str(tmp_path / "rain.csv"), "start": "2020-01-01", "end": "2020-03-01",
                        "native_cadence": "1D", "min_bin_coverage": 1.0,
                        "channels": [{"name": "rain", "unit": "mm day-1", "transform": "log1p", "minimum": 0}]})
    return validate(cfg)


def test_precipitation_zeros_are_observed_and_predictions_are_nonnegative(tmp_path):
    cfg = rain_config(tmp_path)
    pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=4), "rain": [0, 1, -1, 9]}).to_csv(cfg["data"]["path"], index=False)
    times, model, physical, audit = read_panel(cfg)
    assert times[0] == np.datetime64("2020-01-02")
    np.testing.assert_allclose(model[[0, 1, 3], 0], np.log1p([0, 1, 9]))
    assert np.isnan(model[2, 0]) and audit["observed_bins"]["rain"] == 3
    np.testing.assert_allclose(inverse(np.array([[[-2., 0., np.log(10.)]]]), cfg["data"]["channels"], 1), [[[0, 0, 9]]])
    np.testing.assert_allclose(inverse(np.array([[[-5.]]]), [{"name": "temperature"}], 1), [[[-5.]]])


def test_provenance_is_bound_to_input_and_cached_windows(tmp_path):
    cfg = rain_config(tmp_path)
    path = Path(cfg["data"]["path"])
    dates = pd.date_range("2020-01-01", "2020-03-01")
    _, source = climate_data._publish_csv(path, pd.DataFrame({"rain": np.zeros(len(dates))}, index=dates),
                                         {"dataset": "test source"}, dates)
    windows, audit = prepare(cfg)
    assert windows["mask"].all() and audit["source_provenance"] == source
    sidecar = path.with_suffix(".provenance.json")
    source["dataset"] = "changed source description"
    sidecar.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance changed"):
        load_windows(cfg)
    source["input_sha256"] = "incorrect checksum"
    sidecar.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its source provenance"):
        read_panel(cfg)


def test_incomplete_download_does_not_replace_published_csv(tmp_path):
    path = tmp_path / "published.csv"
    path.write_text("previous complete data", encoding="utf-8")
    days = pd.date_range("2020-02-28", "2020-03-01")
    with pytest.raises(ValueError, match="incomplete dataset"):
        climate_data._publish_csv(path, pd.DataFrame({"x": [1., 2.]}, index=days[:2]), {}, days)
    assert path.read_text(encoding="utf-8") == "previous complete data"


def test_configs_read_current_co2_protocol_and_keep_fold_boundaries(tmp_path, monkeypatch):
    protocol = climate_data.co2_protocol()
    monkeypatch.setattr(climate_data, "ROOT", tmp_path)
    monkeypatch.setattr(climate_data, "CLIMATE", tmp_path / "data/climate")
    monkeypatch.setattr(climate_data, "co2_protocol", lambda: protocol)
    points = [{"name": "one", "latitude": 40.5, "longitude": -3.75}]
    spec = {"start": "1999-12-01", "end": "2025-12-31", "sst": {"points": points}, "era5": {"points": points}}
    paths = climate_data.make_configs(spec)
    assert len(paths) == 6
    first = pd.date_range(spec["start"], spec["end"]).to_numpy()
    target_times = first[:, None] + (np.arange(protocol["horizon"])[None, :] + 1).astype("timedelta64[D]")
    for path in paths:
        cfg = load_config(path)
        period = "confirm" if "confirm" in path.stem else "main"
        assert cfg["evaluation"]["test_years"] == protocol[period]["test_years"]
        assert cfg["evaluation"]["train_start"] == protocol[period]["train_start"]
        assert cfg["windows"]["context"] == protocol["context"]
        assert cfg["windows"]["horizon"] == protocol["horizon"]
        assert cfg["evaluation"]["date_basis"] == "bin_start"
        evaluated, _ = evaluation_times({"target_times": target_times}, cfg)
        for year in cfg["evaluation"]["test_years"]:
            fold = fold_indices({"target_times": target_times}, year, cfg)
            assert all(len(idx) > 0 for idx in fold.values())
            for part, idx in fold.items():
                lo, hi = {"train": (protocol[period]["train_start"], f"{year - 1}-01-01"),
                          "validation": (f"{year - 1}-01-01", f"{year}-01-01"),
                          "test": (f"{year}-01-01", f"{year + 1}-01-01")}[part]
                assert (evaluated[idx] >= np.datetime64(lo)).all()
                assert (evaluated[idx] < np.datetime64(hi)).all()
    edited = json.loads(paths[0].read_text(encoding="utf-8"))
    edited["training"]["epochs"] = 2
    paths[0].write_text(json.dumps(edited), encoding="utf-8")
    climate_data.make_configs(spec)
    assert load_config(paths[0])["training"]["epochs"] == 2
    climate_data.make_configs(spec, overwrite=True)
    assert load_config(paths[0])["training"]["epochs"] == DEFAULTS["training"]["epochs"]


def test_measurement_year_edges_match_co2_without_moving_forecast_availability(tmp_path):
    cfg = rain_config(tmp_path)
    cfg["data"].update(start="2019-12-01", end="2020-12-31")
    cfg["evaluation"].update(date_basis="bin_start", test_years=[2020])
    dates = pd.date_range("2019-12-01", "2020-12-31")
    pd.DataFrame({"timestamp": dates, "rain": np.arange(len(dates), dtype=float)}).to_csv(cfg["data"]["path"], index=False)
    times, model, physical, _ = read_panel(cfg)
    windows, _ = build_windows(times, model, physical, cfg["windows"])
    test = fold_indices(windows, 2020, cfg)["test"]
    evaluated, origins = evaluation_times(windows, cfg)
    assert evaluated[test[0], 0] == np.datetime64("2020-01-01")
    assert evaluated[test[-1], -1] == np.datetime64("2020-12-31")
    assert windows["origin"][test[0]] == np.datetime64("2020-01-01")
    assert origins[test[0]] == np.datetime64("2019-12-31")
    assert windows["x"][test[0], 0, -1] == np.log1p(30.)
    assert windows["physical_y"][test[0], 0, 0] == 31.
    assert windows["physical_y"][test[-1], 0, -1] == len(dates) - 1


def test_era5_requests_explicit_model_and_reuses_verified_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(climate_data, "ROOT", tmp_path)
    monkeypatch.setattr(climate_data, "CLIMATE", tmp_path / "data/climate")
    spec = {"start": "2020-02-28", "end": "2020-03-01", "era5": {"base_url": "https://example.test/archive",
            "points": [{"name": "point", "latitude": 40.5, "longitude": -3.75}]}}
    calls = []

    class Response:
        url = "https://example.test/archive?models=era5"
        def raise_for_status(self):
            pass
        def json(self):
            return {"utc_offset_seconds": 0, "latitude": 40.5, "longitude": -3.75,
                    "daily_units": {"temperature_2m_mean": "°C", "precipitation_sum": "mm"},
                    "daily": {"time": ["2020-02-28", "2020-02-29", "2020-03-01"],
                              "temperature_2m_mean": [-1, 0, 1], "precipitation_sum": [0, 0.1, 2]}}

    def get(url, params, timeout):
        calls.append(params)
        return Response()

    monkeypatch.setattr(climate_data.requests, "get", get)
    output = climate_data.download_era5(spec)
    assert calls[0]["models"] == "era5" and calls[0]["elevation"] == "nan"
    assert calls[0]["cell_selection"] == "nearest" and calls[0]["timezone"] == "UTC"
    assert output["precipitation"][0].iloc[0, 0] == 0
    assert output["temperature"][1]["elevation_downscaling"] is False
    climate_data.download_era5(spec)
    assert len(calls) == 1
    raw = tmp_path / "data/climate/raw/era5_open_meteo/point.json"
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cache checksum mismatch"):
        climate_data.download_era5(spec)


@pytest.mark.parametrize("year,title,success", [
    (1999, "NOAA High-resolution Blended Analysis: Daily Values using AVHRR only", True),
    (2016, "NOAA High-resolution Blended Analysis: Daily Values using AVHRR only", False),
    (2016, "NOAA OISST Analysis, Version 2.1", True),
])
def test_sst_historical_version_exception_is_limited(tmp_path, monkeypatch, year, title, success):
    monkeypatch.setattr(climate_data, "ROOT", tmp_path)
    monkeypatch.setattr(climate_data, "CLIMATE", tmp_path / "data/climate")
    dates = pd.date_range(f"{year}-12-30", f"{year}-12-31")
    spec = {"start": str(dates[0].date()), "end": str(dates[-1].date()), "sst": {"base_url": "https://example.test/sst",
            "points": [{"name": "ocean", "latitude": 40, "longitude": -40}]}}
    offsets = (dates - pd.Timestamp("1800-01-01")).days
    metadata = ('Attributes {\n time {\n String units "days since 1800-01-01 00:00:00";\n }\n'
                ' sst {\n String units "degC";\n }\n NC_GLOBAL {\n String title "' + title + '";\n }\n}')
    payload = (f"sst.sst[2][1][1]\n[0][0], 15\n[1][0], 16\n\nsst.time[2]\n{offsets[0]}, {offsets[1]}\n\n"
               "sst.lat[1]\n40.125\n\nsst.lon[1]\n320.125\n")
    monkeypatch.setattr(climate_data, "_get_text", lambda url: metadata if url.endswith(".das") else payload)
    if success:
        frame, provenance = climate_data.download_sst(spec)
        assert len(frame) == 2 and provenance["locations"][0]["longitude"] == -39.875
        assert provenance["subset_files"][0]["source_title"] == title
    else:
        with pytest.raises(ValueError, match="Unexpected OISST source version"):
            climate_data.download_sst(spec)


@pytest.mark.parametrize("rows", ["[0][0], 15\n", "[0][0], 15\n[0][0], 16\n"])
def test_sst_rejects_truncated_and_duplicate_payload_rows(rows):
    payload = "sst.sst[2][1][1]\n" + rows + "\nsst.time[2]\n0, 1\n\nsst.lat[1]\n40.125\n\nsst.lon[1]\n320.125\n"
    with pytest.raises(ValueError, match="Incomplete|duplicate"):
        climate_data._parse_ascii_grid(payload)


def test_ccai_copy_checksums_and_overlap_use_actual_api_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(climate_data, "ROOT", tmp_path)
    monkeypatch.setattr(climate_data, "CLIMATE", tmp_path / "data/climate")
    source = tmp_path / "ccai/data"
    source.mkdir(parents=True)
    dates = pd.date_range("2010-01-01", "2021-12-31")
    for filename, variable, value, unit in (("era5_daily_t2m_2010_2021.nc", "t2m", 273.15, "K"),
                                             ("era5_daily_tp_mm_2010_2021.nc", "tp", .2, "mm day-1")):
        ds = xr.Dataset({variable: (("time", "latitude", "longitude"), np.full((len(dates), 1, 1), value))},
                        coords={"time": dates, "latitude": [40.5], "longitude": [-3.75]})
        ds[variable].attrs["units"] = unit
        ds.to_netcdf(source / filename)
    points = [{"name": "near", "latitude": 40.44, "longitude": -3.72},
              {"name": "outside", "latitude": 0, "longitude": -100}]
    spec = {"ccai_directory": str(source.parent), "era5": {"points": points}}
    record = climate_data.copy_ccai(spec)
    assert all(file_sha(p["source"]) == file_sha(p["copy"]) == p["sha256"] for p in record.values())
    locations = [{"name": "near", "latitude": 40.5, "longitude": -3.75}, points[1]]
    outputs = {"temperature": (pd.DataFrame({"near": np.ones(len(dates)), "outside": np.ones(len(dates))}, index=dates), {"locations": locations}),
               "precipitation": (pd.DataFrame({"near": np.ones(len(dates)) * 1.2, "outside": np.zeros(len(dates))}, index=dates), {"locations": locations})}
    comparison = climate_data.compare_ccai(spec, outputs)
    assert comparison["temperature"]["near"]["days"] == 4383
    assert comparison["temperature"]["near"]["mae"] == pytest.approx(1.)
    assert comparison["precipitation"]["near"]["mae"] == pytest.approx(1.)
    assert comparison["temperature"]["outside"]["comparable"] is False
    copied = Path(record["era5_daily_tp_mm_2010_2021.nc"]["copy"])
    copied.write_bytes(b"changed reference")
    with pytest.raises(ValueError, match="different reference already exists"):
        climate_data.copy_ccai(spec)
