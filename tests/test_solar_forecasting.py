"""Scientific regression tests for configurable dimensions and data boundaries."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_forecasting.config import DEFAULTS, identity, load_config, run_directory, validate
from solar_forecasting.data import (build_windows, fill_history, fold_indices, inverse, load_windows,
                                    prepare, read_panel)
from solar_forecasting.models import (ForecastHead, apply_context, crps_loss, energy_score,
                                      fit_context, native_samples)
from solar_forecasting.scoring import per_window

torch.set_num_threads(2)


def config(tmp_path, channels=1, horizon=7):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["name"] = "test"
    cfg["output_root"] = str(tmp_path / "runs")
    cfg["data"].update({"path": str(tmp_path / "input.csv"), "start": "2018-01-01", "end": "2021-04-30",
                        "native_cadence": "1D", "min_bin_coverage": 1.0,
                        "channels": [{"name": f"flux_{j}", "unit": "test units", "transform": "identity"} for j in range(channels)]})
    cfg["windows"].update({"context": 21, "horizon": horizon})
    cfg["evaluation"].update({"test_years": [2021], "train_start": "2018-01-01", "bootstrap_draws": 50,
                             "max_train_windows": 8, "max_validation_windows": 4, "max_test_windows": 6})
    cfg["model"].update({"noise_dim": max(16, horizon), "hidden_dim": 32})
    cfg["training"].update({"seeds": [17], "epochs": 2, "patience": 2, "batch_size": 4,
                            "head_train_samples": 4, "adjustment_train_samples": 4,
                            "validation_samples": 8, "evaluation_samples": 8, "sample_chunk": 3})
    return validate(cfg)


def write_input(cfg):
    dates = pd.date_range(cfg["data"]["start"], cfg["data"]["end"], freq="1D")
    k = np.arange(len(dates))
    rows = {"timestamp": dates}
    for j, channel in enumerate(cfg["data"]["channels"]):
        rows[channel["name"]] = 10 + j + np.sin(k / 8 + j)
    pd.DataFrame(rows).to_csv(cfg["data"]["path"], index=False)


def fake_encoding(windows):
    # Test fixture only: never used by the production TimesFM loader.
    b, c, _ = windows["x"].shape
    h = windows["y"].shape[-1]
    median = np.repeat(windows["x"][:, :, -1, None], h, axis=-1).astype(np.float32)
    q = median[..., None] + np.linspace(-0.5, 0.5, 9).astype(np.float32)
    features = np.zeros((b, c, 8), dtype=np.float32)
    features[..., 0] = windows["x"].mean(-1)
    return {"features": features, "center": median, "scale": np.ones((b, c, 1), dtype=np.float32),
            "quantiles": q, "native_weight": np.zeros((h, 8), dtype=np.float32), "native_bias": np.zeros(h, dtype=np.float32)}


@pytest.mark.parametrize("channels,horizon", [(1, 1), (1, 7), (3, 21)])
def test_windows_change_sizes_without_future_filling(tmp_path, channels, horizon):
    cfg = config(tmp_path, channels, horizon)
    count = 70
    times = np.arange(count).astype("timedelta64[D]") + np.datetime64("2020-01-01")
    values = np.tile(np.arange(count, dtype=float)[:, None], (1, channels))
    values[cfg["windows"]["context"], 0] = np.nan
    w, excluded = build_windows(times, values, values, cfg["windows"])
    assert w["x"].shape[1:] == (channels, 21)
    assert w["y"].shape[1:] == (channels, horizon)
    if horizon == 1:
        assert excluded["no_target"] == 1  # A window with no labels is discarded.
    else:
        assert not w["mask"][0, 0, 0]
        assert np.isnan(w["y"][0, 0, 0])
    changed = values.copy()
    changed[21:] += 1000
    other, _ = build_windows(times, changed, changed, cfg["windows"])
    np.testing.assert_array_equal(w["x"][0], other["x"][0])


def test_long_gaps_rejected_and_history_only_interpolated(tmp_path):
    cfg = config(tmp_path)
    spec = cfg["windows"]
    x = np.arange(21, dtype=float)[None]
    x[0, 3:5] = np.nan
    filled, imputed = fill_history(x, spec)
    np.testing.assert_allclose(filled[0, 3:5], [3, 4])
    assert imputed.sum() == 2
    x[0, 3:12] = np.nan
    with pytest.raises(ValueError, match="internal_gap"):
        fill_history(x, spec)


def test_quality_coverage_log_transform_and_availability_timestamp(tmp_path):
    cfg = config(tmp_path)
    cfg["data"].update({"start": "2020-01-01", "end": "2020-01-01", "cadence": "1h", "native_cadence": "15min",
                        "min_bin_coverage": 0.8, "channels": [{"name": "x", "unit": "W m-2", "transform": "log10", "quality_column": "q"}]})
    pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=8, freq="15min"),
                  "x": [1, 10, 100, 1000, 10, 10, 10, 10], "q": [0, 0, 0, 1, 0, 0, 0, 0]}).to_csv(cfg["data"]["path"], index=False)
    times, model, physical, _ = read_panel(cfg)
    assert times[0] == np.datetime64("2020-01-01T01:00:00")
    assert np.isnan(model[0, 0])  # Only 3 of 4 good observations: below 80%.
    assert model[1, 0] == 1 and physical[1, 0] == 10
    np.testing.assert_allclose(inverse(np.array([[[1.0]]]), cfg["data"]["channels"], 1), 10)


def test_folds_do_not_cross_years_and_amounts_are_capped(tmp_path):
    cfg = config(tmp_path)
    cfg["evaluation"]["train_years"] = 1
    write_input(cfg)
    w, _ = prepare(cfg)
    f = fold_indices(w, 2021, cfg)
    assert {p: len(i) for p, i in f.items()} == {"train": 8, "validation": 4, "test": 6}
    assert np.all(w["target_times"][f["train"]] >= np.datetime64("2019-01-01"))
    assert np.all(w["target_times"][f["train"]] < np.datetime64("2020-01-01"))
    assert np.all(w["target_times"][f["validation"]] < np.datetime64("2021-01-01"))
    assert np.all(w["target_times"][f["test"]] >= np.datetime64("2021-01-01"))


def test_preprocessing_statistics_do_not_read_test_values():
    raw = np.array([[[1., np.nan]], [[3., 5.]], [[1000., 999.]]], dtype=np.float32)
    stats = fit_context(raw, np.array([0, 1]))
    assert stats["fill"][1] == 5
    altered = raw.copy()
    altered[2] = -100000
    other = fit_context(altered, np.array([0, 1]))
    for k in stats:
        np.testing.assert_array_equal(stats[k], other[k])
    assert np.isfinite(apply_context(raw, stats)).all()


@pytest.mark.parametrize("channels,horizon", [(1, 7), (3, 21), (2, 40)])
@pytest.mark.parametrize("method", ["basic", "residual", "T1", "N1"])
def test_dynamic_heads_warm_start_and_gradients(tmp_path, channels, horizon, method):
    cfg = config(tmp_path, channels, horizon)
    b, d, s = 2, 8, 8
    e = {"features": torch.zeros(b, channels, d), "center": torch.full((b, channels, horizon), 2.),
         "scale": torch.ones(b, channels, 1), "quantiles": torch.linspace(1, 3, 9).expand(b, channels, horizon, 9)}
    context = torch.zeros(b, channels, 3 + channels)
    weight, bias = np.zeros((horizon, d), dtype=np.float32), np.zeros(horizon, dtype=np.float32)
    head = ForecastHead(method, d, context.shape[-1], horizon, cfg["model"], 17, weight, bias)
    x = head(e, context, s, torch.Generator().manual_seed(17))
    assert x.shape == (b, s, channels, horizon)
    if method in ("residual", "N1"):
        torch.testing.assert_close(torch.quantile(x, .5, dim=1), e["quantiles"][..., 4])
    if method == "T1":
        reference = native_samples(e["quantiles"], s, torch.Generator().manual_seed(17))
        torch.testing.assert_close(x, reference)
    if method == "basic":
        noise = torch.randn(b, s, 1, cfg["model"]["noise_dim"], generator=torch.Generator().manual_seed(17))
        torch.testing.assert_close(x, e["center"][:, None] + .05 * noise[..., :horizon].expand(-1, -1, channels, -1))
    y = torch.full((b, channels, horizon), 2.2)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 0, 0] = False
    y[0, 0, 0] = float("nan")
    loss = energy_score(x, y, mask, torch.ones(channels)) + crps_loss(x, y, mask, torch.ones(channels))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


def test_scoring_uses_original_units_and_observed_cells():
    samples = np.ones((2, 8, 1, 3)) * 2
    target = np.array([[[3., np.nan, 2.]], [[2., 2., 2.]]])
    scores = per_window(samples, target, np.isfinite(target))
    np.testing.assert_array_equal(scores["n"][:, 0], [2, 3])
    assert scores["mis80"].sum() == 10
    assert scores["mae"].sum() == 1
    assert scores["crps"].sum() == 1


def test_configuration_identity_and_validation(tmp_path):
    cfg = config(tmp_path)
    changed = copy.deepcopy(cfg)
    changed["windows"]["horizon"] = 8
    assert identity(cfg) != identity(changed)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"windows": {"horizonn": 8}}))
    with pytest.raises(ValueError, match="Unknown settings"):
        load_config(path)
    cfg["model"]["noise_dim"] = 1
    with pytest.raises(ValueError, match="noise_dim"):
        validate(cfg)


def test_changed_input_does_not_reuse_cache(tmp_path):
    cfg = config(tmp_path)
    write_input(cfg)
    prepare(cfg)
    with Path(cfg["data"]["path"]).open("a") as f:
        f.write("2021-05-01,10\n")
    with pytest.raises(ValueError, match="CSV changed"):
        load_windows(cfg)


def test_all_methods_train_score_and_report_with_fixture_encoder(tmp_path, monkeypatch):
    cfg = config(tmp_path, channels=3, horizon=7)
    write_input(cfg)
    windows, _ = prepare(cfg)
    encoded = fake_encoding(windows)
    out = run_directory(cfg)
    np.savez_compressed(out / "encoding.npz", **encoded)
    import solar_forecasting.backbone as backbone
    monkeypatch.setattr(backbone, "load_encoding", lambda *a: encoded)
    from solar_forecasting.workflow import run, report
    run(cfg)
    result = report(cfg)
    assert result["folds"] == [2021] and not result["missing_runs"]
    assert set(result["channels"]) == {"flux_0", "flux_1", "flux_2"}
    for row in result["channels"].values():
        assert set(row["methods"]) == set(cfg["model"]["methods"])
        for scores in row["methods"].values():
            assert scores["n_cells"] == 6 * 7
            assert np.isfinite(scores["mis80"])


@pytest.mark.parametrize("channels,horizon", [(1, 1), (3, 7), (2, 12)])
def test_vendored_timesfm_adapter_api_with_tiny_random_weight_model(channels, horizon):
    # Real vendored inference API, tiny random weights: an API test, not a forecast result.
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "timesfm_base/vendor"), str(root / "timesfm_base/src")]
    from timesfm3.torch.model import TimesFM3Torch
    from timesfm3.torch.configs import ResidualBlockConfig, TransformerConfig, StackedTransformersConfig
    from solar_forecasting.backbone import encode_arrays
    tf = TransformerConfig(model_dims=16, hidden_dims=32, num_heads=2, attention_norm="rms",
         feedforward_norm="rms", qk_norm="rms", use_bias=False, use_rope_seq=True, use_rope_var=False,
         ff_activation="relu", deterministic=True)
    body = TimesFM3Torch(input_patch_len=8, output_patch_len=16,
           residual_block_config=ResidualBlockConfig(hidden_dims=16, output_dims=16, use_bias=False, activation="relu"),
           transformer_config=StackedTransformersConfig(num_layers=1, transformer=tf)).eval()
    x = np.random.default_rng(17).normal(size=(5, channels, 23)).astype(np.float32)
    encoded = encode_arrays(body, x, horizon, "cpu", 2)
    assert encoded["features"].shape == (5, channels, 16)
    assert encoded["quantiles"].shape == (5, channels, horizon, 9)
    assert encoded["native_weight"].shape == (horizon, 16)
    assert all(np.isfinite(value).all() for value in encoded.values())
    with pytest.raises(ValueError, match="horizons <= 16"):
        encode_arrays(body, x, 17, "cpu", 2)
