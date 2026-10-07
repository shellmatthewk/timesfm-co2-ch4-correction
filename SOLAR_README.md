# Configurable solar time-series experiment

This extension reuses the structure of the CO₂ experiment for numerical solar forecasting. SunPy retrieves GOES XRS data. Frozen TimesFM predicts future numerical values, and small heads adjust its predictions or uncertainty. Images are a possible later extension; this runner does not use them.

The original CO₂ scripts and results are unchanged. Solar data, caches, trained heads and reports are kept under ignored directories.

Solar dependencies are defined separately from the historical CO₂ version pins. Each run records its actual package versions in `environment.json` and per-run metadata.

## Change the experiment in one file

Edit `configs/solar_goes.json`. All lengths below are **time steps**, interpreted at `data.cadence`.

| Setting | Meaning | Example |
|---|---|---|
| `windows.context` | Historical input length | `28` daily values |
| `windows.horizon` | Number of future values | `7` future days |
| `windows.stride` | Spacing between window starts | `1` for sliding windows |
| `data.cadence` | Observation interval after aggregation | `1D`, `1h`, `15min` |
| `data.start`, `data.end` | Input data period, inclusive dates | `2018-01-01` to `2024-12-31` |
| `data.channels` | Named numerical variables, units, transforms and quality flags | `xrsa`, `xrsb`, or one selected channel |
| `evaluation.train_years` | Maximum number of recent training calendar years | `3`; `null` uses all years from `train_start` |
| `evaluation.max_train_windows` | Training window cap | `1000`; `null` uses all windows |
| `evaluation.max_validation_windows` | Validation window cap | `200` |
| `evaluation.max_test_windows` | Test window cap | `200` for a bounded smoke run |
| `evaluation.test_years` | Rolling evaluation folds | `[2022, 2023, 2024]` |
| `training.epochs`, `patience` | Training and early-stopping limits | `80`, `12` |
| `training.seeds` | Training/sampling seeds | `[17, 29, 43]` |
| `training.*samples` | Samples used during training, selection and evaluation | `16`, `64`, `256`, `512` |
| `model.methods` | Methods to compare | `persistence`, `timesfm`, `basic`, `residual`, `T1`, `N1` |
| `model.device` | TimesFM encoding device | `auto`, `cpu`, `cuda`, `mps` |

Window caps select evenly spaced windows using timestamps, without looking at target values. A different configuration gets a different output directory: `solar_runs/<name>/<configuration-hash>/`. Use the same configuration and overrides at each stage.

The learned heads support one or multiple channels. Their output size follows the horizon. The hidden width grows to at least twice the horizon for the paired-SiLU initialization; `noise_dim` must be at least the horizon. With the locked TimesFM checkpoint, this direct-head adapter supports up to 64 forecast steps. Longer contexts can be used; longer output horizons need a separate adapter.

## Methods and relation to the CO₂ experiment

- `persistence`: repeat the latest historical value; a deterministic reference.
- `timesfm`: samples from the frozen model's nine quantiles, using the existing interpolation and normal-tail code.
- `basic`: shared-noise Engression initialized from TimesFM's native median head; its median can move.
- `residual`: per-channel-noise Engression with the median locked to TimesFM.
- `T1`: learn a multiplier on TimesFM's deviations from its median.
- `N1`: learn a multiplier on the noise entering median-locked Engression.

As in the CO₂ workflow, each test year is retrained separately, the previous year selects epochs, and training ends before the validation year. Seeds are averaged per window before paired month-block bootstrap comparisons.

The structure is similar to the CO₂ experiment, but this is a new protocol: losses are normalized by each channel's training RMS error, all learned heads use the configured validation sample count, and calendar-season inputs are disabled by default. T1/N1 train with fair CRPS plus 0.5 Energy Score and select by physical-space CRPS/MIS80 relative to TimesFM. Basic/residual use normalized Energy Score for training and selection. There is no conformal calibration.

These are **pure forecasts**. The future-measurement CH₄ LSTM correction is not carried into the solar experiment. Adding it would require a separate, explicitly conditional task.

## Input and preprocessing

The default is GOES-16 XRS one-minute measurements aggregated into daily means. `xrsa` and `xrsb` are modelled in log10 space. Scores and intervals are computed after converting predictions back to W m⁻². Quality flags must equal zero. Nonpositive values are missing, not replaced with a small invented flux.

`data.min_bin_coverage` controls how much valid input is needed per aggregate. Bins are labelled by their **end time in UTC**, when the full aggregate is available. For example, the January 1 daily mean has timestamp January 2 at 00:00 UTC. Fold boundaries use these availability timestamps.

Only historical gaps are filled. Boundary and internal gap lengths and minimum observed history fraction are configurable. Future labels remain missing and are masked in every score. All preprocessing statistics, including missing volatility features, are fitted on training windows only.

For a local CSV, copy `configs/solar_csv.example.json`, set its path and channel definitions, and supply:

```csv
timestamp,f107
2020-01-01,72.4
2020-01-02,73.1
```

Optional per-channel `quality_column` and `good_quality` settings can be used. Set `native_cadence` explicitly for sparse or irregular archives. Upsampling observations is rejected. Duplicate timestamps are rejected. Units in CSV configurations are the user's supplied metadata.

## Run on Windows / PowerShell

Create a separate environment to keep the original CO₂ environment reproducible:

```powershell
python -m venv .venv-solar
.\.venv-solar\Scripts\python.exe -X utf8 -m pip install -r requirements-solar.txt
```

The frozen TimesFM weights are still required for every model except persistence. Download the exact checkpoint described in the main README; the solar loader verifies the existing `timesfm_base/model.lock.json`. Solar encoding extracts the needed native median-head rows directly, so it does not require the separate `native_output_head.npz` extraction step.

```powershell
$solarPython = ".\.venv-solar\Scripts\python.exe"
& $solarPython src/run_solar.py inspect --config configs/solar_goes.json
& $solarPython src/run_solar.py fetch --config configs/solar_goes.json
& $solarPython src/run_solar.py prepare --config configs/solar_goes.json
& $solarPython src/run_solar.py encode --config configs/solar_goes.json
& $solarPython src/run_solar.py run --config configs/solar_goes.json
& $solarPython src/run_solar.py report --config configs/solar_goes.json
```

`fetch` downloads the selected period, which can be large. It requests one month at a time, saves file hashes and provenance, and publishes the CSV only after every request succeeds. Check `inspect` before fetching. `run` prepares/encodes when caches are absent; it does not download solar observations or model weights automatically.

For quick size changes, command-line overrides are also supported:

```powershell
python src/run_solar.py inspect --context 28 --horizon 7 --train-years 3 --max-train-windows 1000 --channels xrsb
```

Edit the JSON for sustained work so the same settings apply to all stages. `--fold 2024 --seed 17` restricts which configured runs are executed without changing the experiment identity. Reports compare only folds with every configured method and seed complete, and list missing runs explicitly.

## Check the plumbing without downloads or weights

```powershell
python src/run_solar.py demo --context 28 --horizon 7
python -m pip install pytest
python -m pytest -q tests/test_solar_forecasting.py
```

The demo generates a labelled synthetic numerical signal and evaluates persistence only. It is not a real solar experiment or evidence of TimesFM performance. Automated tests exercise every learned method with a fixture encoder, including changing channels/horizons, median locking, masked losses, original-unit scores, chronological folds, training-only preprocessing and stale-cache rejection. When the checkpoint-loading dependencies are installed, tests also exercise the vendored TimesFM API using a tiny random-weight model. Fixture encodings and random weights are never used by the production runner.

## Limits and sources

This extension provides an experiment framework; it does not establish solar forecasting performance. The GOES preset is an editable starting point. Satellite transitions, calibration versions, the forecast horizon and activity-stratified evaluation should be chosen for the intended scientific question. Daily means measure a different task from predicting flare peaks or occurrence. The existing TimesFM weight restrictions also apply here.

- [SunPy GOES XRS acquisition and quality filtering](https://docs.sunpy.org/en/stable/generated/gallery/time_series/goes_xrs_example.html)
- [SunPy installation and optional dependencies](https://docs.sunpy.org/en/stable/topic_guide/installation.html)
- [SunPy NOAA solar-cycle indices](https://docs.sunpy.org/en/stable/generated/api/sunpy.timeseries.sources.NOAAIndicesTimeSeries.html): these are monthly, not daily F10.7 data.
- [Official daily F10.7 archive](https://spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-5-en.php), for a separately supplied CSV experiment.
- TimesFM, Engression and scoring-method references are retained in the main README and original source comments.
