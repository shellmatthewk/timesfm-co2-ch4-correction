# Numerical SST and ERA5 forecasting

This extension forecasts daily numerical values with the configurable runner introduced for solar data. It uses the current CO₂ study's dates and 14-day history / 14-day forecast lengths. Images are not required.

## Data and experiment periods

The source configuration is [configs/climate_sources.json](configs/climate_sources.json). Downloads cover **December 1, 1999–December 31, 2025**. The December 1999 buffer supplies history for early 2000 windows.

| Preset suffix | Test years | Training begins | Validation for test year Y | Training ends |
|---|---|---|---|---|
| `main` | 2020–2025 | 2010-01-01 | Y−1 | End of Y−2 |
| `confirm` | 2014–2019 | 2000-01-01 | Y−1 | End of Y−2 |

The setup reads these dates from `multigas/src/mg/multigas.py` and the window lengths from `multigas/src/mg/data.py`. Their SHA-256 hashes are recorded in `data/climate/co2_date_protocol.json`.

| Dataset | Series | Units | Configurations |
|---|---|---|---|
| NOAA daily OISST | Four ocean grid cells: North Atlantic, northeast Pacific, equatorial Pacific, Indian Ocean | °C | `oisst_main.json`, `oisst_confirm.json` |
| ERA5 daily mean 2 m temperature | Four European grid cells near Edinburgh, Madrid, Helsinki, Athens | °C | `era5_temperature_main.json`, `era5_temperature_confirm.json` |
| ERA5 daily precipitation total | The same four European grid cells | mm/day | `era5_precipitation_main.json`, `era5_precipitation_confirm.json` |

Each preset treats its four locations as four target channels. It uses persistence, frozen TimesFM, basic Engression, residual Engression, T1 and N1. The defaults use seeds 17, 29 and 43, with up to 80 epochs and validation-based early stopping. These are new climate experiments; the CO₂ results do not establish that any method improves climate forecasts. The original CO₂ scripts, observations and saved results are preserved.

## Sources and relation to the CCAI project

**SST:** [NOAA OISST v2.1](https://www.ncei.noaa.gov/products/optimum-interpolation-sst), DOI [10.25921/RE9P-PT57](https://doi.org/10.25921/RE9P-PT57), distributed through the [NOAA PSL annual-file catalog](https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.html). The downloader retrieves small subsets through bounded monthly OPeNDAP requests and saves compact time-by-location NetCDF files. It validates the returned dates, selected coordinates, source title, units and ocean-point values. NOAA states that pre-2016 SST values are unchanged between v2 and v2.1. PSL retains the older title on those annual files; the downloader permits that title only before 2016. Later files must identify version 2.1. The 2002 Reynolds paper linked in the conversation describes an earlier weekly, 1° product.

**ERA5:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), explicitly requesting `models=era5`, `cell_selection=nearest`, UTC daily statistics, and `elevation=nan` to disable elevation downscaling. The full date span comes from this one model and distributor. The API's default model mixture is not used. This route works without a Copernicus API key. The raw API responses, exact request parameters and hashes are saved locally.

The two CCAI NetCDF files are copied unchanged into `data/climate/ccai_reference/`; source and copy checksums must match. Their numerical point series are also exported separately. The copied temperature field extends into January 2022, while precipitation stops at December 2021. Neither supplies the complete CO₂ study period. The CCAI task was spatial reconstruction; this extension forecasts future values at fixed locations.

**The extended ERA5 exports are not exact continuations of the CCAI files.** The API and copied fields differ in processing or precision. Over all 4,383 days of 2010–2021 at these four cells, temperature mean absolute differences are about 0.025°C. Precipitation mean absolute differences range from 0.080 to 0.223 mm/day; API minus CCAI mean differences range from −0.077 to −0.220 mm/day. These are source differences, not forecast errors. The main exports use the API across the entire period to avoid a seam caused by stitching different sources. Detailed checks are in `data/climate/era5_ccai_overlap_comparison.json`.

Open-Meteo documents 0.1 mm precision for its hourly precipitation and daily totals computed from 24 hourly values. That precision is relevant to the source comparison; the measured differences above do not identify every processing difference. Newly selected cells outside the copied CCAI grid are marked as unavailable for comparison rather than compared to a distant boundary cell.

## Preprocessing and what forecasts mean

Temperature uses the identity transform. Precipitation uses `log1p`, which preserves dry days as observed zeros, and a physical minimum of zero. Negative input precipitation is missing. After inverse transformation, forecast samples below zero are clipped to zero before scoring. The minimum applies to all six methods. This can put probability mass at zero; it is not a separately fitted precipitation occurrence model.

Input CSV timestamps identify the measurement day. The runner stores each completed daily bin by its end in UTC: January 1's mean or total is available at the January 2 bin end. Climate presets set `evaluation.date_basis=bin_start`, so fold selection and month-block reporting use the measurement day, matching the CO₂ date convention. For example, the first 2020 test window predicts January 1–14 from the historical daily values through December 31, with its latest historical bin completed at January 1 at 00:00 UTC. Complete forecast windows must stay within a partition. The solar default retains `bin_end` selection. Only historical gaps can be filled; future labels remain masked. All preprocessing statistics are fitted on training windows.

Both OISST and ERA5 are retrospective analysis products. This setup preserves temporal ordering but does not reconstruct historical releases or impose publication delays. Results would assess forecasts of retrospective fields; they would not by themselves establish operational forecast performance. OISST's analysis error is not used as forecast uncertainty.

## Run and change amounts

The local environment is `.venv-solar`. For a fresh environment:

```powershell
python -m venv .venv-solar
.\.venv-solar\Scripts\python.exe -X utf8 -m pip install -r requirements-climate.txt
```

The locked TimesFM checkpoint is required for every method except persistence. It follows `timesfm_base/model.lock.json`, including its non-commercial license. The loader verifies file sizes and SHA-256 hashes before inference. See the main README for downloading that exact revision.

```powershell
$climatePython = ".\.venv-solar\Scripts\python.exe"
& $climatePython -u -X utf8 src/setup_climate.py all
& $climatePython src/run_solar.py inspect --config configs/oisst_main.json
& $climatePython src/run_solar.py run --config configs/oisst_main.json
& $climatePython src/run_solar.py report --config configs/oisst_main.json
```

`setup_climate.py all` copies references, downloads all three datasets, creates any missing presets and prepares their windows. Existing experiment JSON edits are preserved; `--reset-presets` explicitly restores defaults from the source configuration. Individual stages are `copy`, `sst`, `era5`, `configure` and `prepare`. This command does not train models. Prepared data and results live under `climate_runs/<name>/<configuration-hash>/`. The data and results are ignored by Git.

Edit the experiment JSON to change history, horizon, stride, target channels, training history, window caps, epochs, seeds or sample counts. The same knobs described in [SOLAR_README.md](SOLAR_README.md) apply. Select one existing location through `data.channels`, or use the CLI:

```powershell
& $climatePython src/run_solar.py inspect --config configs/oisst_main.json --context 28 --horizon 7 --channels north_atlantic --train-years 3 --max-train-windows 1000
```

Use the same overrides for `prepare`, `encode`, `run` and `report`, or save them in a copied JSON. A changed configuration has a separate cache and output directory. This adapter supports forecast horizons up to 64 time steps with the locked checkpoint. Encoding currently processes all prepared windows, even when training and scoring caps are set.

To add or move locations, edit the source configuration and rerun the relevant download stage. Update the target channel names in your experiment JSON, or use `configure --reset-presets` to regenerate the defaults, then prepare. Changed source requests use separate raw caches. Requested and actual grid coordinates are recorded. This starts from grid-cell time series; it does not require copying a full global climate archive.

## Verification

```powershell
& $climatePython -B -X utf8 -m pytest -q tests/test_solar_forecasting.py tests/test_climate_setup.py
& $climatePython -u -B -X utf8 src/smoke_climate.py
```

The automated suite checks climate-specific transforms, zeros and physical bounds, source integrity, incomplete downloads, OISST version handling, chronological folds, and the existing forecasting methods. HTTP responses in unit tests are fixtures. The separate smoke script uses real downloaded data and the checksum-verified pretrained checkpoint: eight training windows, four validation windows and four test windows for test year 2020, one seed and one epoch. Its records are saved in `climate_runs/setup_smoke/`. This checks execution and finite scores; it does not constitute the full study or evidence that a learned method improves forecasting.

The current local PyTorch installation is CPU-only. `model.device=auto` uses the CPU here. The test run also emitted a NumPy/NetCDF binary-size warning; NetCDF read/write checks and downloaded data validation passed. Actual setup package versions are recorded in `data/climate/setup_environment.json`.
