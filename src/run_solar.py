"""Command line entry point: inspect, fetch, prepare, encode, run, report, or demo."""
from __future__ import annotations

import argparse
import copy
import json
import sys

from solar_forecasting.config import DEFAULTS, METHODS, ROOT, load_config, run_directory, validate, write_json


def parser():
    p = argparse.ArgumentParser(description="Configurable numerical time-series experiment based on the CO2 workflow")
    p.add_argument("stage", choices=("inspect", "fetch", "prepare", "encode", "run", "report", "demo"))
    p.add_argument("--config", default=str(ROOT / "configs/solar_goes.json"))
    p.add_argument("--context", type=int, help="Number of historical time steps")
    p.add_argument("--horizon", type=int, help="Number of forecast time steps")
    p.add_argument("--stride", type=int)
    p.add_argument("--start", help="First input date, YYYY-MM-DD")
    p.add_argument("--end", help="Last input date, YYYY-MM-DD")
    p.add_argument("--cadence", help="For example 1D, 1h or 15min")
    p.add_argument("--input", help="Input CSV path")
    p.add_argument("--channels", nargs="+", help="Select names from the configuration's channels")
    p.add_argument("--train-years", type=int, help="Limit the training history to this many calendar years")
    p.add_argument("--max-train-windows", type=int)
    p.add_argument("--max-validation-windows", type=int)
    p.add_argument("--max-test-windows", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--methods", nargs="+", choices=METHODS)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--fold", type=int, nargs="+", help="Run a subset of configured test years")
    p.add_argument("--seed", type=int, nargs="+", help="Run a subset of configured seeds")
    return p


def configured(args):
    overrides = {"data": {}, "windows": {}, "evaluation": {}, "model": {}, "training": {}}
    mapping = {"data": {"start": "start", "end": "end", "cadence": "cadence", "input": "path"},
               "windows": {"context": "context", "horizon": "horizon", "stride": "stride"},
               "evaluation": {"train_years": "train_years", "max_train_windows": "max_train_windows",
                              "max_validation_windows": "max_validation_windows", "max_test_windows": "max_test_windows"},
               "model": {"methods": "methods", "device": "device"}, "training": {"epochs": "epochs"}}
    for section, fields in mapping.items():
        for arg, key in fields.items():
            if getattr(args, arg) is not None:
                overrides[section][key] = getattr(args, arg)
    cfg = load_config(args.config, overrides)
    if args.channels:
        names = {c["name"] for c in cfg["data"]["channels"]}
        if len(set(args.channels)) != len(args.channels) or set(args.channels) - names:
            raise ValueError(f"--channels must select unique configured names from {sorted(names)}")
        lookup = {c["name"]: c for c in cfg["data"]["channels"]}
        cfg["data"]["channels"] = [lookup[n] for n in args.channels]
    return validate(cfg)


def demo(args):
    """A synthetic plumbing check; no SunPy downloads or fabricated TimesFM results."""
    import numpy as np
    import pandas as pd
    cfg = copy.deepcopy(DEFAULTS)
    cfg["name"] = "synthetic_demo"
    cfg["data"].update({"source": "csv", "path": "solar_runs/demo_input.csv", "start": "2018-01-01",
                        "end": "2021-12-31", "native_cadence": "1D", "min_bin_coverage": 1.0,
                        "channels": [{"name": "synthetic_flux", "unit": "arbitrary units", "transform": "identity"}]})
    cfg["windows"].update({"context": args.context or 14, "horizon": args.horizon or 14})
    cfg["evaluation"].update({"test_years": [2021], "train_start": "2018-01-01", "bootstrap_draws": 100,
                             "max_test_windows": args.max_test_windows or 30})
    cfg["model"]["methods"] = ["persistence"]
    cfg["training"]["seeds"] = [17]
    cfg = validate(cfg)
    path = ROOT / cfg["data"]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2018-01-01", "2021-12-31", freq="1D")
    k = np.arange(len(dates))
    flux = 100 + 20 * np.sin(2 * np.pi * k / 27) + np.random.default_rng(17).normal(0, 2, len(k))
    pd.DataFrame({"timestamp": dates, "synthetic_flux": flux}).to_csv(path, index=False)
    print("SYNTHETIC DEMONSTRATION ONLY: persistence baseline, no trained TimesFM or real solar results.")
    from solar_forecasting.data import prepare
    from solar_forecasting.workflow import run, report
    prepare(cfg)
    run(cfg)
    report(cfg)
    write_json(run_directory(cfg) / "DEMONSTRATION_ONLY.json", {"synthetic": True, "real_solar_results": False})


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    if args.stage == "demo":
        demo(args)
        return
    cfg = configured(args)
    if args.stage == "inspect":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"Output directory: {run_directory(cfg)}")
        print("Lengths are time steps at the selected cadence. Future measured covariates are not used.")
    elif args.stage == "fetch":
        from solar_forecasting.sunpy_data import fetch_goes
        fetch_goes(cfg)
    elif args.stage == "prepare":
        from solar_forecasting.data import prepare
        prepare(cfg)
    elif args.stage == "encode":
        from solar_forecasting.data import load_windows
        from solar_forecasting.backbone import encode
        windows, audit = load_windows(cfg)
        encode(cfg, windows, audit)
    else:
        from solar_forecasting.workflow import run, report
        if args.stage == "run":
            run(cfg, args.fold, args.seed)
        else:
            report(cfg)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
