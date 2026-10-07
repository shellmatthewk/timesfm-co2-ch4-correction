"""Download numerical climate series and copy CCAI references; no model training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys

from solar_forecasting.config import ROOT, load_config, write_json


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Set up OISST and ERA5 numerical experiments using the current CO2 dates")
    p.add_argument("stage", choices=("all", "copy", "sst", "era5", "configure", "prepare"))
    p.add_argument("--sources", default=str(ROOT / "configs/climate_sources.json"))
    p.add_argument("--reset-presets", action="store_true", help="Replace existing preset JSON files with defaults from the source configuration")
    args = p.parse_args()
    spec = json.loads(open(args.sources, encoding="utf-8").read())
    from solar_forecasting.climate_data import CLIMATE, co2_protocol, copy_ccai, download_sst, download_era5, compare_ccai, make_configs, validate_sources
    validate_sources(spec)
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "xarray", "netCDF4", "requests")}
    write_json(CLIMATE / "setup_environment.json", {"python": sys.version, **versions})
    print("CO2 date protocol:", co2_protocol(), flush=True)
    if args.stage in ("all", "copy"):
        copy_ccai(spec)
    if args.stage in ("all", "sst"):
        download_sst(spec)
    if args.stage in ("all", "era5"):
        outputs = download_era5(spec)
        if (CLIMATE / "ccai_reference/provenance.json").exists():
            compare_ccai(spec, outputs)
    if args.stage in ("all", "configure", "prepare"):
        paths = make_configs(spec, overwrite=args.reset_presets)
        if args.stage in ("all", "prepare"):
            from solar_forecasting.data import prepare
            audits = {}
            for path in paths:
                _, audit = prepare(load_config(path))
                audits[path.name] = audit
            write_json(CLIMATE / "setup_audit.json", {"sources": spec, "co2_protocol": co2_protocol(), "experiments": audits})
    print(f"Climate setup stage complete: {args.stage}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
