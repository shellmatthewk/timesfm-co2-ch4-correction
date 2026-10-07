"""Bounded check using real climate data and the locked pretrained model."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys

import numpy as np
import torch

from solar_forecasting.backbone import encode_arrays, load_body
from solar_forecasting.config import ROOT, load_config, write_json
from solar_forecasting.data import fold_indices, load_windows
from solar_forecasting.workflow import evaluate, train_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("oisst", "era5_temperature", "era5_precipitation"),
                        default=["oisst", "era5_temperature", "era5_precipitation"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    body, lock = load_body("cpu")
    environment = {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "torch", "huggingface_hub", "safetensors")}
    environment["python"] = sys.version
    for dataset in args.datasets:
        original = load_config(ROOT / "configs" / f"{dataset}_main.json")
        windows, audit = load_windows(original)
        cfg = copy.deepcopy(original)
        cfg["evaluation"].update(max_train_windows=8, max_validation_windows=4, max_test_windows=4)
        cfg["training"].update(epochs=1, seeds=[17], head_train_samples=8, adjustment_train_samples=8,
                               validation_samples=16, evaluation_samples=16, batch_size=4, sample_chunk=4)
        fold = fold_indices(windows, 2020, cfg)
        if any(not len(idx) for idx in fold.values()):
            raise ValueError("Smoke test needs nonempty 2020 training/validation/test folds")
        selected = np.concatenate(list(fold.values()))
        if len(set(selected)) != len(selected):
            raise ValueError("Smoke test fold partitions overlap")
        small = {k: v[selected] for k, v in windows.items()}
        offset, local = 0, {}
        for part, idx in fold.items():
            local[part] = np.arange(offset, offset + len(idx))
            offset += len(idx)
        encoded = encode_arrays(body, small["x"], cfg["windows"]["horizon"], "cpu", 4)
        record = {"purpose": "Setup validation only; one epoch and small window caps are not research results",
                  "full_study_completed": False, "dataset": dataset, "input_sha256": audit["input_sha256"],
                  "source_provenance": audit.get("source_provenance"), "model_lock": lock,
                  "device": "cpu", "config": cfg, "fold": 2020, "selected_window_indices": selected.tolist(),
                  "environment": environment,
                  "selected_window_origins": small["origin"].astype(str).tolist(), "fold_sizes": {p: len(i) for p, i in local.items()},
                  "encoding_shapes": {k: list(v.shape) for k, v in encoded.items()}, "methods": {}}
        for method in cfg["model"]["methods"]:
            model, context, training = None, None, None
            if method not in ("persistence", "timesfm"):
                model, context, training = train_head(method, encoded, small, local, cfg, 17)
            scores = evaluate(method, model, encoded, small, context, local["test"], cfg, 17, 16)
            if not all(np.isfinite(v).all() for v in scores.values()):
                raise FloatingPointError(f"Nonfinite physical-space scores for {dataset}/{method}")
            record["methods"][method] = {"finite_scores": True, "scored_cells": int(scores["n"].sum()),
                                          "training": training}
        write_json(ROOT / "climate_runs/setup_smoke" / f"{dataset}.json", record)
        print(f"Real-data smoke test passed: {dataset}, six methods, {len(selected)} windows", flush=True)


if __name__ == "__main__":
    main()
