"""Experiment settings and stable identities for caches and results."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("persistence", "timesfm", "basic", "residual", "T1", "N1")
DEFAULTS = {
    "name": "solar", "output_root": "solar_runs",
    "data": {"source": "csv", "path": "data/solar/input.csv", "timestamp_column": "timestamp",
             "start": "2018-01-01", "end": "2024-12-31", "cadence": "1D", "aggregation": "mean",
             "min_bin_coverage": 0.8, "native_cadence": None, "satellite": 16,
             "channels": [{"name": "xrsb", "unit": "W m-2", "transform": "log10"}]},
    "windows": {"context": 14, "horizon": 14, "stride": 1, "max_edge_gap": 7,
                "max_internal_gap": 7, "min_observed_history_fraction": 0.5},
    "evaluation": {"test_years": [2022, 2023, 2024], "train_start": "2018-01-01",
                   "train_years": None, "validation_stride": 7, "max_train_windows": None,
                   "max_validation_windows": None, "max_test_windows": None,
                   "bootstrap_draws": 4000, "bootstrap_seed": 20260929},
    "model": {"methods": list(METHODS), "device": "auto", "encode_batch": 8,
              "noise_dim": 1280, "hidden_dim": 64, "rule_hidden_dim": 16,
              "initial_noise_scale": 0.05, "calendar_features": False},
    "training": {"seeds": [17, 29, 43], "epochs": 80, "patience": 12, "batch_size": 32,
                 "head_train_samples": 16, "adjustment_train_samples": 64,
                 "validation_samples": 256, "evaluation_samples": 512, "sample_chunk": 8,
                 "head_lr": 0.0002, "rule_lr": 0.001, "weight_decay": 0.01,
                 "grad_clip": 1.0, "threads": 2},
}


def _merge(base, update, path=""):
    if not isinstance(update, dict):
        raise ValueError(f"{path or 'configuration'} must be an object")
    unknown = set(update) - set(base)
    if unknown:
        raise ValueError(f"Unknown settings at {path or 'root'}: {sorted(unknown)}")
    for key, value in update.items():
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{path}.{key}".strip("."))
        else:
            base[key] = copy.deepcopy(value)


def positive(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def validate(cfg):
    from datetime import datetime
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cfg["name"]):
        raise ValueError("name must contain only letters, digits, underscores and hyphens")
    d, w, e, m, t = (cfg[k] for k in ("data", "windows", "evaluation", "model", "training"))
    if d["source"] not in ("csv", "goes_xrs"):
        raise ValueError("data.source must be csv or goes_xrs")
    for label, value in (("data.start", d["start"]), ("data.end", d["end"]), ("evaluation.train_start", e["train_start"])):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be YYYY-MM-DD") from exc
    if d["start"] > d["end"]:
        raise ValueError("data.start must be on or before data.end")
    if d["aggregation"] not in ("mean", "median", "max"):
        raise ValueError("data.aggregation must be mean, median or max")
    for key in ("min_bin_coverage",):
        if not 0 < d[key] <= 1:
            raise ValueError(f"data.{key} must be in (0, 1]")
    channels = d["channels"]
    if not isinstance(channels, list) or not channels:
        raise ValueError("data.channels must be a nonempty list")
    names = []
    for channel in channels:
        if not isinstance(channel, dict) or not isinstance(channel.get("name"), str) or not channel["name"]:
            raise ValueError("Every channel needs a nonempty name")
        if set(channel) - {"name", "unit", "transform", "quality_column", "good_quality"}:
            raise ValueError(f"Unknown channel settings: {channel}")
        if channel.get("transform", "identity") not in ("identity", "log10"):
            raise ValueError("Channel transform must be identity or log10")
        names.append(channel["name"])
    if len(set(names)) != len(names):
        raise ValueError("Channel names must be unique")
    for key in ("context", "horizon", "stride"):
        positive(w[key], f"windows.{key}", 2 if key == "context" else 1)
    for key in ("max_edge_gap", "max_internal_gap"):
        positive(w[key], f"windows.{key}", 0)
    if not 0 < w["min_observed_history_fraction"] <= 1:
        raise ValueError("windows.min_observed_history_fraction must be in (0, 1]")
    if not isinstance(e["test_years"], list) or not e["test_years"]:
        raise ValueError("evaluation.test_years must be a nonempty list")
    for year in e["test_years"]:
        positive(year, "test year", 1900)
    if e["test_years"] != sorted(set(e["test_years"])):
        raise ValueError("test_years must be unique and increasing")
    for key in ("train_years", "max_train_windows", "max_validation_windows", "max_test_windows"):
        if e[key] is not None:
            positive(e[key], f"evaluation.{key}")
    for key in ("validation_stride", "bootstrap_draws"):
        positive(e[key], f"evaluation.{key}")
    if not m["methods"] or len(set(m["methods"])) != len(m["methods"]) or set(m["methods"]) - set(METHODS):
        raise ValueError(f"model.methods must be a unique nonempty subset of {METHODS}")
    if m["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("model.device must be auto, cpu, cuda or mps")
    for key in ("encode_batch", "noise_dim", "hidden_dim", "rule_hidden_dim"):
        positive(m[key], f"model.{key}")
    if any(x in m["methods"] for x in ("basic", "residual", "N1")) and m["noise_dim"] < w["horizon"]:
        raise ValueError("model.noise_dim must be >= windows.horizon for the Engression warm start")
    if not isinstance(m["calendar_features"], bool):
        raise ValueError("model.calendar_features must be true or false")
    if not isinstance(t["seeds"], list) or not t["seeds"] or len(set(t["seeds"])) != len(t["seeds"]):
        raise ValueError("training.seeds must be a nonempty list of unique integers")
    for seed in t["seeds"]:
        positive(seed, "seed", 0)
    for key in ("epochs", "patience", "batch_size", "sample_chunk", "threads"):
        positive(t[key], f"training.{key}")
    for key in ("head_train_samples", "adjustment_train_samples", "validation_samples", "evaluation_samples"):
        positive(t[key], f"training.{key}", 2)
    for label, value in (("head_lr", t["head_lr"]), ("rule_lr", t["rule_lr"]), ("grad_clip", t["grad_clip"]),
                         ("initial_noise_scale", m["initial_noise_scale"])):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < float("inf"):
            raise ValueError(f"{label} must be finite and positive")
    if not isinstance(t["weight_decay"], (int, float)) or not 0 <= t["weight_decay"] < float("inf"):
        raise ValueError("weight_decay must be finite and nonnegative")
    return cfg


def load_config(path, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    for section, values in (overrides or {}).items():
        _merge(cfg[section], values, section)
    return validate(cfg)


def identity(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_directory(cfg):
    return resolve(cfg["output_root"]) / cfg["name"] / identity(cfg)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
