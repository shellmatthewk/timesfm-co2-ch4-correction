"""Rolling frozen-TimesFM experiment with fold-specific learned heads."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import sys
import time

import numpy as np
import torch

from .config import file_sha, run_directory, write_json
from .data import channel_names, fold_indices, inverse, load_windows
from .models import (ForecastHead, apply_context, crps_loss, energy_score, fit_context,
                     native_samples, raw_context, training_scale)
from .scoring import METRICS, bootstrap_difference, per_window, pooled


def tensors(encoded, idx):
    return {k: torch.as_tensor(np.ascontiguousarray(encoded[k][idx]), dtype=torch.float32)
            for k in ("features", "center", "scale", "quantiles")}


def prediction_chunks(method, model, encoded, windows, context, idx, cfg, seed, samples):
    chunk = cfg["training"]["sample_chunk"]
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for start in range(0, len(idx), chunk):
            rows = idx[start:start + chunk]
            if method == "persistence":
                last = torch.as_tensor(windows["x"][rows, :, -1], dtype=torch.float32)
                x = last[:, None, :, None].expand(-1, samples, -1, cfg["windows"]["horizon"])
            else:
                e = tensors(encoded, rows)
                if method == "timesfm":
                    x = native_samples(e["quantiles"], samples, rng)
                else:
                    x = model(e, torch.from_numpy(context[rows]), samples, rng)
            yield rows, x.numpy().astype(np.float64)


def evaluate(method, model, encoded, windows, context, idx, cfg, seed, samples):
    parts = []
    for rows, x in prediction_chunks(method, model, encoded, windows, context, idx, cfg, seed, samples):
        x = inverse(x, cfg["data"]["channels"], channel_axis=2)
        parts.append(per_window(x, windows["physical_y"][rows], windows["mask"][rows]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def train_head(method, encoded, windows, fold, cfg, seed):
    t, mc = cfg["training"], cfg["model"]
    tr, va = fold["train"], fold["validation"]
    raw = raw_context(windows, encoded, mc["calendar_features"])
    stats = fit_context(raw, tr)
    context = apply_context(raw, stats)
    scale = torch.from_numpy(training_scale(windows["y"][tr], windows["mask"][tr], encoded["quantiles"][tr, ..., 4]))
    model = ForecastHead(method, encoded["features"].shape[-1], context.shape[-1], cfg["windows"]["horizon"], mc, seed,
                         encoded["native_weight"], encoded["native_bias"])
    head, rule = model.head_parameters(), model.rule_parameters()
    optimizers = []
    if head:
        optimizers.append((torch.optim.AdamW(head, lr=t["head_lr"], weight_decay=t["weight_decay"]), head))
    if rule:
        optimizers.append((torch.optim.AdamW(rule, lr=t["rule_lr"], weight_decay=0.0), rule))
    reference = None
    if method in ("T1", "N1"):
        reference = evaluate("timesfm", None, encoded, windows, None, va, cfg, seed + 100000, t["validation_samples"])

    def validate():
        model.eval()
        if method in ("T1", "N1"):
            scores = evaluate(method, model, encoded, windows, context, va, cfg, seed + 100000, t["validation_samples"])
            ratios = []
            for channel in range(windows["y"].shape[1]):
                if not scores["n"][:, channel].sum():
                    continue
                for metric in ("crps", "mis80"):
                    baseline = reference[metric][:, channel].sum()
                    ratios.append(scores[metric][:, channel].sum() / max(baseline, 1e-12))
            if not ratios:
                raise ValueError("No observed validation targets")
            return float(np.mean(ratios))
        total = 0.0
        for rows, x in prediction_chunks(method, model, encoded, windows, context, va, cfg, seed + 100000, t["validation_samples"]):
            y = torch.from_numpy(np.nan_to_num(windows["y"][rows]).astype(np.float32))
            mask = torch.from_numpy(windows["mask"][rows])
            total += float(energy_score(torch.from_numpy(x).float(), y, mask, scale)) * len(rows)
        return total / len(va)

    best = validate()
    if not np.isfinite(best):
        raise FloatingPointError("Nonfinite initial validation loss")
    best_state, best_epoch, bad, epochs_run = copy.deepcopy(model.state_dict()), 0, 0, 0
    history = [{"epoch": 0, "validation": best}]
    started = time.monotonic()
    count = t["adjustment_train_samples"] if method in ("T1", "N1") else t["head_train_samples"]
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(tr)
        rng = torch.Generator().manual_seed(seed * 10000 + epoch)
        for begin in range(0, len(order), t["batch_size"]):
            rows = order[begin:begin + t["batch_size"]]
            e = tensors(encoded, rows)
            x = model(e, torch.from_numpy(context[rows]), count, rng)
            y = torch.from_numpy(np.nan_to_num(windows["y"][rows]).astype(np.float32))
            mask = torch.from_numpy(windows["mask"][rows])
            loss = energy_score(x, y, mask, scale)
            if method in ("T1", "N1"):
                loss = crps_loss(x, y, mask, scale) + 0.5 * loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            for optimizer, _ in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer, params in optimizers:
                torch.nn.utils.clip_grad_norm_(params, t["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
        value = validate()
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss")
        epochs_run = epoch
        history.append({"epoch": epoch, "validation": value})
        if value < best - 1e-9:
            best, best_state, best_epoch, bad = value, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        print(f"{method} seed={seed} epoch={epoch} validation={value:.6g} best={best_epoch}", flush=True)
        if bad >= t["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    metadata = {"best_epoch": best_epoch, "epochs_run": epochs_run, "validation_score": best,
                "selection": "physical-space CRPS/MIS80 relative to TimesFM" if reference is not None else "normalized model-space Energy Score",
                "seconds": time.monotonic() - started, "loss_scale": scale.tolist(),
                "context_statistics": {k: v.tolist() for k, v in stats.items()},
                "effective_hidden_dim": getattr(model, "hidden_dim", None), "history": history}
    return model, context, metadata


def run(cfg, years=None, seeds=None):
    torch.set_num_threads(cfg["training"]["threads"])
    windows, audit = load_windows(cfg)
    methods = cfg["model"]["methods"]
    encoded = None
    if any(m != "persistence" for m in methods):
        from .backbone import load_encoding
        encoded = load_encoding(cfg, windows, audit)
    years = years or cfg["evaluation"]["test_years"]
    seeds = seeds if seeds is not None else cfg["training"]["seeds"]
    if set(years) - set(cfg["evaluation"]["test_years"]) or set(seeds) - set(cfg["training"]["seeds"]):
        raise ValueError("Selected folds/seeds must be listed in the configuration")
    out = run_directory(cfg)
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "torch", "scipy", "sunpy", "huggingface_hub", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(out / "environment.json", versions)
    for year in years:
        fold = fold_indices(windows, year, cfg)
        if not len(fold["test"]):
            raise ValueError(f"Fold {year} has no test windows")
        if any(m not in ("persistence", "timesfm") for m in methods) and (not len(fold["train"]) or not len(fold["validation"])):
            raise ValueError(f"Fold {year} needs nonempty training and validation periods; inspect the configuration")
        for method in methods:
            for seed in seeds:
                model, context, meta = None, None, {}
                if method not in ("persistence", "timesfm"):
                    model, context, meta = train_head(method, encoded, windows, fold, cfg, seed)
                scores = evaluate(method, model, encoded, windows, context, fold["test"], cfg, seed + 300000, cfg["training"]["evaluation_samples"])
                target = out / f"fold{year}" / method
                target.mkdir(parents=True, exist_ok=True)
                if model is not None:
                    torch.save(model.state_dict(), target / f"seed{seed}_head.pt")
                np.savez_compressed(target / f"seed{seed}_scores.npz", origins=windows["origin"][fold["test"]], **scores)
                write_json(target / f"seed{seed}.json", {"method": method, "fold": year, "seed": seed,
                           "windows_sha256": audit["windows_sha256"],
                           "encoding_sha256": file_sha(out / "encoding.npz") if encoded is not None else None,
                           "environment": versions,
                           "windows": {k: len(v) for k, v in fold.items()}, **meta})
                print(f"DONE {year} {method} seed={seed}: {len(fold['test'])} test windows", flush=True)


def report(cfg):
    out = run_directory(cfg)
    _, audit = load_windows(cfg)
    methods, seeds = cfg["model"]["methods"], cfg["training"]["seeds"]
    complete, missing = [], []
    for year in cfg["evaluation"]["test_years"]:
        absent = [f"{year}/{method}/seed{seed}" for method in methods for seed in seeds
                  if not (out / f"fold{year}" / method / f"seed{seed}_scores.npz").is_file()]
        if absent:
            missing.extend(absent)
        else:
            complete.append(year)
    if not complete:
        raise ValueError("No complete fold: run all configured methods and seeds for at least one test year")
    combined = {}
    for method in methods:
        parts = []
        for year in complete:
            runs = []
            for seed in seeds:
                path = out / f"fold{year}" / method
                meta = json.loads((path / f"seed{seed}.json").read_text(encoding="utf-8"))
                if meta["windows_sha256"] != audit["windows_sha256"]:
                    raise ValueError("Results refer to different input windows; rerun the affected folds")
                if method != "persistence" and meta["encoding_sha256"] != file_sha(out / "encoding.npz"):
                    raise ValueError("Results refer to a different encoding; rerun the affected folds")
                with np.load(path / f"seed{seed}_scores.npz", allow_pickle=False) as saved:
                    runs.append(dict(saved))
            if not all(np.array_equal(runs[0]["origins"], r["origins"]) and np.array_equal(runs[0]["n"], r["n"]) for r in runs):
                raise ValueError("Seeds do not score identical windows/cells")
            parts.append({k: runs[0][k] if k == "origins" else np.mean([r[k] for r in runs], 0) for k in runs[0]})
        combined[method] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    reference = "timesfm" if "timesfm" in methods else methods[0]
    base = combined[reference]
    for scores in combined.values():
        if not np.array_equal(scores["origins"], base["origins"]) or not np.array_equal(scores["n"], base["n"]):
            raise ValueError("Methods do not score identical windows/cells")
    e = cfg["evaluation"]
    result = {"experiment": cfg["name"], "folds": complete, "missing_runs": missing, "reference": reference,
              "notes": "Pure forecasts. Scores in physical units. Seeds averaged before month-block bootstrap; intervals do not include retraining uncertainty.",
              "channels": {}}
    lines = ["# Solar time-series experiment", "", f"Completed test years: {complete}. Reference: {reference}.",
             "Scores use observed targets in physical units. Lower MIS80, CRPS and MAE are better; coverage should approach 80%.",
             "Three or fewer seeds may not capture training variability; the month-block intervals condition on the fitted runs.", ""]
    if cfg["name"] == "synthetic_demo":
        lines[2:2] = ["SYNTHETIC DEMONSTRATION ONLY: persistence baseline; no real solar data or TimesFM results.", ""]
    if missing:
        lines += [f"Incomplete evaluation: {len(missing)} configured runs are missing. Only complete folds are compared.", ""]
    for channel, spec in enumerate(cfg["data"]["channels"]):
        name, unit = spec["name"], spec.get("unit", "unspecified")
        rows = {}
        lines += [f"## {name} ({unit})", "", "| Method | MIS80 | Coverage80 | Width80 | CRPS | MAE | RMSE | MIS80 minus reference [95% CI] |",
                  "|---|---:|---:|---:|---:|---:|---:|---| "]
        for method, scores in combined.items():
            row = pooled(scores, channel)
            row["minus_reference"] = bootstrap_difference(scores["mis80"][:, channel], base["mis80"][:, channel],
                   scores["n"][:, channel], scores["origins"], e["bootstrap_draws"], e["bootstrap_seed"])
            rows[method] = row
            diff = row["minus_reference"]
            ci = diff["ci95"]
            comparison = "—" if method == reference else (f"{diff['difference']:+.6g} [{ci[0]:+.6g}, {ci[1]:+.6g}]" if ci else "insufficient observed month blocks")
            fmt = lambda key: "—" if row[key] is None else f"{row[key]:.6g}"
            cov = "—" if row["coverage80"] is None else f"{100 * row['coverage80']:.1f}%"
            lines.append(f"| {method} | {fmt('mis80')} | {cov} | {fmt('width80')} | {fmt('crps')} | {fmt('mae')} | {fmt('rmse')} | {comparison} |")
        result["channels"][name] = {"unit": unit, "methods": rows}
        lines.append("")
    write_json(out / "report.json", result)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Report: {out / 'report.md'}")
    return result
