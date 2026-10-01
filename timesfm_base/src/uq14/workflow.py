"""Numbered, independent stages for the fixed 14-to-14 station experiment.

Training reads train/validation labels only. Conformal fitting reads calibration
labels after checkpoint selection. Test labels enter only the evaluation stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "calibration", "test")
STATIONS = ("BRW", "MLO", "SMO", "SPO")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cfg():
    return json.loads((ROOT / "configs/experiment.json").read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def identity():
    paths = [ROOT / "configs/experiment.json", ROOT / "model.lock.json"]
    paths += sorted((ROOT / "data").glob("*.csv"))
    paths += sorted((ROOT / "src/uq14").glob("*.py"))
    paths += sorted((ROOT / "vendor/timesfm3").rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths if p.name != "report.py"}


def load_split(name):
    manifest = json.loads((ROOT / "08_复现检查/experiment_identity.json").read_text())
    path = ROOT / "02_数据" / f"{name}_windows.npz"
    if manifest["identity"] != identity() or manifest["window_sha256"][name] != sha(path):
        raise ValueError("Prepared windows differ from their source/config/code identity. Prepare fresh windows first.")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def prepare():
    from .data import load_windows
    data, audit = load_windows(ROOT / "data/daily_panel.csv", ROOT / "data/daily_quality.csv")
    for split, values in data.items():
        np.savez_compressed(ROOT / "02_数据" / f"{split}_windows.npz", **values)
        print(f"{split}: {len(values['x'])} windows, {int(values['mask'].sum())} observed targets", flush=True)
    write_json(ROOT / "02_数据/data_report.json", audit)
    write_json(ROOT / "08_复现检查/experiment_identity.json", {"identity": identity(),
               "window_sha256": {s: sha(ROOT / "02_数据" / f"{s}_windows.npz") for s in data}})


def backbone(device="cpu"):
    from .backbone import load_backbone
    lock = json.loads((ROOT / "model.lock.json").read_text())
    directory = ROOT / lock["local_directory"]
    for name, detail in lock["files"].items():
        p = directory / name
        if not p.is_file() or p.stat().st_size != detail["bytes"] or sha(p) != detail["sha256"]:
            raise ValueError(f"Locked pretrained file mismatch: {p}")
    return load_backbone(directory, device=device)


def make_model(body, head_type, seed):
    from .model import Station14ForecastModel
    c = cfg()
    seed_all(seed)
    return Station14ForecastModel(body, head_type=head_type, noise_dim=16,
                                  unfreeze_last_n=0, hidden_dim=64,
                                  noise_scale=c["noise_initial_std_normalized"])


@torch.no_grad()
def encode():
    c = cfg()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    body = backbone(device)
    model = make_model(body, "deterministic", 17)
    all_audit = {"device": device, "torch_version": str(torch.__version__),
                 "model_summary": model.trainable_summary(), "splits": {},
                 "identity": identity(), "labels_used_for_encoding": False}
    output = ROOT / "03_模型/encoded"
    output.mkdir(exist_ok=True)
    for split in SPLITS:
        arrays = load_split(split)
        parts, native, parity = [], [], []
        for start in range(0, len(arrays["x"]), c["encode_batch_size"]):
            x = torch.from_numpy(arrays["x"][start:start + c["encode_batch_size"]]).to(device)
            enc = model.encode(x)
            parts.append({k: v.detach().cpu() for k, v in enc.items()})
            native.append(model.native_quantiles(x).detach().cpu())
            direct = model.sample_from_encoded(enc, n_samples=1)[:, 0]
            parity.append(float((direct - model.native_raw_q50(x)).abs().max().cpu()))
        enc = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        if any(not bool(v.isfinite().all()) for v in enc.values()):
            raise ValueError("Nonfinite encoded features.")
        enc["native_quantiles"] = torch.cat(native)
        torch.save(enc, output / f"{split}.pt")
        maxdiff = max(parity)
        if maxdiff > 0.001:
            raise AssertionError(f"Warm initialization diverged from native raw q50: {maxdiff}")
        all_audit["splits"][split] = {"windows": len(arrays["x"]), "raw_q50_max_abs_difference_ppm": maxdiff,
                                    "cache_sha256": sha(output / f"{split}.pt"),
                                    "zero_scale_station_contexts": int((enc["scale"] == 0).sum()),
                                    "shapes": {k: list(v.shape) for k, v in enc.items()}}
        print(f"Encoded {split}: {len(arrays['x'])}; raw-q50 parity {maxdiff:.8g} ppm", flush=True)
    write_json(ROOT / "03_模型/encoding_audit.json", all_audit)


def encoded(split):
    audit = json.loads((ROOT / "03_模型/encoding_audit.json").read_text())
    path = ROOT / "03_模型/encoded" / f"{split}.pt"
    if audit["identity"] != identity() or audit["splits"][split]["cache_sha256"] != sha(path):
        raise ValueError("Data, model, or scoring code changed after encoding; use a fresh run.")
    return torch.load(ROOT / "03_模型/encoded" / f"{split}.pt", map_location="cpu", weights_only=True)


def subset(enc, ids):
    return {k: enc[k][ids] for k in ("features", "center", "scale")}


@torch.no_grad()
def predict(model, enc, samples, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    outputs = []
    for start in range(0, len(enc["features"]), 16):
        out = model.sample_from_encoded(subset(enc, slice(start, start + 16)),
                                        n_samples=samples, generator=generator)
        outputs.append(out.detach().cpu().numpy())
    return np.concatenate(outputs)


def run_dir(head_type, seed):
    return ROOT / "05_训练" / f"{head_type}_seed{seed}"


def train(head_type):
    from .losses import masked_mse, masked_energy_score
    c = cfg()
    torch.set_num_threads(c["threads"])
    data, val = load_split("train"), load_split("validation")
    enc, venc = encoded("train"), encoded("validation")
    body = backbone("cpu")
    for seed in c["seeds"]:
        out = run_dir(head_type, seed)
        if (out / "complete.json").exists():
            if json.loads((out / "complete.json").read_text())["identity"] != identity():
                raise ValueError("Existing run identity mismatch.")
            print(f"Already complete: {out.name}", flush=True)
            continue
        if (out / "history.json").exists():
            raise FileExistsError(f"Incomplete run found; retain it and choose a fresh directory: {out}")
        out.mkdir(exist_ok=True)
        model = make_model(body, head_type, seed)
        model.head.to("cpu")
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=c["head_lr"], weight_decay=c["weight_decay"])
        target = torch.from_numpy(data["y"].astype(np.float32))
        mask = torch.from_numpy(data["mask"])
        vy, vm = torch.from_numpy(val["y"].astype(np.float32)), torch.from_numpy(val["mask"])
        loss_fn = masked_mse if head_type == "deterministic" else masked_energy_score
        def validate():
            pred = torch.from_numpy(predict(model, venc, 1 if head_type == "deterministic" else c["validation_samples"], seed + 100000))
            return float(loss_fn(pred, vy, vm))
        best, epoch_best, bad = validate(), 0, 0
        initial = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
        best_state = {k: v.clone() for k, v in initial.items()}
        history = [{"epoch": 0, "train_loss": None, "validation_loss": best, "selected": True}]
        started = time.perf_counter()
        for epoch in range(1, c["epochs"] + 1):
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(target))
            generator = torch.Generator(device="cpu").manual_seed(seed * 10000 + epoch)
            total = 0.0
            for start in range(0, len(order), c["batch_size"]):
                ids = order[start:start + c["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                pred = model.sample_from_encoded(subset(enc, ids),
                    n_samples=1 if head_type == "deterministic" else c["train_samples"], generator=generator)
                loss = loss_fn(pred, target[ids], mask[ids])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), c["grad_clip"], error_if_nonfinite=True)
                if epoch == 1 and start == 0:
                    write_json(out / "gradient_audit.json", {"head_gradients": [n for n,p in model.head.named_parameters() if p.grad is not None],
                        "backbone_has_gradients": any(p.grad is not None for p in body.parameters()),
                        "backbone_trainable_parameters": sum(p.numel() for p in body.parameters() if p.requires_grad),
                        "train_input_shape": list(data["x"][ids].shape), "train_target_shape": list(target[ids].shape)})
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            model.eval()
            score = validate()
            improved = score < best - 1e-9
            if improved:
                best, epoch_best, bad = score, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
            else:
                bad += 1
            history.append({"epoch": epoch, "train_loss": total / len(target), "validation_loss": score, "selected": improved})
            if epoch == 1 or epoch % 10 == 0:
                print(f"{head_type} seed={seed} epoch={epoch} train={total/len(target):.5f} validation={score:.5f} best={epoch_best}", flush=True)
            if bad >= c["patience"]:
                break
        model.head.load_state_dict(best_state)
        checkpoint = {"head_state": best_state, "head_type": head_type, "seed": seed,
                      "best_epoch": epoch_best, "identity": identity(), "config": c}
        torch.save(checkpoint, out / "best_head.pt")
        # Independent adapter reload must reproduce the fixed-noise predictions.
        before = predict(model, venc, 4 if head_type == "engression" else 1, seed + 700000)
        reloaded = make_model(body, head_type, seed + 1000)
        saved = torch.load(out / "best_head.pt", map_location="cpu", weights_only=True)
        reloaded.head.load_state_dict(saved["head_state"], strict=True)
        after = predict(reloaded, venc, 4 if head_type == "engression" else 1, seed + 700000)
        if not np.array_equal(before, after):
            raise AssertionError("Reloaded head predictions differ.")
        write_json(out / "history.json", history)
        write_json(out / "complete.json", {"identity": identity(), "best_epoch": epoch_best,
                   "best_validation_loss": best, "selection_loss": c["mse_selection" if head_type == "deterministic" else "engression_selection"],
                   "epochs_run": epoch, "seconds": time.perf_counter() - started,
                   "selected_parameter_max_change": max(float((best_state[k] - initial[k]).abs().max()) for k in initial),
                   "reload_prediction_max_difference": 0.0, "parameter_summary": model.trainable_summary(),
                   "checkpoint_sha256": sha(out / "best_head.pt"),
                   "test_labels_used": False, "calibration_labels_used": False})
        print(f"Finished {out.name}: selected epoch {epoch_best}", flush=True)


def fitted_models(body):
    for head_type in cfg()["head_types"]:
        for seed in cfg()["seeds"]:
            out = run_dir(head_type, seed)
            if not (out / "complete.json").exists():
                raise FileNotFoundError(f"Train all predetermined cases first: {out.name}")
            completed = json.loads((out / "complete.json").read_text())
            if completed["checkpoint_sha256"] != sha(out / "best_head.pt"):
                raise ValueError("Selected checkpoint was modified after training.")
            saved = torch.load(out / "best_head.pt", map_location="cpu", weights_only=True)
            if saved["identity"] != identity():
                raise ValueError("Checkpoint identity mismatch.")
            model = make_model(body, head_type, seed)
            model.head.load_state_dict(saved["head_state"], strict=True)
            model.eval()
            yield out.name, model, seed, saved["best_epoch"]


def prediction_set(split):
    c = cfg()
    enc = encoded(split)
    native = enc["native_quantiles"].numpy().astype(np.float64)
    yield "native_timesfm", {"quantiles": native, "point": native[..., 4]}, None
    body = backbone("cpu")
    for name, model, seed, epoch in fitted_models(body):
        samples = predict(model, enc, 1 if model.head_type == "deterministic" else c["eval_samples"],
                          seed + (200000 if split == "calibration" else 300000))
        if model.head_type == "deterministic":
            yield name, {"point": samples[:, 0].astype(np.float64)}, epoch
        else:
            quantiles = np.quantile(samples.astype(np.float64), np.arange(1, 10) / 10, axis=1).transpose(1, 2, 3, 0)
            yield name, {"samples": samples, "quantiles": quantiles, "point": quantiles[..., 4]}, epoch


def forecast_weight_identity(name):
    if name == "native_timesfm":
        return json.loads((ROOT / "model.lock.json").read_text())["files"]
    return {"best_head_sha256": sha(ROOT / "05_训练" / name / "best_head.pt")}


def calibrate():
    from .conformal import fit_cqr, fit_absolute_residual
    data = load_split("calibration")
    for name, prediction, epoch in prediction_set("calibration"):
        out = ROOT / "06_共形校准" / name
        out.mkdir(exist_ok=True)
        if "quantiles" in prediction:
            cal = fit_cqr(prediction["quantiles"][..., 0], prediction["quantiles"][..., -1],
                          data["y"], data["mask"], alpha=cfg()["conformal_alpha"])
            kind = "expansion_only_CQR"
        else:
            cal = fit_absolute_residual(prediction["point"], data["y"], data["mask"], alpha=cfg()["conformal_alpha"])
            kind = "absolute_residual_split_conformal"
        np.savez_compressed(out / "calibration_predictions.npz", **prediction)
        np.savez_compressed(out / "correction.npz", q=cal.q, raw_q=cal.raw_q, cal_n=cal.cal_n, rank=cal.rank)
        write_json(out / "calibration.json", {"method": kind, "model": name, "alpha": .2,
                   "calibration": cal.to_dict(), "selected_epoch": epoch, "test_labels_used": False,
                   "calibration_prediction_sha256": sha(out / "calibration_predictions.npz"),
                   "correction_sha256": sha(out / "correction.npz"),
                   "forecast_weights": forecast_weight_identity(name), "identity": identity()})
        print(f"Calibrated {name}: n={cal.cal_n.min()}..{cal.cal_n.max()}, finite q={np.isfinite(cal.q).sum()}/56", flush=True)


def evaluate():
    from .metrics import evaluate_predictions
    data = load_split("test")
    rows = []
    csv_rows = []
    for name, prediction, epoch in prediction_set("test"):
        directory = ROOT / "06_共形校准" / name
        metadata = json.loads((directory / "calibration.json").read_text())
        if (metadata["identity"] != identity() or metadata["forecast_weights"] != forecast_weight_identity(name)
                or metadata["correction_sha256"] != sha(directory / "correction.npz")):
            raise ValueError("Conformal identity mismatch.")
        with np.load(directory / "correction.npz") as correction:
            q, cal_n = correction["q"], correction["cal_n"]
        out = ROOT / "07_评估结果" / name
        out.mkdir(exist_ok=True)
        kwargs = {"point": prediction["point"], "station_names": STATIONS}
        if "samples" in prediction:
            kwargs.pop("point")
            kwargs["samples"] = prediction["samples"]
        elif "quantiles" in prediction:
            kwargs["quantiles"] = prediction["quantiles"]
        before = evaluate_predictions(data["y"], data["mask"], **kwargs)
        if "quantiles" in prediction:
            lo, hi = prediction["quantiles"][..., 0], prediction["quantiles"][..., -1]
        else:
            lo = hi = prediction["point"]
        lower, upper = lo - q[None], hi + q[None]
        after = evaluate_predictions(data["y"], data["mask"], point=prediction["point"],
                                     lower=lower, upper=upper, station_names=STATIONS)
        # This postprocessor changes endpoints only, never the median/point.
        for key in ("mae", "rmse", "bias"):
            if before["overall"].get(key) != after["overall"].get(key):
                raise AssertionError(f"Conformal unexpectedly changed point metric {key}.")
        item = {"model": name, "selected_epoch": epoch, "before": before, "after_conformal": after,
                "cal_n_min": int(cal_n.min()), "cal_n_max": int(cal_n.max()),
                "q_min_ppm": float(q.min()), "q_max_ppm": float(q.max()),
                "conformal_changes_point_predictions": False,
                "post_conformal_crps_wis": "not_defined: only 80% interval endpoints calibrated; no new full distribution"}
        rows.append(item)
        np.savez_compressed(out / "test_predictions.npz", **prediction, lower80_conformal=lower,
                            upper80_conformal=upper, target=data["y"], observed_mask=data["mask"],
                            origins=data["origins"], forecast_dates=data["forecast_dates"])
        write_json(out / "metrics.json", item)
        for b in range(len(data["y"])):
            for station in range(4):
                for lead in range(14):
                    csv_rows.append({"model": name, "origin": str(data["origins"][b]),
                        "forecast_date": str(data["forecast_dates"][b, lead]), "station": STATIONS[station], "lead": lead+1,
                        "observed": bool(data["mask"][b,station,lead]),
                        "target_ppm": float(data["y"][b,station,lead]) if data["mask"][b,station,lead] else "",
                        "point_ppm": float(prediction["point"][b,station,lead]),
                        "lower80_raw": float(lo[b,station,lead]) if "quantiles" in prediction else "",
                        "upper80_raw": float(hi[b,station,lead]) if "quantiles" in prediction else "",
                        "lower80_conformal": float(lower[b,station,lead]), "upper80_conformal": float(upper[b,station,lead])})
        print(f"Evaluated {name}: MAE={before['overall']['mae']:.5f}, raw_cov={before['overall'].get('coverage80')}, calibrated_cov={after['overall'].get('coverage80')}", flush=True)
    write_json(ROOT / "07_评估结果/all_results.json", {"protocol": cfg(), "models": rows,
                "test_windows": len(data["y"]), "observed_test_cells": int(data["mask"].sum()),
                "complete_test_windows": int(data["mask"].all(axis=(1,2)).sum()),
                "identity": identity(), "temporal_exchangeability_guarantee_claimed": False})
    with (ROOT / "07_评估结果/predictions_all_methods.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)


def main(stage=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "encode", "train_mse", "train_engression", "conformal", "evaluate", "report"], default=stage)
    args = parser.parse_args()
    torch.set_num_threads(cfg()["threads"])
    if args.stage == "prepare": prepare()
    elif args.stage == "encode": encode()
    elif args.stage == "train_mse": train("deterministic")
    elif args.stage == "train_engression": train("engression")
    elif args.stage == "conformal": calibrate()
    elif args.stage == "evaluate": evaluate()
    elif args.stage == "report":
        from .report import report
        report()
    else: parser.error("Choose a stage.")


if __name__ == "__main__":
    main()
