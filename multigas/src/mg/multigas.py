"""Step 3: joint CO2-CH4 trajectories at BRW and MLO (no conformal calibration, no empirical copula).

Slots: 0 BRW-CO2, 1 MLO-CO2, 2 BRW-CH4, 3 MLO-CH4. The head is the unchanged
median-locked residual Engression (rq.engression_np.Model 'residual_centered' with
rq.station_noise.StationHead): sample = TimesFM median + scale * (g - median_S g).

Noise per slot = [own part (640 dims) | second part (640 dims)].
  A, C : the second part is shared by the two gases of a station  -> the gases can move together
  B    : the second part is drawn independently for every slot    -> no cross-gas link
At the warm start the output reads only the first 14 own-noise dimensions, so every
variant starts from independent gases; the link has to be learned.
Inputs per slot:
  A, B : [own TimesFM features | other gas's features at the station | sin, cos(month) |
          14-day history co-movement of the two gases | is_CH4]
  C    : [own TimesFM features | is_CH4]

Usage (PYTHONPATH=src):
  python -m mg.multigas build
  python -m mg.multigas train {A,B,C} YEAR SEED
  python -m mg.multigas reference YEAR          (D: TimesFM, independent cells)
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np
import torch

from .data import ROOT, sha, write_json

sys.path.insert(0, str(ROOT / "src"))
from rq.engression_np import Model, AdamW, clip_grad_norm, energy_score_and_grad            # noqa: E402
from rq.station_noise import StationHead                                                   # noqa: E402
from cop.marginal import icdf_from_z                                                       # noqa: E402
from uq14.metrics import evaluate_predictions                                              # noqa: E402
from .mg_metrics import all_multigas_per_window, cell_per_window, implied_and_observed      # noqa: E402

FOLDS = (2020, 2021, 2022, 2023, 2024, 2025)
TRAIN_START = "2010-01-01"
CONFIRM_FOLDS = (2014, 2015, 2016, 2017, 2018, 2019)   # confirmation (configs: step3_confirmation_2014_2019)
CONFIRM_TRAIN_START = "2000-01-01"
SLOTS = ("BRW_CO2", "MLO_CO2", "BRW_CH4", "MLO_CH4")
PAIR = {0: 2, 1: 3, 2: 0, 3: 1}                       # the other gas at the same station
STATION_OF_SLOT = np.array([0, 1, 0, 1])
CFG = {"own_noise": 640, "second_noise": 640, "hidden_dim": 64, "noise_initial_std_normalized": 0.05,
       "head_lr": 2e-4, "weight_decay": 0.01, "grad_clip": 1.0, "batch_size": 32, "train_samples": 16,
       "validation_samples": 256, "eval_samples": 512, "epochs": 80, "patience": 12, "validation_stride_days": 7}
OFFSETS = {"validation": 100_000, "test": 300_000}
DATA = ROOT / "02_数据/multigas.npz"
_CACHE = {}


def detrend(v):
    t = np.arange(v.shape[-1])
    design = np.vstack([t, np.ones_like(t)]).T
    coef = np.linalg.lstsq(design, v.reshape(-1, v.shape[-1]).T, rcond=None)[0]
    return (v.reshape(-1, v.shape[-1]) - (design @ coef).T).reshape(v.shape)


def build():
    wc, wh = dict(np.load(ROOT / "02_数据/daily_co2_windows.npz")), dict(np.load(ROOT / "02_数据/daily_ch4_windows.npz"))
    ec = torch.load(ROOT / "03_模型/encoded/daily_co2.pt", weights_only=True)
    eh = torch.load(ROOT / "03_模型/encoded/daily_ch4.pt", weights_only=True)
    ic, ih = {o: i for i, o in enumerate(wc["origins"])}, {o: i for i, o in enumerate(wh["origins"])}
    common = [o for o in ic if o in ih]
    a, b = np.array([ic[o] for o in common]), np.array([ih[o] for o in common])
    keep = wc["mask"][a][:, :2].any((1, 2)) & wh["mask"][b].any((1, 2))
    a, b = a[keep], b[keep]

    def slots(co2, ch4):
        return np.concatenate([co2[a][:, :2], ch4[b]], axis=1)
    x = slots(wc["x"], wh["x"]).astype(np.float64)
    dx = detrend(x)
    hist = np.full((len(a), 2), np.nan)
    for s in range(2):
        for i in range(len(a)):
            u, v = dx[i, s], dx[i, s + 2]
            if u.std() > 0 and v.std() > 0:
                hist[i, s] = np.corrcoef(u, v)[0, 1]
    arrays = {"origins": wc["origins"][a], "forecast_dates": wc["forecast_dates"][a],
              "y": slots(wc["y"], wh["y"]), "mask": slots(wc["mask"], wh["mask"]), "x": x.astype(np.float32),
              "imputed_mask": slots(wc["imputed_mask"], wh["imputed_mask"]),
              "features": slots(ec["features"].numpy(), eh["features"].numpy()).astype(np.float32),
              "scale": slots(ec["scale"].numpy(), eh["scale"].numpy()).astype(np.float64),
              "quantiles": slots(ec["native_quantiles"].numpy(), eh["native_quantiles"].numpy()).astype(np.float64),
              "history_comovement": np.nan_to_num(hist, nan=0.0)}
    assert np.array_equal(arrays["forecast_dates"], wh["forecast_dates"][b])
    np.savez(DATA, **arrays)
    audit = {"windows": int(len(a)), "slots": SLOTS, "sha256": sha(DATA),
             "history_comovement_missing": int(np.isnan(hist).sum()),
             "label_fraction_by_slot": arrays["mask"].mean((0, 2)).round(3).tolist()}
    write_json(ROOT / "03_检查/multigas_build.json", audit)
    print(audit)


def data():
    if not _CACHE:
        d = dict(np.load(DATA))
        d["months"] = np.array([int(o[5:7]) for o in d["origins"]])
        d["q50"] = d["quantiles"][..., 4]
        _CACHE.update(d)
    return _CACHE


def fold(year):
    d = data()
    first, last = d["forecast_dates"][:, 0], d["forecast_dates"][:, -1]
    start = CONFIRM_TRAIN_START if year in CONFIRM_FOLDS else TRAIN_START
    spans = {"train": (start, f"{year - 2}-12-31"), "validation": (f"{year - 1}-01-01", f"{year - 1}-12-31"),
             "test": (f"{year}-01-01", f"{year}-12-31")}
    return {p: np.flatnonzero((first >= a) & (last <= b)) for p, (a, b) in spans.items()}


def norm_constants(train_idx):
    """Per slot RMS of (y - TimesFM median) over the training windows (fixed per fold)."""
    d = data()
    e = d["y"][train_idx] - d["q50"][train_idx]
    m = d["mask"][train_idx]
    return np.sqrt(np.array([np.mean(np.square(e[:, k][m[:, k]])) for k in range(4)]))


def inputs(variant, idx):
    d = data()
    own = d["features"][idx].astype(np.float64)
    gas = np.broadcast_to(np.array([0.0, 0.0, 1.0, 1.0])[None, :, None], (len(idx), 4, 1))
    if variant == "C":
        return np.concatenate([own, gas], axis=-1)
    other = own[:, [PAIR[k] for k in range(4)]]
    angle = 2 * np.pi * (d["months"][idx] - 0.5) / 12
    season = np.broadcast_to(np.stack([np.sin(angle), np.cos(angle)], -1)[:, None], (len(idx), 4, 2))
    hist = d["history_comovement"][idx][:, STATION_OF_SLOT][..., None]
    return np.concatenate([own, other, season, hist, gas], axis=-1)


def enc(variant, idx):
    d = data()
    return {"features": inputs(variant, idx), "scale": d["scale"][idx], "q50": d["q50"][idx],
            "center": np.zeros_like(d["q50"][idx])}


def noise(rng, n_windows, n_samples, variant):
    """Every variant consumes the same random stream, so A, B and C see the same own noise
    and, at the warm start, give identical samples. A and C copy the CO2 slot's second
    part to the CH4 slot of the same station; B keeps all four independent."""
    own = rng.standard_normal((n_windows, n_samples, 4, CFG["own_noise"]))
    second = rng.standard_normal((n_windows, n_samples, 4, CFG["second_noise"]))
    if variant != "B":
        second = second[:, :, STATION_OF_SLOT]
    return np.concatenate([own, second], axis=-1)


def build_model(variant, seed, feature_dim):
    head = StationHead(feature_dim, CFG["own_noise"] + CFG["second_noise"], CFG["hidden_dim"], np.random.default_rng(seed))
    return Model("residual_centered", head, np.zeros((14, feature_dim)), np.zeros(14), CFG["noise_initial_std_normalized"])


def predict(model, e, n_samples, seed, variant, chunk=4):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, len(e["features"]), chunk):
        part = {k: v[i:i + chunk] for k, v in e.items()}
        out.append(model.sample(part, noise(rng, len(part["features"]), n_samples, variant)))
    return np.concatenate(out)


def normalized_es(x, y, mask, c, chunk=8):
    """Window-mean normalized ES, computed in chunks of windows (same value, bounded memory)."""
    total = 0.0
    for i in range(0, len(x), chunk):
        loss = energy_score_and_grad(x[i:i + chunk] / c[None, None, :, None], y[i:i + chunk] / c[None, :, None],
                                     mask[i:i + chunk], 0.5, need_grad=False)[0]
        total += loss * len(x[i:i + chunk])
    return total / len(x)


def evaluate(name, year, seed, samples, idx, c, extra):
    d = data()
    y, mask, q50 = d["y"][idx], d["mask"][idx], d["q50"][idx]
    result = {}
    for gas, cols in (("CO2", [0, 1]), ("CH4", [2, 3])):
        result[f"single_cell_{gas}"] = evaluate_predictions(y[:, cols], mask[:, cols], samples=samples[:, :, cols],
                                                            station_names=["BRW", "MLO"], include_energy=False)
        result[f"single_cell_{gas}"]["unit"] = "ppm" if gas == "CO2" else "ppb"
    per_window = all_multigas_per_window(samples, y, mask, q50, c)
    result["joint"] = {k: float(v[0].sum() / v[1].sum()) if v[1].sum() > 0 else None for k, v in per_window.items()}
    cells = cell_per_window(samples, y, mask)
    diag = implied_and_observed(samples, y, mask)
    out = ROOT / "06_结果" / f"multigas_{name}" / f"fold{year}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_window.npz", **{f"{k}__num": v[0] for k, v in per_window.items()},
                        **{f"{k}__den": v[1] for k, v in per_window.items()}, **cells)
    np.savez_compressed(out / "diagnostic.npz", origins=d["origins"][idx], months=d["months"][idx],
                        history_comovement=d["history_comovement"][idx], **diag)
    write_json(out / "metrics.json", {"method": name, "fold": year, "seed": seed, "test_windows": int(len(idx)),
                                      "normalization_constants": dict(zip(SLOTS, c.tolist())), **extra, **result})
    co2, ch4, j = result["single_cell_CO2"]["overall"], result["single_cell_CH4"]["overall"], result["joint"]
    print(f"EVAL multigas_{name} fold{year} seed{seed}: CO2 CRPS {co2['crps']:.4f} cov {co2['coverage80']:.3f} "
          f"MIS {co2['mis80']:.3f} | CH4 CRPS {ch4['crps']:.3f} cov {ch4['coverage80']:.3f} MIS {ch4['mis80']:.2f} | "
          f"ES {j['energy_score']:.4f} diffCRPS {j['gas_difference_crps']:.4f} "
          f"bothBrier {j['both_above_brier']:.4f} VSx {j['variogram_cross_gas']:.4f}", flush=True)


def train(variant, year, seed):
    name = {"A": "A_linked_conditional", "B": "B_unlinked", "C": "C_linked_unconditional"}[variant]
    out = ROOT / "04_训练" / f"multigas_{name}" / f"fold{year}_seed{seed}"
    f = fold(year)
    c = norm_constants(f["train"])
    d = data()
    tr, va_idx = f["train"], f["validation"][::CFG["validation_stride_days"]]
    e_tr, e_va = enc(variant, tr), enc(variant, va_idx)
    y, mask = d["y"][tr], d["mask"][tr]
    model = build_model(variant, seed, e_tr["features"].shape[-1])
    params = model.head.params()
    if (out / "complete.json").exists():
        done = json.loads((out / "complete.json").read_text())
        model.head.set_params(np.load(out / "best_head.npz"))
    else:
        out.mkdir(parents=True, exist_ok=True)
        opt = AdamW(params, CFG["head_lr"], CFG["weight_decay"])
        scale4 = c[None, None, :, None]

        def validate():
            x = predict(model, e_va, CFG["validation_samples"], seed + OFFSETS["validation"], variant)
            return float(normalized_es(x, d["y"][va_idx], d["mask"][va_idx], c))
        best = validate()
        best_epoch, bad, hist = 0, 0, [{"epoch": 0, "validation_es": best}]
        best_params = {k: p.copy() for k, p in params.items()}
        t0 = time.time()
        for epoch in range(1, CFG["epochs"] + 1):
            order = np.random.default_rng(seed + epoch).permutation(len(tr))
            rng = np.random.default_rng(seed * 10000 + epoch)
            total = 0.0
            for i in range(0, len(order), CFG["batch_size"]):
                ids = order[i:i + CFG["batch_size"]]
                part = {k: v[ids] for k, v in e_tr.items()}
                x, state = model.sample(part, noise(rng, len(ids), CFG["train_samples"], variant), need_grad=True)
                loss, dxn = energy_score_and_grad(x / scale4, y[ids] / c[None, :, None], mask[ids], 0.5)
                grads = model.backward(dxn / scale4, part, state)
                clip_grad_norm(grads, CFG["grad_clip"])
                opt.step(params, grads)
                total += loss * len(ids)
            val = validate()
            improved = val < best - 1e-9
            if improved:
                best, best_epoch, bad = val, epoch, 0
                best_params = {k: p.copy() for k, p in params.items()}
            else:
                bad += 1
            hist.append({"epoch": epoch, "train_es": total / len(tr), "validation_es": val})
            print(f"multigas_{name} fold{year} seed{seed} ep {epoch} train {total/len(tr):.4f} val {val:.4f} "
                  f"best {best_epoch} ({time.time()-t0:.0f}s)", flush=True)
            if bad >= CFG["patience"]:
                break
        np.savez_compressed(out / "best_head.npz", **best_params)
        write_json(out / "history.json", hist)
        done = {"variant": variant, "fold": year, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                "best_validation_es": best, "seconds": time.time() - t0, "config": CFG,
                "training_windows": int(len(tr)), "validation_windows": int(len(va_idx)),
                "input_dim": int(e_tr["features"].shape[-1]), "checkpoint_sha256": sha(out / "best_head.npz"),
                "periods_read_for_training": ["train", "validation"]}
        write_json(out / "complete.json", done)
        model.head.set_params(best_params)
    te = f["test"]
    samples = predict(model, enc(variant, te), CFG["eval_samples"], seed + OFFSETS["test"], variant)
    point_gap = float(np.abs(np.median(samples, 1) - d["q50"][te]).max())
    evaluate(name, year, seed, samples, te, c, {"selected_epoch": done["best_epoch"], "training": done,
                                                "point_minus_native_median_max_abs": point_gap})


def reference(year):
    """D: TimesFM quantile function, independent cells (three sampling seeds)."""
    d = data()
    f = fold(year)
    c = norm_constants(f["train"])
    te = f["test"]
    q = torch.as_tensor(d["quantiles"][te])
    for seed in (17, 29, 43):
        g = torch.Generator().manual_seed(seed + OFFSETS["test"])
        z = torch.randn(len(te), CFG["eval_samples"], 4, 14, generator=g, dtype=torch.float64)
        samples = icdf_from_z(q, z).numpy()
        evaluate("D_timesfm_independent", year, seed, samples, te, c, {})


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "build":
        build()
    elif stage == "train":
        train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif stage == "reference":
        reference(int(sys.argv[2]))
    else:
        raise SystemExit("unknown stage")
