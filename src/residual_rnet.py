"""Trained residual Engression for both gases + small-network r + correction with the same-day observed CH4
(user request 2026-10-01). Two stations throughout; no linkage inside any model.

  CO2 : the two-station residual Engression of row 15 (settings in residual_engression/configs, locked median,
        per-station noise), already trained; its saved test samples are used as they are.
  CH4 : a residual Engression trained here with the same code and settings as row 15 (rq.engression_np Model
        "residual_centered" with StationHead, noise 1280, hidden 64, lr 2e-4, batch 32, 16 training samples, at most
        80 epochs, patience 12, selection by validation Energy Score), on the old split's 14-day-grid windows that
        have CH4 (train 2017-2022, validation 2023), seeds 17, 29, 43. Only difference: the Energy Score is computed on
        CH4 divided by each station's RMS of (y - TimesFM median) over the training windows (CH4 is in ppb, about
        ten times larger numbers than CO2 in ppm; the multigas project normalizes the same way).
  r   : r_variants.fit_mlp (unchanged), fitted on the 2017-2022 daily windows (2023 every 7th day for early
        stopping) from the observed normal scores under the two residual Engressions (seed 17 heads).
  correction: r_variants.gauss_correct (unchanged) on every CO2 cell whose same-day CH4 was observed.
Scored on the 14 test windows like co2_same14 (BRW + MLO, raw), mean over the three seeds.
Check: without the correction the CO2 scores reproduce row 15.

Usage: python src/residual_rnet.py train SEED | same14

In this repository only the parts used by the large-sample scripts run (enc, build, predict, CFG, cells, fit_mlp, gauss_correct);
the 14-window functions (train, heads, same14) need files of the earlier projects that are not included.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

NEW = Path(__file__).resolve().parents[1]
GCOPY = NEW / "multigas"
R15 = NEW / "residual_engression"
sys.path.insert(0, str(GCOPY / "src"))
from mg import multigas as mg                                                      # noqa: E402
from mg.old_split_g import splits, old_indices, evaluate_predictions               # noqa: E402
from mg.r_variants import cells, fit_mlp, gauss_correct                            # noqa: E402
from rq.engression_np import Model, AdamW, clip_grad_norm, energy_score_and_grad, native_q50_weights   # noqa: E402
from rq.station_noise import StationHead                                           # noqa: E402

CFG = json.loads((R15 / "configs/experiment.json").read_text())
SEEDS = (17, 29, 43)
OUT = NEW / "training" / "residual_ch4"
REPORT = NEW / "results" / "reports"
NATIVE = NEW / "residual_engression" / "native_output_head.npz"   # src/extract_native_head.py


def enc(idx, slots):
    d = mg.data()
    return {"features": d["features"][idx][:, slots].astype(np.float64), "scale": d["scale"][idx][:, slots].astype(np.float64),
            "q50": d["quantiles"][idx][:, slots][..., 4].astype(np.float64), "center": np.zeros((len(idx), len(slots), 14))}


def build(seed):
    H = np.load(NATIVE)
    W, b = native_q50_weights(H["output_head_weight"], H["output_head_bias"])
    head = StationHead(W.shape[1], CFG["noise_dim"], CFG["hidden_dim"], np.random.default_rng(seed))
    return Model("residual_centered", head, W, b, CFG["noise_initial_std_normalized"])


def predict(model, e, S, seed, chunk=4):
    """As rq.run.predict of row 15: one noise vector per sample and station."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, len(e["features"]), chunk):
        part = {k: v[i:i + chunk] for k, v in e.items()}
        out.append(model.sample(part, rng.standard_normal((len(part["features"]), S, 2, CFG["noise_dim"]))))
    return np.concatenate(out)


def ch4_constants(tr):
    d = mg.data()
    err = d["y"][tr][:, 2:4] - d["quantiles"][tr][:, 2:4, :, 4]
    m = d["mask"][tr][:, 2:4]
    return np.sqrt(np.array([np.mean(np.square(err[:, k][m[:, k]])) for k in range(2)]))


def train(seed):
    d = mg.data()
    tr, va = old_indices("train"), old_indices("validation")
    c = ch4_constants(tr)
    e_tr, e_va = enc(tr, [2, 3]), enc(va, [2, 3])
    y, mask = d["y"][tr][:, 2:4].astype(np.float64), d["mask"][tr][:, 2:4]
    yv, mv = d["y"][va][:, 2:4].astype(np.float64), d["mask"][va][:, 2:4]
    scale = c[None, None, :, None]
    model = build(seed)
    params = model.head.params()
    opt = AdamW(params, CFG["head_lr"], CFG["weight_decay"])

    def validate():
        x = predict(model, e_va, CFG["validation_samples"], seed + 100000)
        return float(energy_score_and_grad(x / scale, yv / c[None, :, None], mv, .5, need_grad=False)[0])
    best, best_epoch, bad = validate(), 0, 0
    best_params = {k: p.copy() for k, p in params.items()}
    t0 = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(y))
        rng = np.random.default_rng(seed * 10000 + epoch)
        for i in range(0, len(order), CFG["batch_size"]):
            ids = order[i:i + CFG["batch_size"]]
            e = {k: v[ids] for k, v in e_tr.items()}
            x, state = model.sample(e, rng.standard_normal((len(ids), CFG["train_samples"], 2, CFG["noise_dim"])), need_grad=True)
            loss, dxn = energy_score_and_grad(x / scale, y[ids] / c[None, :, None], mask[ids], CFG["pair_weight"])
            grads = model.backward(dxn / scale, e, state)
            clip_grad_norm(grads, CFG["grad_clip"])
            opt.step(params, grads)
        val = validate()
        if val < best - 1e-9:
            best, best_epoch, bad = val, epoch, 0
            best_params = {k: p.copy() for k, p in params.items()}
        else:
            bad += 1
        if bad >= CFG["patience"]:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"seed{seed}_best_head.npz", **best_params)
    (OUT / f"seed{seed}_complete.json").write_text(json.dumps({"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
        "best_validation_es": best, "seconds": time.time() - t0, "train_windows": int(len(tr)), "validation_windows": int(len(va)),
        "ch4_normalization_ppb": c.tolist()}, indent=1) + "\n")
    print(f"CH4 residual seed {seed}: best epoch {best_epoch} of {epoch}, validation ES {best:.4f}", flush=True)


def heads(seed):
    co2, ch4 = build(seed), build(seed)
    co2.head.set_params(np.load(R15 / "04_训练" / f"residual_centered_station_z1280_seed{seed}" / "best_head.npz"))
    ch4.head.set_params(np.load(OUT / f"seed{seed}_best_head.npz"))
    return co2, ch4


def joint(idx, co2, ch4, S, seed):
    """[W, S, 4 slots, 14]: CO2 from the CO2 residual Engression, CH4 from the CH4 one (independent)."""
    return np.concatenate([predict(co2, enc(idx, [0, 1]), S, seed), predict(ch4, enc(idx, [2, 3]), S, seed + 1)], axis=2)


def fit_cells(idx, co2, ch4, seed, block=400):
    parts = [cells(idx[s:s + block], x=joint(idx[s:s + block], co2, ch4, 256, seed + s)) for s in range(0, len(idx), block)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def same14():
    d = mg.data()
    tr, va = splits("daily")[:2]
    te = splits("grid")[3]
    base = np.load(NEW / "方法/基础_20260920/02_数据/test_windows.npz")["origins"]
    keep = base != "2025-12-16"
    assert np.array_equal(base[keep], d["origins"][te])
    co2, ch4 = heads(17)
    rho_fn = fit_mlp(fit_cells(tr, co2, ch4, 700_000), fit_cells(va, co2, ch4, 800_000), 2017)
    keys = ("mae", "rmse", "crps", "energy_score_observed_normalized", "wis9", "coverage80", "mean_width80", "mis80")
    runs = {"residual": [], "residual_rnet": []}
    y, mask = d["y"][te][:, :2], d["mask"][te][:, :2]
    for seed in SEEDS:
        co2, ch4 = heads(seed)
        saved = np.load(R15 / "06_结果" / f"residual_centered_station_z1280_seed{seed}" / "test_predictions.npz")["samples"][keep].astype(np.float64)
        x = np.concatenate([saved, predict(ch4, enc(te, [2, 3]), 512, seed + mg.OFFSETS["test"])], axis=2)
        c = cells(te, x=x)
        rho = rho_fn(c)
        xc = x.copy()
        for s in (0, 1):
            sel = c["s"] == s
            w, t = c["w"][sel], c["t"][sel]
            xc[w, :, s, t] = gauss_correct(x[w, :, s, t], x[w, :, s + 2, t], d["y"][te][w, s + 2, t], rho[sel])   # observed CH4
        for name, xs in (("residual", x), ("residual_rnet", xc)):
            res = evaluate_predictions(y, mask, samples=xs[:, :, :2], station_names=["BRW", "MLO"])
            r = {k: res["overall"][k] for k in keys}
            r.update({f"{st}_{k}": res["by_station"][st][k] for st in ("BRW", "MLO") for k in ("coverage80", "mean_width80", "mis80", "crps", "mae")})
            runs[name].append(r)
    result = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in runs.items()}
    result["rho_test_cells"] = {"BRW": float(rho[c["s"] == 0].mean()), "MLO": float(rho[c["s"] == 1].mean())}
    REPORT.mkdir(exist_ok=True)
    (REPORT / "residual_rnet_same14.json").write_text(json.dumps(result, indent=1) + "\n")
    for k in ("residual", "residual_rnet"):
        r = result[k]
        print(f"{k:14s} " + " ".join(f"{m}={r[m]:.3f}" for m in keys) + f" | BRW mis {r['BRW_mis80']:.3f} | MLO mis {r['MLO_mis80']:.3f}")
    print("mean r on the test cells:", result["rho_test_cells"])


if __name__ == "__main__":
    train(int(sys.argv[2])) if sys.argv[1] == "train" else same14()
