"""Large-sample test (two stations) of the basic Engression with 1280-dim noise (ID 5 of the 14-window table; user
request 2026-10-01 "加上基础1280维噪声的engression", without correction). Settings of ID 5 as re-run in the residual
Engression project's NumPy implementation (case control_z1280 of the earlier residual Engression project): head "control",
sample = TimesFM center + scale * g([h, eps]) with g warm-started to TimesFM's native median + 0.05 * eps[:14]; one
1280-dim noise vector per sample shared by the two stations; Energy Score in ppm over 2 stations x 14 days (pair
weight 1/2), 16 training samples, AdamW lr 2e-4 decay 0.01, gradient clip 1, batch 32, at most 80 epochs, patience 12,
selection by validation ES (256 samples, epoch 0 eligible). Settings fixed before any large-sample result of it.
Folds and seeds exactly as rolling_residual.train (training windows up to year-2; validation year-1, every 7th day;
seeds 17, 29, 43) and scored as rolling_residual.fold (512 samples, seed + 300000; cells with both gases observed, and
cells whose same-day CH4 is missing).

Usage: python src/rolling_basic.py run YEAR SEED          (NOISE_SMOKE=1: tiny quick check)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_rnet as rr                                                       # noqa: E402
from residual_rnet import mg, enc, CFG, NATIVE                                   # noqa: E402
from mg.conditional_co2 import scores                                            # noqa: E402
from rq.engression_np import Head, Model, AdamW, clip_grad_norm, energy_score_and_grad, native_q50_weights   # noqa: E402

SMOKE = os.environ.get("NOISE_SMOKE") == "1"
NEW = rr.NEW
HEADS = NEW / "training" / ("rolling_basic_smoke" if SMOKE else "rolling_basic")
OUT = NEW / "results" / ("rolling_basic_smoke" if SMOKE else "rolling_basic")
ENCODED = rr.GCOPY / "03_模型/encoded/daily_co2.pt"            # two-station TimesFM encoding of the daily CO2 windows
WINDOWS = rr.GCOPY / "02_数据/daily_co2_windows.npz"
LABELS = ("BRW_CO2", "MLO_CO2")
_C = {}


def center_all():
    """TimesFM's center (RevIN mean + trend) for the windows of mg.data(); multigas.npz does not keep it."""
    if not _C:
        d = mg.data()
        w = np.load(WINDOWS)
        row = {o: i for i, o in enumerate(w["origins"])}
        a = np.array([row[o] for o in d["origins"]])
        e = torch.load(ENCODED, weights_only=True)
        assert np.array_equal(e["features"].numpy()[a], d["features"][:, :2])
        _C["center"] = e["center"].numpy()[a].astype(np.float64)
    return _C["center"]


def enc_basic(idx):
    e = enc(idx, [0, 1])
    e["center"] = center_all()[idx]
    return e


def build(seed):
    H = np.load(NATIVE)
    W, b = native_q50_weights(H["output_head_weight"], H["output_head_bias"])
    head = Head(W.shape[1], CFG["noise_dim"], CFG["hidden_dim"], np.random.default_rng(seed))
    return Model("control", head, W, b, CFG["noise_initial_std_normalized"])


def predict(model, e, S, seed, chunk=4):
    """One noise vector per sample, shared by the two stations."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, len(e["features"]), chunk):
        part = {k: v[i:i + chunk] for k, v in e.items()}
        out.append(model.sample(part, rng.standard_normal((len(part["features"]), S, CFG["noise_dim"]))))
    return np.concatenate(out)


def run(year, seed):
    d = mg.data()
    f = mg.fold(year)
    tr, va, te = f["train"], f["validation"][::mg.CFG["validation_stride_days"]], f["test"]
    if SMOKE:
        tr = tr[::25]
    e_tr, e_va, e_te = enc_basic(tr), enc_basic(va), enc_basic(te)
    y, mask = d["y"][tr][:, :2].astype(np.float64), d["mask"][tr][:, :2]
    yv, mv = d["y"][va][:, :2].astype(np.float64), d["mask"][va][:, :2]
    model = build(seed)
    params = model.head.params()
    opt = AdamW(params, CFG["head_lr"], CFG["weight_decay"])

    def validate():
        x = predict(model, e_va, CFG["validation_samples"], seed + 100000)
        return float(energy_score_and_grad(x, yv, mv, .5, need_grad=False)[0])
    best, best_epoch, bad = validate(), 0, 0
    best_params = {k: p.copy() for k, p in params.items()}
    t0 = time.time()
    print(f"basic {year} s{seed} ep 0 val {best:.4f} best 0 (0s)", flush=True)
    for epoch in range(1, (1 if SMOKE else CFG["epochs"]) + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(y))
        rng = np.random.default_rng(seed * 10000 + epoch)
        for i in range(0, len(order), CFG["batch_size"]):
            ids = order[i:i + CFG["batch_size"]]
            e = {k: v[ids] for k, v in e_tr.items()}
            x, state = model.sample(e, rng.standard_normal((len(ids), CFG["train_samples"], CFG["noise_dim"])), need_grad=True)
            loss, dx = energy_score_and_grad(x, y[ids], mask[ids], CFG["pair_weight"])
            grads = model.backward(dx, e, state)
            clip_grad_norm(grads, CFG["grad_clip"])
            opt.step(params, grads)
        val = validate()
        if val < best - 1e-9:
            best, best_epoch, bad = val, epoch, 0
            best_params = {k: p.copy() for k, p in params.items()}
        else:
            bad += 1
        print(f"basic {year} s{seed} ep {epoch} val {val:.4f} best {best_epoch} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= CFG["patience"]:
            break
    model.head.set_params(best_params)
    seconds = time.time() - t0
    HEADS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HEADS / f"basic_fold{year}_seed{seed}.npz", **best_params)
    x = predict(model, e_te, CFG["eval_samples"], seed + mg.OFFSETS["test"])
    yt, mt = d["y"][te], d["mask"][te].astype(bool)
    sums = {}
    for s, label in enumerate(LABELS):
        for kind, sel in (("both", mt[:, s] & mt[:, s + 2]), ("co2only", mt[:, s] & ~mt[:, s + 2])):
            w, t = np.nonzero(sel)
            for m, val in scores(x[w, :, s, t], yt[w, s, t]).items():
                a = np.zeros(len(te))
                np.add.at(a, w, val)
                sums[f"{label}__{kind}__none__{m}"] = a
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"basic_fold{year}_seed{seed}.npz", origins=d["origins"][te], months=d["months"][te], **sums)
    (OUT / f"basic_fold{year}_seed{seed}.json").write_text(json.dumps({"fold": year, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
        "best_validation_es": best, "seconds": seconds, "windows": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))}},
        indent=1) + "\n")
    mis = lambda lab: sums[f"{lab}__both__none__mis80"].sum() / sums[f"{lab}__both__none__n"].sum()
    print(f"DONE basic {year} s{seed}: best epoch {best_epoch} of {epoch}; MIS80 (both gases observed) BRW {mis('BRW_CO2'):.3f} "
          f"MLO {mis('MLO_CO2'):.3f}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[2]), int(sys.argv[3]))
