"""Large-sample test (two stations) of the corrections on two bases (user request 2026-10-01, settings fixed before
any large-sample result of these variants was seen):

  base "res" : residual Engression for CO2 and for CH4 (locked median, per-station noise; settings of row 15),
               trained per test year on the fold's daily windows (training years, validation year every 7th day),
               seeds 17, 29, 43. CO2 in ppm as row 15; CH4 divided by each station's RMS of (y - TimesFM median)
               over the training windows (as residual_rnet.py).
  base "tfm" : TimesFM's own quantiles (two-station encoding), samples drawn cell by cell (as timesfm_rnet.py).
  corrections: none; rnet (small network r, same day; r_variants.fit_mlp + gauss_correct, unchanged);
               bilstm (bidirectional LSTM over the 14 days of CH4 scores; bilstm_correct.py, unchanged).
  r network and LSTM are fitted per fold on the fold's training years (validation year for early stopping) from the
  seed-17 base samples. Folds: test years 2020-2025 (training from 2010) and 2014-2019 (training from 2000).
Scores per CO2 cell (conditional_co2.scores): cells with both gases observed (compared with TimesFM + CH4) and,
for the LSTM, cells whose same-day CH4 is missing.

Usage: python src/rolling_residual.py train co2|ch4 YEAR SEED | fold res|tfm YEAR | report [confirm]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_rnet as rr                                                         # noqa: E402
import bilstm_correct as bc                                                        # noqa: E402
from residual_rnet import mg, enc, build, predict, CFG, cells, fit_mlp, gauss_correct   # noqa: E402
from timesfm_rnet import tf_samples                                                # noqa: E402
from mg.conditional_co2 import scores, SEASON                                      # noqa: E402
from mg.summary3 import Bootstrap                                                  # noqa: E402
from rq.engression_np import AdamW, clip_grad_norm, energy_score_and_grad          # noqa: E402

NEW = rr.NEW
HEADS = NEW / "training" / "rolling_residual"
OUT = NEW / "results" / "rolling_corrections"
SEEDS = (17, 29, 43)
SLOTS = {"co2": [0, 1], "ch4": [2, 3]}
LABELS = ("BRW_CO2", "MLO_CO2")
METRICS = ("coverage80", "width80", "mis80", "crps", "abs_error", "n")


def constants(gas, tr):
    if gas == "co2":
        return np.ones(2)
    d = mg.data()
    err = d["y"][tr][:, 2:4] - d["quantiles"][tr][:, 2:4, :, 4]
    m = d["mask"][tr][:, 2:4]
    return np.sqrt(np.array([np.mean(np.square(err[:, k][m[:, k]])) for k in range(2)]))


def train(gas, year, seed):
    out = HEADS / f"{gas}_fold{year}_seed{seed}"
    if (out / "complete.json").exists():
        print("already complete", flush=True)
        return
    d = mg.data()
    f = mg.fold(year)
    tr, va = f["train"], f["validation"][::mg.CFG["validation_stride_days"]]
    sl = SLOTS[gas]
    c = constants(gas, tr)
    scale = c[None, None, :, None]
    e_tr, e_va = enc(tr, sl), enc(va, sl)
    y, mask = d["y"][tr][:, sl].astype(np.float64), d["mask"][tr][:, sl]
    yv, mv = d["y"][va][:, sl].astype(np.float64), d["mask"][va][:, sl]
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
        print(f"{gas} fold{year} seed{seed} ep {epoch} val {val:.4f} best {best_epoch} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= CFG["patience"]:
            break
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "best_head.npz", **best_params)
    (out / "complete.json").write_text(json.dumps({"gas": gas, "fold": year, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
        "best_validation_es": best, "seconds": time.time() - t0, "train_windows": int(len(tr)), "validation_windows": int(len(va)),
        "normalization": c.tolist()}, indent=1) + "\n")
    print(f"DONE {gas} fold{year} seed{seed}: best epoch {best_epoch} of {epoch}, validation ES {best:.4f}", flush=True)


def res_heads(year, seed):
    co2, ch4 = build(seed), build(seed)
    co2.head.set_params(np.load(HEADS / f"co2_fold{year}_seed{seed}" / "best_head.npz"))
    ch4.head.set_params(np.load(HEADS / f"ch4_fold{year}_seed{seed}" / "best_head.npz"))
    return co2, ch4


def base_samples(base, idx, S, seed, year=None, hseed=17):
    if base == "tfm":
        return tf_samples(idx, S, seed)
    co2, ch4 = res_heads(year, hseed)
    return np.concatenate([predict(co2, enc(idx, [0, 1]), S, seed), predict(ch4, enc(idx, [2, 3]), S, seed + 1)], axis=2)


def blockwise(fn, idx, block=400):
    parts = [fn(idx[s:s + block], s) for s in range(0, len(idx), block)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fold(base, year):
    d = mg.data()
    f = mg.fold(year)
    tr, va, te = f["train"], f["validation"], f["test"]
    def fit_data(idx, seed):
        """Cells for the r network and per-window arrays for the LSTM, from the same seed-17 base samples."""
        def one(i, s):
            x = base_samples(base, i, 256, seed + s, year)
            return {**{"c_" + k: v for k, v in cells(i, x=x).items()}, **{"a_" + k: v for k, v in bc.arrays(i, x).items()}}
        R = blockwise(one, idx)
        return {k[2:]: v for k, v in R.items() if k[0] == "c"}, {k[2:]: v for k, v in R.items() if k[0] == "a"}
    C_tr, A_tr = fit_data(tr, 700_000)
    C_va, A_va = fit_data(va, 800_000)
    rho_fn = fit_mlp(C_tr, C_va, year)
    lstm, info = bc.fit("bilstm", bc.sequences(A_tr), bc.sequences(A_va), seed=year)
    y, mask = d["y"][te], d["mask"][te].astype(bool)
    sums = {}
    for seed in SEEDS:
        x = base_samples(base, te, 512, seed + mg.OFFSETS["test"], year, hseed=seed)
        A = bc.arrays(te, x)
        mu_s, sg_s = bc.same_day_params(A, te, x, rho_fn)
        with torch.no_grad():
            mu, ls = lstm(torch.from_numpy(bc.sequences(A)[0]))
        mu_l, sg_l = mu.numpy().reshape(len(te), 2, 14).astype(np.float64), np.exp(ls.numpy()).reshape(len(te), 2, 14).astype(np.float64)
        for s, label in enumerate(LABELS):
            for cell_kind, sel in (("both", mask[:, s] & mask[:, s + 2]), ("co2only", mask[:, s] & ~mask[:, s + 2])):
                w, t = np.nonzero(sel)
                xa, ya = x[w, :, s, t], y[w, s, t]
                versions = {"none": xa, "rnet": bc.shift_correct(xa, mu_s[w, s, t], sg_s[w, s, t]),
                            "bilstm": bc.shift_correct(xa, mu_l[w, s, t], sg_l[w, s, t])}
                for v, xs in versions.items():
                    for m, val in scores(xs, ya).items():
                        a = np.zeros(len(te))
                        np.add.at(a, w, val)
                        sums.setdefault(f"{label}__{cell_kind}__{v}__{m}", []).append(a)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{base}_fold{year}.npz", origins=d["origins"][te], months=d["months"][te],
                        **{k: np.mean(v, 0) for k, v in sums.items()})
    (OUT / f"{base}_fold{year}_lstm.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"DONE fold {base} {year}: lstm {info}", flush=True)


def report(tag):
    folds = mg.CONFIRM_FOLDS if tag else mg.FOLDS
    d = mg.data()
    data = {}
    for base in ("tfm", "res"):
        parts = [dict(np.load(OUT / f"{base}_fold{y}.npz")) for y in folds]
        data[base] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert np.array_equal(data["tfm"]["origins"], data["res"]["origins"])
    boot = Bootstrap(data["tfm"]["origins"])
    tf = np.load(rr.GCOPY / "06_结果" / "timesfm_covariates" / f"past_future{tag}.npz")
    roll = np.concatenate([mg.fold(y)["test"] for y in folds])
    where = {r: i for i, r in enumerate(tf["rows"])}
    q = tf["quantiles"][[where[r] for r in roll]]
    y, mask = d["y"][roll], d["mask"][roll].astype(bool)
    rows = [("tfm", "none", "TimesFM 不修正"), ("tfm", "rnet", "TimesFM + r 修正（简化方法）"), ("tfm", "bilstm", "TimesFM + 双向 LSTM"),
            ("res", "none", "残差 Engression 不修正"), ("res", "rnet", "残差 Engression + r 修正"), ("res", "bilstm", "残差 Engression + 双向 LSTM")]
    result = {}
    for k, label in enumerate(LABELS):
        both = mask[:, k] & mask[:, k + 2]
        lo, hi = q[:, k, :, 0], q[:, k, :, 8]
        yy = np.nan_to_num(y[:, k])
        tf_mis = np.where(both, (hi - lo) + 10 * np.maximum(lo - yy, 0) + 10 * np.maximum(yy - hi, 0), 0).sum(1)
        tf_cov = np.where(both, (yy >= lo) & (yy <= hi), 0).sum(1)
        tf_n = both.sum(1).astype(float)
        result[label] = {"timesfm_ch4": {"mis80": float(tf_mis.sum() / tf_n.sum()), "coverage80": float(tf_cov.sum() / tf_n.sum())}}
        for base, v, name in rows:
            g = lambda m, kind="both": data[base][f"{label}__{kind}__{v}__{m}"]
            n = g("n").sum()
            r = {m: float(g(m).sum() / n) for m in METRICS[:-1]}
            r["minus_timesfm_ch4"] = boot.diff((g("mis80"), g("n")), (tf_mis, tf_n))
            if v != "none":
                r["minus_same_base_none"] = boot.diff((g("mis80"), g("n")), (data[base][f"{label}__both__none__mis80"], data[base][f"{label}__both__none__n"]))
            if v == "bilstm":
                r["minus_rnet"] = boot.diff((g("mis80"), g("n")), (data[base][f"{label}__both__rnet__mis80"], data[base][f"{label}__both__rnet__n"]))
            n2 = g("n", "co2only").sum()
            r["co2only_cells"] = {"n": int(n2), **{m: float(g(m, "co2only").sum() / n2) for m in METRICS[:-1]}}
            result[label][name] = r
    period = "2014–2019 确认" if tag else "2020–2025 滚动检验"
    L = [f"# 大样本（两站）：两种底子、三种修正（{period}）", "",
         "自动生成（`python src/rolling_residual.py report`）。两种气体当天都有观测的格子，单位 ppm，三个种子平均，不校准。"
         "差值后的方括号是 95% 区间（按月分块的自助法），不含 0 就是显著。设置在看到这些结果之前就定好了。", ""]
    for label in LABELS:
        r = result[label]
        L += [f"## {label}", "", "| 方法 | MIS80 | 减 TimesFM + CH₄ | 覆盖率 | 区间宽度 | CRPS | MAE |", "|---|---:|---:|---:|---:|---:|---:|",
              f"| TimesFM + CH₄ 协变量 | {r['timesfm_ch4']['mis80']:.3f} | — | {100 * r['timesfm_ch4']['coverage80']:.1f}% | — | — | — |"]
        for _, _, name in rows:
            x = r[name]
            dd = x["minus_timesfm_ch4"]
            L.append(f"| {name} | {x['mis80']:.3f} | {dd['difference']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}] | "
                     f"{100 * x['coverage80']:.1f}% | {x['width80']:.3f} | {x['crps']:.3f} | {x['abs_error']:.3f} |")
        for base, name_r, name_l in (("tfm", "TimesFM + r 修正（简化方法）", "TimesFM + 双向 LSTM"), ("res", "残差 Engression + r 修正", "残差 Engression + 双向 LSTM")):
            dd = r[name_l]["minus_rnet"]
            L.append(f"\n双向 LSTM 减只看当天（{'TimesFM' if base == 'tfm' else '残差 Engression'} 底子）：{dd['difference']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]。")
        L += ["", "当天没测到 CH₄ 的格子（只看当天的修正不动它们）：", "", "| 方法 | 格子数 | MIS80 | CRPS | MAE |", "|---|---:|---:|---:|---:|"]
        for _, _, name in rows:
            x = r[name]["co2only_cells"]
            L.append(f"| {name} | {x['n']} | {x['mis80']:.3f} | {x['crps']:.3f} | {x['abs_error']:.3f} |")
        L.append("")
    (rr.REPORT / f"large_two_station{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    (rr.REPORT / f"large_two_station{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "train":
        train(a[1], int(a[2]), int(a[3]))
    elif a[0] == "fold":
        fold(a[1], int(a[2]))
    else:
        report("_confirm" if a[1:] == ["confirm"] else "")
