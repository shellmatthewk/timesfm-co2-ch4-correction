"""The round-3 model G (unchanged settings) retrained on the base project's split and scored like the old
12-scheme table, CO2 only, BRW+MLO (configs: co2_only_two_station_table).

  grid : the old 14-day-grid windows that also have CH4 (train 134, validation 8, calibration 12, test 14)
  daily: same years, daily windows (train targets 2017-2022, validation every 7th daily window of 2023),
         same calibration and test windows
Native TimesFM is scored on the same 14 test / 12 calibration windows.
"large": CO2-only raw and cal scores of TimesFM and G on the rolling test windows (2020-2025), calibrated
on the previous year (for G also its epoch-selection year), from the saved round-3 checkpoints.

Usage (PYTHONPATH=src):
  python -m mg.old_split_g train {grid,daily} SEED
  python -m mg.old_split_g large YEAR SEED
  python -m mg.old_split_g report
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np

from . import multigas as mg, linkage as lk
from .data import ROOT, sha, write_json

sys.path.insert(0, str(ROOT / "src"))
from rq.engression_np import AdamW, clip_grad_norm, energy_score_and_grad      # noqa: E402
from uq14.metrics import evaluate_predictions                                   # noqa: E402
from uq14.conformal import fit_cqr, apply_cqr                                   # noqa: E402

OLD = ROOT.parent / "基础_20260920/02_数据"  # two-station rerun 2026-10-01: the two-station windows (same origins)
OUT = ROOT / "06_结果" / "old_split"
SEEDS = (17, 29, 43)


def old_indices(split):
    d = mg.data()
    pos = {o: i for i, o in enumerate(d["origins"])}
    o = np.load(OLD / f"{split}_windows.npz")["origins"]
    return np.array([pos[x] for x in o if x in pos])


def splits(kind):
    d = mg.data()
    first, last = d["forecast_dates"][:, 0], d["forecast_dates"][:, -1]
    cal, test = old_indices("calibration"), old_indices("test")
    if kind == "grid":
        return old_indices("train"), old_indices("validation"), cal, test
    train = np.flatnonzero((first >= "2017-01-01") & (last <= "2022-12-31"))
    val = np.flatnonzero((first >= "2023-01-01") & (last <= "2023-12-31"))[::mg.CFG["validation_stride_days"]]
    return train, val, cal, test


def co2_scores(y, mask, lower_cal, upper_cal, y_cal, mask_cal, samples=None, quantiles=None):
    """Old-table columns for CO2 (BRW, MLO): raw from samples or quantiles, cal by per-station-per-lead CQR."""
    if samples is not None:
        raw = evaluate_predictions(y, mask, samples=samples, station_names=["BRW", "MLO"])["overall"]
        q = np.quantile(samples, [0.1, 0.5, 0.9], axis=1)
        lo, med, hi = q[0], q[1], q[2]
    else:
        raw = evaluate_predictions(y, mask, quantiles=quantiles, station_names=["BRW", "MLO"])["overall"]
        lo, med, hi = quantiles[..., 0], quantiles[..., 4], quantiles[..., 8]
    fit = fit_cqr(lower_cal, upper_cal, y_cal, mask_cal, alpha=0.2)
    adj = apply_cqr(fit, lo, hi, point=med)
    cal = evaluate_predictions(y, mask, point=adj["point"], lower=adj["lower"], upper=adj["upper"],
                               station_names=["BRW", "MLO"])["overall"]
    return {"mae": raw["mae"], "rmse": raw["rmse"], "crps": raw.get("crps"), "es": raw.get("energy_score_observed_normalized"),
            "wis9": raw.get("wis9"), "coverage80_raw": raw["coverage80"], "coverage80_cal": cal["coverage80"],
            "width80_raw": raw["mean_width80"], "width80_cal": cal["mean_width80"], "mis80_raw": raw["mis80"],
            "mis80_cal": cal["mis80"], "cal_n_min": int(fit.cal_n.min()), "cal_n_max": int(fit.cal_n.max())}


def native(test, cal):
    d = mg.data()
    q, qc = d["quantiles"][test][:, :2], d["quantiles"][cal][:, :2]
    return co2_scores(d["y"][test][:, :2], d["mask"][test][:, :2], qc[..., 0], qc[..., 8],
                      d["y"][cal][:, :2], d["mask"][cal][:, :2], quantiles=q)


def train(kind, seed, variant="G"):
    tr, va, cal, te = splits(kind)
    out = ROOT / "04_训练" / "old_split" / f"{variant}_{kind}_seed{seed}"
    d = mg.data()
    c = mg.norm_constants(tr)
    e_tr, e_va = mg.enc("A", tr), mg.enc("A", va)
    zv = lk.Z_OF[variant]
    z_tr, z_va = lk.z_inputs(zv, tr), lk.z_inputs(zv, va)
    y, mask = d["y"][tr], d["mask"][tr]
    model = lk.build(seed, e_tr["features"].shape[-1])
    rhead = lk.RHead(z_tr.shape[-1], lk.R_CFG["hidden"], np.random.default_rng(seed + lk.R_CFG["init_stream_offset"]))
    params, rparams = model.head.params(), rhead.params()
    if (out / "complete.json").exists():
        done = json.loads((out / "complete.json").read_text())
        model.head.set_params(np.load(out / "best_head.npz"))
        rhead.set_params(np.load(out / "best_rhead.npz"))
    else:
        out.mkdir(parents=True, exist_ok=True)
        opt, ropt = AdamW(params, mg.CFG["head_lr"], mg.CFG["weight_decay"]), AdamW(rparams, lk.R_CFG["lr"], lk.R_CFG["weight_decay"])
        scale4 = c[None, None, :, None]

        def validate():
            x, _ = lk.predict(model, rhead, e_va, z_va, mg.CFG["validation_samples"], seed + mg.OFFSETS["validation"])
            return float(lk.combined_loss(x, d["y"][va], d["mask"][va], c))
        best = validate()
        best_epoch, bad, t0 = 0, 0, time.time()
        best_params = {k: p.copy() for k, p in params.items()}
        best_rparams = {k: p.copy() for k, p in rparams.items()}
        for epoch in range(1, mg.CFG["epochs"] + 1):
            order = np.random.default_rng(seed + epoch).permutation(len(tr))
            rng = np.random.default_rng(seed * 10000 + epoch)
            for i in range(0, len(order), mg.CFG["batch_size"]):
                ids = order[i:i + mg.CFG["batch_size"]]
                part = {k: v[ids] for k, v in e_tr.items()}
                r, rcache = rhead.forward(z_tr[ids])
                base = mg.noise(rng, len(ids), mg.CFG["train_samples"], "B")
                x, state = model.sample(part, lk.coupled(base, r), need_grad=True)
                loss, dxn = energy_score_and_grad(x / scale4, y[ids] / c[None, :, None], mask[ids], 0.5)
                loss2, dx2 = lk.pair_es(x / scale4, y[ids] / c[None, :, None], mask[ids])
                grads = model.backward((dxn + lk.LAMBDA * dx2) / scale4, part, state)
                rgrads = rhead.backward(lk.du_from_deps(model.head.d_eps_ch4, base, r), rcache)
                clip_grad_norm({**{f"m.{k}": v for k, v in grads.items()}, **{f"r.{k}": v for k, v in rgrads.items()}}, mg.CFG["grad_clip"])
                opt.step(params, grads)
                ropt.step(rparams, rgrads)
            val = validate()
            if val < best - 1e-9:
                best, best_epoch, bad = val, epoch, 0
                best_params = {k: p.copy() for k, p in params.items()}
                best_rparams = {k: p.copy() for k, p in rparams.items()}
            else:
                bad += 1
            print(f"old_split {variant} {kind} seed{seed} ep {epoch} val {val:.4f} best {best_epoch} ({time.time()-t0:.0f}s)", flush=True)
            if bad >= mg.CFG["patience"]:
                break
        np.savez_compressed(out / "best_head.npz", **best_params)
        np.savez_compressed(out / "best_rhead.npz", **best_rparams)
        done = {"variant": variant, "kind": kind, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch, "best_validation_loss": best,
                "windows": {"train": int(len(tr)), "validation": int(len(va)), "calibration": int(len(cal)), "test": int(len(te))},
                "seconds": time.time() - t0, "checkpoint_sha256": sha(out / "best_head.npz")}
        write_json(out / "complete.json", done)
        model.head.set_params(best_params)
        rhead.set_params(best_rparams)
    xs_te, _ = lk.predict(model, rhead, mg.enc("A", te), lk.z_inputs(zv, te), mg.CFG["eval_samples"], seed + mg.OFFSETS["test"])
    xs_ca, _ = lk.predict(model, rhead, mg.enc("A", cal), lk.z_inputs(zv, cal), mg.CFG["eval_samples"], seed + 200_000)
    qc = np.quantile(xs_ca[:, :, :2], [0.1, 0.9], axis=1)
    res = co2_scores(d["y"][te][:, :2], d["mask"][te][:, :2], qc[0], qc[1], d["y"][cal][:, :2], d["mask"][cal][:, :2],
                     samples=xs_te[:, :, :2])
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"{variant}_{kind}_seed{seed}.json", {"training": done, **res})
    print("EVAL old_split", variant, kind, seed, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)}, flush=True)


def large(year, seed):
    """Rolling test year `year`: CO2-only scores of G (saved round-3 checkpoint) and TimesFM, cal on year-1."""
    f = mg.fold(year)
    te, cal = f["test"], f["validation"]
    d = mg.data()
    ck = ROOT / "04_训练" / "multigas_G_explicit_conditional_pairloss" / f"fold{year}_seed{seed}"
    model = lk.build(seed, mg.enc("A", te[:1])["features"].shape[-1])
    model.head.set_params(np.load(ck / "best_head.npz"))
    rhead = lk.RHead(5, lk.R_CFG["hidden"], np.random.default_rng(0))
    rhead.set_params(np.load(ck / "best_rhead.npz"))
    xs_te, _ = lk.predict(model, rhead, mg.enc("A", te), lk.z_inputs("E", te), mg.CFG["eval_samples"], seed + mg.OFFSETS["test"])
    xs_ca, _ = lk.predict(model, rhead, mg.enc("A", cal), lk.z_inputs("E", cal), mg.CFG["eval_samples"], seed + 200_000)
    qc = np.quantile(xs_ca[:, :, :2], [0.1, 0.9], axis=1)
    g = co2_scores(d["y"][te][:, :2], d["mask"][te][:, :2], qc[0], qc[1], d["y"][cal][:, :2], d["mask"][cal][:, :2], samples=xs_te[:, :, :2])
    g["n_cells"] = int(d["mask"][te][:, :2].sum())
    g["n_windows"] = int(len(te))
    t = native(te, cal)
    t["n_cells"], t["n_windows"] = g["n_cells"], g["n_windows"]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"large_fold{year}_seed{seed}.json", {"G": g, "TimesFM": t})
    print("large", year, seed, "G MIS", round(g["mis80_raw"], 3), "->", round(g["mis80_cal"], 3), "| TimesFM", round(t["mis80_raw"], 3), "->", round(t["mis80_cal"], 3), flush=True)


COLS = [("mae", "MAE", 3), ("rmse", "RMSE", 3), ("crps", "CRPS", 3), ("es", "ES", 3), ("wis9", "WIS9", 3),
        ("coverage80_raw", "Cov80 raw", None), ("coverage80_cal", "Cov80 cal", None), ("width80_raw", "Width raw", 3),
        ("width80_cal", "Width cal", 3), ("mis80_raw", "MIS80 raw", 3), ("mis80_cal", "MIS80 cal", 3)]


def cell(v, nd):
    if v is None:
        return "N/A"
    return f"{100 * v:.1f}%" if nd is None else f"{v:.{nd}f}"


SHOW = [c for c in COLS if not c[0].endswith("_cal")]
SHOW = [(k, h.replace(" raw", ""), nd) for k, h, nd in SHOW]


def report():
    old = json.loads((ROOT / "07_报告/old_table_two_station.json").read_text())["two_station"]
    rows = []
    for label, r in old.items():
        b, a = r["before"], r["after_conformal"]
        rows.append((label, {"mae": b["mae"], "rmse": b["rmse"], "crps": b["crps"], "es": None, "wis9": b["wis9"],
                             "coverage80_raw": b["coverage80"], "coverage80_cal": a["coverage80"], "width80_raw": b["mean_width80"],
                             "width80_cal": a["mean_width80"], "mis80_raw": b["mis80"], "mis80_cal": a["mis80"]}))
    _, _, cal, te = splits("grid")
    t14 = native(te, cal)
    new = {}
    for kind in ("grid", "daily"):
        runs = [json.loads((OUT / f"G_{kind}_seed{s}.json").read_text()) for s in SEEDS]
        new[kind] = {k: float(np.mean([r[k] for r in runs])) for k, _, _ in COLS}
        new[kind]["best_epochs"] = [r["training"]["best_epoch"] for r in runs]
    large_rows = {}
    for m in ("TimesFM", "G"):
        per_seed = []
        for s in SEEDS:
            parts = [json.loads((OUT / f"large_fold{y}_seed{s}.json").read_text())[m] for y in mg.FOLDS]
            n = np.array([p["n_cells"] for p in parts], float)
            w = np.array([p["n_windows"] for p in parts], float)
            agg = {}
            for k, _, _ in COLS:
                if any(p.get(k) is None for p in parts):
                    agg[k] = None
                elif k == "rmse":
                    agg[k] = float(np.sqrt((n * np.array([p[k] for p in parts]) ** 2).sum() / n.sum()))
                elif k == "es":
                    agg[k] = float((w * np.array([p[k] for p in parts])).sum() / w.sum())
                else:
                    agg[k] = float((n * np.array([p[k] for p in parts])).sum() / n.sum())
            per_seed.append(agg)
        large_rows[m] = {k: (None if per_seed[0][k] is None else float(np.mean([p[k] for p in per_seed]))) for k, _, _ in COLS}
    header = "| ID | Method | " + " | ".join(h for _, h, _ in SHOW) + " |"
    sep = "|---|---|" + "---:|" * len(SHOW)
    L = ["# CO₂ 单项指标：两个站（BRW、MLO），原来 12 种方案的表加上新模型", "",
         "自动生成（`python -m mg.old_split_g report`）。只看 CO₂，不看两种气体之间的指标。按要求不放共形校准后的数，全部是模型直接给出的结果。", "",
         "## 1. 原表的时间划分（测试 2025 年）", "",
         "原有方案的行都来自原来每次运行保存的分站结果，把 BRW、MLO 两站合并（按观测格数加权），再对种子取平均，和原表的做法一样。把四个站都合并时，与原表及各项目报告的数字相同（最大差 0.000002）。ES 是整条轨迹的分数，无法从分站结果算出，原来的样本也没有保存，所以标“—”。", "",
         header, sep]
    for i, (label, r) in enumerate(rows, 1):
        L.append(f"| {i} | {label} | " + " | ".join("—" if k == "es" else cell(r[k], nd) for k, _, nd in SHOW) + " |")
    L.append(f"| 1′ | Native TimesFM（14 个测试窗口） | " + " | ".join(cell(t14.get(k), nd) for k, _, nd in SHOW) + " |")
    nid = len(rows) + 1
    L.append(f"| **{nid}** | **新模型 G（原划分，原窗口）** | " + " | ".join(cell(new["grid"][k], nd) for k, _, nd in SHOW) + " |")
    L.append(f"| {nid}b | 新模型 G（原划分的年份，每天一个窗口） | " + " | ".join(cell(new["daily"][k], nd) for k, _, nd in SHOW) + " |")
    L += ["",
          f"- 第 1–{nid - 1} 行用原来的 15 个测试窗口（第 13–15 行是 9/23 和 9/28 的残差 Engression，同样的时间划分，同样从保存的分站结果换算）。新模型需要 CH₄，其中 1 个窗口没有 CH₄ 历史，所以第 1′、{nid}、{nid}b 行用 14 个测试窗口；所有方案在完全相同 14 个窗口上的对比见 co2_same14_table.md。",
          f"- 第 {nid} 行：设置和第三轮完全相同，只把训练数据换成原表的 134 个训练窗口（2017–2022）、8 个验证窗口（2023）。三个种子选中的轮次：{new['grid']['best_epochs']}。",
          f"- 第 {nid}b 行：同样的年份，但每天一个窗口（训练 1855 个），是新模型平时的用法。选中的轮次：{new['daily']['best_epochs']}。",
          "- 只有 14 个测试窗口，数字波动很大，第二步已经证明这种规模的结论不可靠。可靠的比较见下一节。", "",
          "## 2. 大测试集（2020–2025 年，1271 个测试窗口）", "",
          "用第三轮保存的模型，三种子平均。", "",
          "| Method | " + " | ".join(h for _, h, _ in SHOW) + " |", "|---|" + "---:|" * len(SHOW)]
    for m, label in (("TimesFM", "Native TimesFM"), ("G", "新模型 G")):
        L.append(f"| {label} | " + " | ".join(cell(large_rows[m][k], nd) for k, _, nd in SHOW) + " |")
    (ROOT / "07_报告/co2_two_station_table.md").write_text("\n".join(L) + "\n")
    write_json(ROOT / "07_报告/co2_two_station_table.json", {"old_rows": dict(rows), "timesfm_14": t14, "new": new, "large": large_rows})
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "large":
        large(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "report":
        report()
    else:
        raise SystemExit("unknown stage")
