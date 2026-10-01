"""TimesFM + small-network r + correction with the same-day observed CH4, without G (user request 2026-10-01).

Two stations throughout (TimesFM reads only BRW and MLO). TimesFM's own quantiles give each gas's predictive
distribution; samples are drawn from them cell by cell (cop.marginal.icdf_from_z, normal tails, as the project's
TimesFM reference "D"). The small network r of r_variants (fit_mlp, unchanged) is trained on the observed normal
scores under these TimesFM distributions in the fold's training years, with the validation year for early
stopping. The CO2 samples of a cell are then corrected with the same-day observed CH4 by the Gaussian formula
(r_variants.gauss_correct, unchanged). This is the r_variants "mlp" variant with TimesFM in place of G.

  same14  : old split (train 2017-2022 daily windows, validation 2023 every 7th day), the 14 test windows, every CO2
            cell (cells without same-day CH4 keep TimesFM's samples), scored like co2_same14 (BRW + MLO, raw)
  rolling : test years 2020-2025 and 2014-2019 (training from 2010 / 2000 to Y-2, validation Y-1), cells with
            both gases observed, against TimesFM + CH4 covariate (two stations), month-block bootstrap
Sampling seeds 17, 29, 43 (scores averaged); the network is fitted once per fold, from samples of seed 17.

Usage: python src/timesfm_rnet.py same14 | run [confirm] | report [confirm]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

NEW = Path(__file__).resolve().parents[1]
GCOPY = NEW / "multigas"
sys.path.insert(0, str(GCOPY / "src"))
from mg import multigas as mg                                               # noqa: E402
from mg.old_split_g import splits, evaluate_predictions                     # noqa: E402
from mg.r_variants import cells, fit_mlp, gauss_correct                     # noqa: E402
from mg.conditional_co2 import scores, SEASON                               # noqa: E402
from mg.summary3 import Bootstrap                                           # noqa: E402
from cop.marginal import icdf_from_z                                        # noqa: E402

OUT = NEW / "results" / "timesfm_rnet"
REPORT = NEW / "results" / "reports"
SEEDS = (17, 29, 43)
LABELS = ("BRW_CO2", "MLO_CO2")
METRICS = ("coverage80", "width80", "mis80", "crps", "abs_error")


def tf_samples(idx, n, seed, block=200):
    """Samples from TimesFM's quantiles (two-station encoding), [W, n, 4 slots, 14]."""
    d = mg.data()
    g = torch.Generator().manual_seed(seed)
    out = []
    for s in range(0, len(idx), block):
        q = torch.as_tensor(d["quantiles"][idx[s:s + block]])
        z = torch.randn(len(q), n, 4, 14, generator=g, dtype=torch.float64)
        out.append(icdf_from_z(q, z).numpy())
    return np.concatenate(out)


def fit_cells(idx, seed, block=400):
    """Observed normal scores under TimesFM's distributions, in blocks (memory)."""
    parts = [cells(idx[s:s + block], x=tf_samples(idx[s:s + block], 256, seed + s)) for s in range(0, len(idx), block)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def network(train, validation, seed):
    return fit_mlp(fit_cells(train, 700_000), fit_cells(validation, 800_000), seed)


def corrected(x, te, rho_fn):
    """Corrected copy of the samples: CO2 cells whose same-day CH4 was observed."""
    d = mg.data()
    c = cells(te, x=x)
    rho = rho_fn(c)
    xc = x.copy()
    for s in (0, 1):
        sel = c["s"] == s
        w, t = c["w"][sel], c["t"][sel]
        xc[w, :, s, t] = gauss_correct(x[w, :, s, t], x[w, :, s + 2, t], d["y"][te][w, s + 2, t], rho[sel])   # observed CH4
    return xc, c, rho


def same14():
    d = mg.data()
    tr, va = splits("daily")[:2]
    te = splits("grid")[3]
    rho_fn = network(tr, va, 2017)
    keys = ("mae", "rmse", "crps", "energy_score_observed_normalized", "wis9", "coverage80", "mean_width80", "mis80")
    runs = {"timesfm": [], "timesfm_rnet": []}
    y, mask = d["y"][te][:, :2], d["mask"][te][:, :2]
    for seed in SEEDS:
        x = tf_samples(te, 512, seed + mg.OFFSETS["test"])
        xc, c, rho = corrected(x, te, rho_fn)
        for name, xs in (("timesfm", x), ("timesfm_rnet", xc)):
            res = evaluate_predictions(y, mask, samples=xs[:, :, :2], station_names=["BRW", "MLO"])
            r = {k: res["overall"][k] for k in keys}
            r.update({f"{s}_{k}": res["by_station"][s][k] for s in ("BRW", "MLO") for k in ("coverage80", "mean_width80", "mis80", "crps", "mae")})
            runs[name].append(r)
    result = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in runs.items()}
    result["rho_test_cells"] = {"BRW": float(rho[c["s"] == 0].mean()), "MLO": float(rho[c["s"] == 1].mean())}
    REPORT.mkdir(exist_ok=True)
    (REPORT / "timesfm_rnet_same14.json").write_text(json.dumps(result, indent=1) + "\n")
    for k in ("timesfm", "timesfm_rnet"):
        r = result[k]
        print(f"{k:13s} " + " ".join(f"{m}={r[m]:.3f}" for m in keys) + f" | BRW mis {r['BRW_mis80']:.3f} | MLO mis {r['MLO_mis80']:.3f}")
    print("mean r on the test cells:", result["rho_test_cells"])


def run(folds, tag):
    d = mg.data()
    OUT.mkdir(parents=True, exist_ok=True)
    for year in folds:
        f = mg.fold(year)
        rho_fn = network(f["train"], f["validation"], year)
        te = f["test"]
        sums = {}
        for seed in SEEDS:
            x = tf_samples(te, 512, seed + mg.OFFSETS["test"])
            xc, c, rho = corrected(x, te, rho_fn)
            for s, label in enumerate(LABELS):
                sel = c["s"] == s
                w, t = c["w"][sel], c["t"][sel]
                ya = d["y"][te][w, s, t]
                for v, xs in (("none", x[w, :, s, t]), ("rnet", xc[w, :, s, t])):
                    for m, val in scores(xs, ya).items():
                        a = np.zeros(len(te))
                        np.add.at(a, w, val)
                        sums.setdefault(f"{label}__{v}__{m}", []).append(a)
                sums.setdefault(f"{label}__rho", []).append(np.bincount(w, weights=rho[sel], minlength=len(te)))
        np.savez_compressed(OUT / f"fold{year}.npz", origins=d["origins"][te], months=d["months"][te],
                            **{k: np.mean(v, 0) for k, v in sums.items()})
        print(f"fold {year} done", flush=True)


def report(folds, tag):
    parts = [dict(np.load(OUT / f"fold{y}.npz")) for y in folds]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    boot = Bootstrap(data["origins"])
    seasons = np.array([SEASON[int(m)] for m in data["months"]])
    d = mg.data()
    tf = np.load(GCOPY / "06_结果" / "timesfm_covariates" / f"past_future{tag}.npz")
    roll = np.concatenate([mg.fold(y)["test"] for y in folds])
    where = {r: i for i, r in enumerate(tf["rows"])}
    q = tf["quantiles"][[where[r] for r in roll]]
    y, mask = d["y"][roll], d["mask"][roll].astype(bool)
    period = "2014–2019 确认" if tag else "2020–2025 滚动检验"
    result, lines = {}, [f"# TimesFM + 小网络 r + 用 CH₄ 实测值修正（不用 G；{period}，两站）", "",
                         "自动生成（`python src/timesfm_rnet.py report`）。两种气体当天都有观测的格子，单位 ppm，三个抽样种子平均，不校准。"
                         "差值是 95% 区间（按月分块的自助法）；区间不含 0 就是显著。", ""]
    for k, label in enumerate(LABELS):
        both = mask[:, k] & mask[:, k + 2]
        lo, hi = q[:, k, :, 0], q[:, k, :, 8]
        yy = np.nan_to_num(y[:, k])
        tf_mis = np.where(both, (hi - lo) + 10 * np.maximum(lo - yy, 0) + 10 * np.maximum(yy - hi, 0), 0).sum(1)
        tf_n = both.sum(1).astype(float)
        assert np.allclose(tf_n, data[f"{label}__none__n"])
        result[label] = {}
        lines += [f"## {label}", "", "| 季节 | 格子数 | TimesFM 不修正 | TimesFM + r + 修正 | TimesFM + CH₄ 协变量 | 修正后减 TimesFM + CH₄ | 平均 r |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for sel_name in ("all", "DJF", "MAM", "JJA", "SON"):
            sel = None if sel_name == "all" else seasons == sel_name
            w = np.ones(len(seasons)) if sel is None else sel.astype(float)
            n = (data[f"{label}__none__n"] * w).sum()
            row = {v: {m: float((data[f"{label}__{v}__{m}"] * w).sum() / n) for m in METRICS} for v in ("none", "rnet")}
            row["timesfm_ch4_mis80"] = float((tf_mis * w).sum() / (tf_n * w).sum())
            row["rnet_minus_timesfm_ch4"] = boot.diff((data[f"{label}__rnet__mis80"], data[f"{label}__rnet__n"]), (tf_mis, tf_n), sel)
            row["rnet_minus_none"] = boot.diff((data[f"{label}__rnet__mis80"], data[f"{label}__rnet__n"]),
                                               (data[f"{label}__none__mis80"], data[f"{label}__none__n"]), sel)
            row["mean_r"] = float((data[f"{label}__rho"] * w).sum() / n)
            row["n_cells"] = int(n)
            result[label][sel_name] = row
            dd = row["rnet_minus_timesfm_ch4"]
            lines.append(f"| {sel_name} | {int(n)} | {row['none']['mis80']:.3f} | {row['rnet']['mis80']:.3f} | {row['timesfm_ch4_mis80']:.3f} | "
                         f"{dd['difference']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}] | {row['mean_r']:+.2f} |")
        a = result[label]["all"]
        lines += ["", f"全年：覆盖率 {100 * a['none']['coverage80']:.1f}% → {100 * a['rnet']['coverage80']:.1f}%，区间宽度 {a['none']['width80']:.3f} → "
                      f"{a['rnet']['width80']:.3f}，CRPS {a['none']['crps']:.3f} → {a['rnet']['crps']:.3f}，MAE {a['none']['abs_error']:.3f} → {a['rnet']['abs_error']:.3f}。", ""]
    REPORT.mkdir(exist_ok=True)
    (REPORT / f"timesfm_rnet{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    (REPORT / f"timesfm_rnet{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    folds, tag = (mg.CONFIRM_FOLDS, "_confirm") if sys.argv[2:] == ["confirm"] else (mg.FOLDS, "")
    {"same14": lambda: same14(), "run": lambda: run(folds, tag), "report": lambda: report(folds, tag)}[sys.argv[1]]()
