"""CO2 (and CH4) given the same-day OBSERVED value of the other gas, from the joint samples of the
round-3 models G and H (configs: co2_given_observed_ch4).

Scenario: CH4 has been measured at the station on that day, CO2 is not yet known (gap filling, CO2 data
arriving later). This is conditional estimation, not a better forecast: when forecasting, future CH4 is
unknown too.

Conditioning is a regression adjustment inside the model's own 512 samples, per window, station and day,
done in rank (normal-score) space:
    u = normal scores of the ranks of the CO2 samples, v = the same for the CH4 samples,
    v_obs = normal score of the observed CH4 within the CH4 samples (bounded: at most +-2.9 for 512 samples),
    rho = corr(u, v);  u_cond = u - rho * (v - v_obs);  conditional CO2 = CO2 sample quantile at Phi(u_cond).
The first (pre-registered) version did the same adjustment on the raw values; it blew up in windows where the
model's CH4 spread is almost zero (BRW, gap-filled CH4 histories), so it was replaced after the first summary.
Nothing is estimated from observed data. The unconditional samples are regenerated with the evaluation
seeds and checked against the saved per-window scores.

Usage (PYTHONPATH=src):
  python -m mg.conditional_co2 run {G,H} YEAR SEED
  python -m mg.conditional_co2 summary
"""
from __future__ import annotations

import sys
import numpy as np

from . import multigas as mg, linkage as lk
from .data import ROOT, write_json

sys.path.insert(0, str(ROOT / "src"))
from cop.joint_metrics import crps_units, interval_units      # noqa: E402
from scipy.stats import norm, rankdata                          # noqa: E402

OLD = ROOT / "旧版_G自带r"       # results with G's own r, kept for comparison; the final method uses the network r (g_rnet.py)
OUT = OLD / "06_结果" / "co2_given_ch4"
TARGETS = (("BRW_CO2", 0, 2), ("MLO_CO2", 1, 3), ("BRW_CH4", 2, 0), ("MLO_CH4", 3, 1))   # (label, target slot, given slot)
METRICS = ("coverage80", "width80", "mis80", "crps", "abs_error", "n")
PERIODS = {"main": mg.FOLDS, "confirm": mg.CONFIRM_FOLDS}
SEEDS = (17, 29, 43)
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
          9: "SON", 10: "SON", 11: "SON"}


def samples(variant, year, seed):
    name = lk.NAMES[variant]
    ck = ROOT / "04_训练" / f"multigas_{name}" / f"fold{year}_seed{seed}"
    te = mg.fold(year)["test"]
    e = mg.enc("A", te)
    z = lk.z_inputs(lk.Z_OF[variant], te)
    model = lk.build(seed, e["features"].shape[-1])
    model.head.set_params(np.load(ck / "best_head.npz"))
    rhead = lk.RHead(z.shape[-1], lk.R_CFG["hidden"], np.random.default_rng(0))
    rhead.set_params(np.load(ck / "best_rhead.npz"))
    x, _ = lk.predict(model, rhead, e, z, mg.CFG["eval_samples"], seed + mg.OFFSETS["test"])
    return te, x


def condition_linear(xa, xb, yb):
    """First version (raw values; unstable when the given gas has almost no spread). Kept for the record."""
    ca, cb = xa - xa.mean(1, keepdims=True), xb - xb.mean(1, keepdims=True)
    var = (cb ** 2).mean(1)
    ok = var > 1e-12
    b = np.where(ok, (ca * cb).mean(1) / np.where(ok, var, 1.0), 0.0)
    return xa - b[:, None] * (xb - yb[:, None]), b


def condition(xa, xb, yb):
    """Regression adjustment in rank (normal-score) space. xa, xb [N,S] samples of the target and the given
    gas; yb [N] observed given value. Returns conditional target samples and the normal-score correlation."""
    s = xa.shape[1]
    u = norm.ppf(rankdata(xa, axis=1) / (s + 1))
    v = norm.ppf(rankdata(xb, axis=1) / (s + 1))
    below = (xb < yb[:, None]).sum(1) + 0.5 * (xb == yb[:, None]).sum(1)
    v_obs = norm.ppf((below + 0.5) / (s + 1))
    uc, vc = u - u.mean(1, keepdims=True), v - v.mean(1, keepdims=True)
    den = np.sqrt((uc ** 2).sum(1) * (vc ** 2).sum(1))
    rho = np.where(den > 1e-12, (uc * vc).sum(1) / np.where(den > 1e-12, den, 1.0), 0.0)
    u_cond = u - rho[:, None] * (v - v_obs[:, None])
    pos = np.clip(norm.cdf(u_cond) * (s + 1) - 1, 0, s - 1)          # rank position 0..S-1
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, s - 1)
    ordered = np.sort(xa, axis=1)
    frac = pos - lo
    rows = np.arange(len(xa))[:, None]
    return ordered[rows, lo] * (1 - frac) + ordered[rows, hi] * frac, rho


def scores(xs, ya):
    covered, width, score = interval_units(xs, ya)
    return {"coverage80": covered, "width80": width, "mis80": score, "crps": crps_units(xs, ya),
            "abs_error": np.abs(np.median(xs, axis=1) - ya), "n": np.ones(len(ya))}


def run(variant, year, seed):
    te, x = samples(variant, year, seed)
    d = mg.data()
    y, mask = d["y"][te], d["mask"][te]
    saved = np.load(ROOT / "06_结果" / f"multigas_{lk.NAMES[variant]}" / f"fold{year}_seed{seed}" / "per_window.npz")
    worst = 0.0                                          # the regenerated samples are the evaluated ones
    for k in range(4):
        b_i, t_i = np.nonzero(mask[:, k])
        crps = np.zeros(len(te))
        np.add.at(crps, b_i, crps_units(x[b_i, :, k, t_i], y[b_i, k, t_i]))
        worst = max(worst, float(np.abs(crps - saved["cell_crps"][:, k]).max()))
    assert worst < 1e-8, worst
    out = {}
    for label, target, given in TARGETS:
        b_i, t_i = np.nonzero(mask[:, target] & mask[:, given])
        xa, xb = x[b_i, :, target, t_i], x[b_i, :, given, t_i]
        ya, yb = y[b_i, target, t_i], y[b_i, given, t_i]
        xc, slope = condition(xa, xb, yb)
        for kind, xs in (("uncond", xa), ("cond", xc)):
            for m, v in scores(xs, ya).items():
                a = np.zeros(len(te))
                np.add.at(a, b_i, v)
                out[f"{label}__{kind}__{m}"] = a
        rho = np.array([np.corrcoef(xa[i], xb[i])[0, 1] if xb[i].std() > 0 and xa[i].std() > 0 else 0.0 for i in range(len(xa))])
        a = np.zeros(len(te))
        np.add.at(a, b_i, rho)
        out[f"{label}__sample_correlation_sum"] = a
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{variant}_fold{year}_seed{seed}.npz", months=d["months"][te], origins=d["origins"][te], **out)
    print(f"done {variant} {year} {seed} (check vs saved scores: {worst:.1e})", flush=True)


def load(variant, folds):
    """Per-window sums pooled over folds and averaged over seeds."""
    parts = []
    for year in folds:
        z = [dict(np.load(OUT / f"{variant}_fold{year}_seed{s}.npz")) for s in SEEDS]
        keys = [k for k in z[0] if "__" in k]
        parts.append({"months": z[0]["months"], "origins": z[0]["origins"],
                      **{k: np.mean([zz[k] for zz in z], 0) for k in keys}})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summary():
    from .summary3 import Bootstrap
    result = {"note": "conditional estimation given the same-day observed value of the other gas; not a forecast"}
    lines = ["# 已知当天 CH₄ 实测值时的 CO₂（以及反过来）", "",
             "情景：同一站同一天，CH₄ 已测到、CO₂ 还不知道（补缺测、数据晚到）。这是“已知一种气体时估计另一种”，不是更好的 14 天预测。",
             "做法：在模型自己的 512 种情景里做回归调整，没有用实测数据估计任何参数。自动生成，数字来自 co2_given_ch4_summary.json。", ""]
    for period, folds in PERIODS.items():
        data = {v: load(v, folds) for v in ("G", "H")}
        boot = Bootstrap(data["G"]["origins"])
        seasons = np.array([SEASON[int(m)] for m in data["G"]["months"]])
        res = {}
        for label, _, _ in TARGETS:
            res[label] = {}
            for sel_name, sel in [("all", None)] + [(s, seasons == s) for s in ("DJF", "MAM", "JJA", "SON")]:
                row = {}
                for v in ("G", "H"):
                    for kind in ("uncond", "cond"):
                        den = data[v][f"{label}__{kind}__n"]
                        for m in METRICS[:-1]:
                            num = data[v][f"{label}__{kind}__{m}"]
                            w = np.ones(len(num)) if sel is None else sel.astype(float)
                            row[f"{v}_{kind}_{m}"] = float((num * w).sum() / (den * w).sum())
                    row[f"{v}_mean_sample_correlation"] = float(
                        (data[v][f"{label}__sample_correlation_sum"] * (1 if sel is None else sel)).sum()
                        / (data[v][f"{label}__cond__n"] * (1 if sel is None else sel)).sum())
                for m in ("width80", "mis80", "crps", "abs_error", "coverage80"):
                    pc = (data["G"][f"{label}__cond__{m}"], data["G"][f"{label}__cond__n"])
                    pu = (data["G"][f"{label}__uncond__{m}"], data["G"][f"{label}__uncond__n"])
                    ph = (data["H"][f"{label}__cond__{m}"], data["H"][f"{label}__cond__n"])
                    row[f"G_cond_minus_uncond_{m}"] = boot.diff(pc, pu, sel)
                    row[f"G_cond_minus_H_cond_{m}"] = boot.diff(pc, ph, sel)
                res[label][sel_name] = row
        result[period] = res
        title = "主检验 2020–2025" if period == "main" else "确认 2014–2019"
        lines += [f"## {title}", "",
                  "| 格 | 季节 | 模型里两者的相关 | 覆盖率 未知→已知 | 宽度 未知→已知 | MIS80 未知→已知 | CRPS 未知→已知 | 中位数误差 未知→已知 | 宽度变化（95% 区间） | MIS80 变化 | H 已知时的 MIS80 | G 减 H（MIS80） |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for label, _, _ in TARGETS:
            nd = 3 if "CO2" in label else 2
            for sel_name in ("all", "DJF", "MAM", "JJA", "SON"):
                r = res[label][sel_name]
                f = lambda k: f"{r[k]:.{nd}f}"
                dw, dm, gh = r["G_cond_minus_uncond_width80"], r["G_cond_minus_uncond_mis80"], r["G_cond_minus_H_cond_mis80"]
                lines.append(
                    f"| {label} | {sel_name} | {r['G_mean_sample_correlation']:+.2f} | {100 * r['G_uncond_coverage80']:.1f}% → {100 * r['G_cond_coverage80']:.1f}% | "
                    f"{f('G_uncond_width80')} → {f('G_cond_width80')} | {f('G_uncond_mis80')} → {f('G_cond_mis80')} | "
                    f"{f('G_uncond_crps')} → {f('G_cond_crps')} | {f('G_uncond_abs_error')} → {f('G_cond_abs_error')} | "
                    f"{100 * dw['difference'] / r['G_uncond_width80']:+.1f}% [{dw['ci95'][0]:+.{nd}f}, {dw['ci95'][1]:+.{nd}f}] | "
                    f"{100 * dm['difference'] / r['G_uncond_mis80']:+.1f}% [{dm['ci95'][0]:+.{nd}f}, {dm['ci95'][1]:+.{nd}f}] | "
                    f"{f('H_cond_mis80')} | {gh['difference']:+.{nd}f} [{gh['ci95'][0]:+.{nd}f}, {gh['ci95'][1]:+.{nd}f}] |")
            lines.append("")
    write_json(OLD / "07_报告/co2_given_ch4_summary.json", result)
    (OLD / "07_报告/co2_given_ch4_tables.md").write_text("\n".join(lines) + "\n")
    print("written")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "summary":
        summary()
    else:
        raise SystemExit("unknown stage")
