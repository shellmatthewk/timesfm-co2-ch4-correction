"""Step 3 summary: seed-averaged per-window scores pooled over the six test years,
paired block bootstrap over calendar months of the forecast origin, the
pre-registered success criteria, and the implied-versus-observed correlation check.

Usage: PYTHONPATH=src python -m mg.summary3            test years 2020-2025 -> 07_报告/step3_summary.json, step3_tables.md
       PYTHONPATH=src python -m mg.summary3 confirm    test years 2014-2019 (B, D, G, H) -> 07_报告/step3_confirm_summary.json, ..._tables.md
"""
from __future__ import annotations

import json
import numpy as np

from .data import ROOT, write_json
from .multigas import FOLDS, CONFIRM_FOLDS, SLOTS

METHODS = {"A": "multigas_A_linked_conditional", "B": "multigas_B_unlinked",
           "C": "multigas_C_linked_unconditional", "D": "multigas_D_timesfm_independent",
           "E": "multigas_E_explicit_conditional", "F": "multigas_F_explicit_constant",
           "G": "multigas_G_explicit_conditional_pairloss", "H": "multigas_H_explicit_constant_pairloss"}
NAMES = {"A": "A 联动·随情况变", "B": "B 不联动", "C": "C 联动·只看自己的历史", "D": "D TimesFM（独立）",
         "E": "E 显式 r·随情况变", "F": "F 显式 r·每站固定",
         "G": "G 显式 r·随情况变＋二维损失", "H": "H 显式 r·每站固定＋二维损失"}
SEEDS = (17, 29, 43)
RUN_FOLDS, TAG, METHOD_FILTER = FOLDS, "step3", None      # switched by the "confirm" argument
N_BOOT = 4000
CROSS_CRITERIA = ("gas_difference_crps", "variogram_cross_gas", "both_above_brier")
CELL = ("crps", "mis80", "coverage80", "width80", "abs_error")
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
          9: "SON", 10: "SON", 11: "SON"}
SEASONS = ("DJF", "MAM", "JJA", "SON")
STATION_PAIRS = {"BRW": (0, 2), "MLO": (1, 3)}


def load(method):
    """{metric: (num[N], den[N])} pooled over folds and averaged over seeds, plus the diagnostics."""
    folds = []
    for year in RUN_FOLDS:
        runs = [ROOT / "06_结果" / METHODS[method] / f"fold{year}_seed{s}" for s in SEEDS]
        if not all((r / "metrics.json").exists() for r in runs):
            return None
        pw = [dict(np.load(r / "per_window.npz")) for r in runs]
        dg = [dict(np.load(r / "diagnostic.npz")) for r in runs]
        mj = [json.loads((r / "metrics.json").read_text()) for r in runs]
        c = np.array([mj[0]["normalization_constants"][s] for s in SLOTS])
        metrics = {}
        for key in pw[0]:
            if key.endswith("__num"):
                name = key[:-5]
                den = pw[0][name + "__den"]
                assert all(np.array_equal(p[name + "__den"], den) for p in pw)
                metrics[name] = (np.mean([p[key] for p in pw], 0), den)
        cell = {k: np.mean([p[f"cell_{k}"] for p in pw], 0) for k in CELL}
        n = pw[0]["cell_n"]
        for gas, cols in (("CO2", [0, 1]), ("CH4", [2, 3])):
            for k in CELL:
                metrics[f"{gas}_{k}"] = (cell[k][:, cols].sum(1), n[:, cols].sum(1))
        for i, slot in enumerate(SLOTS):
            for k in CELL:
                metrics[f"{slot}_{k}"] = (cell[k][:, i], n[:, i])
        for k in ("crps", "mis80", "width80"):
            metrics[f"all4_normalized_{k}"] = ((cell[k] / c).sum(1), n.sum(1))
        folds.append({"metrics": metrics, "origins": dg[0]["origins"], "months": dg[0]["months"],
                      "history": dg[0]["history_comovement"], "year": np.full(len(n), year),
                      "implied": np.nanmean([d["implied_correlation"] for d in dg], 0),
                      "pit": np.stack([d["observed_pit_score"] for d in dg]),
                      "test_windows": mj[0]["test_windows"], "selected_epochs": [m.get("selected_epoch") for m in mj],
                      "learned_r": (np.mean([np.load(r / "linkage_r.npz")["r"] for r in runs], 0)
                                    if (runs[0] / "linkage_r.npz").exists() else np.full((len(n), 2), np.nan))})
    common = [k for k in folds[0]["metrics"] if all(k in f["metrics"] for f in folds)]
    out = {"metrics": {k: (np.concatenate([f["metrics"][k][0] for f in folds]),
                           np.concatenate([f["metrics"][k][1] for f in folds])) for k in common}}
    for k in ("origins", "months", "history", "year", "implied", "learned_r"):
        out[k] = np.concatenate([f[k] for f in folds])
    out["pit"] = np.concatenate([f["pit"] for f in folds], axis=1)          # [seed, N, 4, 14]
    out["selected_epochs"] = {str(y): f["selected_epochs"] for y, f in zip(RUN_FOLDS, folds)}
    return out


def ratio(pair, sel=None):
    num, den = pair
    if sel is not None:
        num, den = num[sel], den[sel]
    return float(num.sum() / den.sum()) if den.sum() > 0 else None


class Bootstrap:
    """Paired resampling of calendar-month blocks (same draws for every method and metric)."""

    def __init__(self, origins, seed=20260929):
        labels = np.array([o[:7] for o in origins])
        self.blocks, self.block_of = np.unique(labels, return_inverse=True)
        rng = np.random.default_rng(seed)
        nb = len(self.blocks)
        self.counts = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_BOOT)])

    def reps(self, pair, sel=None):
        num, den = pair
        w = np.ones(len(num)) if sel is None else sel.astype(float)
        bn = np.bincount(self.block_of, weights=num * w, minlength=len(self.blocks))
        bd = np.bincount(self.block_of, weights=den * w, minlength=len(self.blocks))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.counts @ bn) / (self.counts @ bd)

    def diff(self, p1, p2, sel=None):
        v1, v2 = ratio(p1, sel), ratio(p2, sel)
        est = np.nan if v1 is None or v2 is None else v1 - v2
        r = self.reps(p1, sel) - self.reps(p2, sel)
        lo, hi = np.nanpercentile(r, [2.5, 97.5])
        return {"difference": est, "ci95": [float(lo), float(hi)]}

    def ci(self, pair, sel=None):
        lo, hi = np.nanpercentile(self.reps(pair, sel), [2.5, 97.5])
        return [float(lo), float(hi)]


def correlation(u, v):
    ok = np.isfinite(u) & np.isfinite(v)
    return float(np.corrcoef(u[ok], v[ok])[0, 1]) if ok.sum() > 10 else None


def diagnostic(d, groups):
    """For each station and group: mean implied correlation and observed correlation of the PIT scores
    over the (window, day) cells with both labels observed."""
    out = {}
    for st, (k1, k2) in STATION_PAIRS.items():
        both = np.isfinite(d["pit"][0][:, k1]) & np.isfinite(d["pit"][0][:, k2])        # [N,14]
        imp = d["implied"][:, 0 if st == "BRW" else 1]
        for g, sel in groups.items():
            cells = both & sel[:, None]
            obs = [correlation(p[:, k1][cells], p[:, k2][cells]) for p in d["pit"]]
            out[f"{st}|{g}"] = {"implied": float(np.nanmean(imp[cells])) if cells.any() else None,
                                "observed": float(np.mean(obs)) if None not in obs else None,
                                "cells": int(cells.sum())}
    return out


def reliability(d, n_bins=5):
    """Cells (window, station, day) with both labels, grouped into quintiles of the model's implied
    correlation (per station); within each group the mean implied correlation and the observed
    correlation of the model's own PIT scores. A useful conditional link gives observed values that
    rise across the groups and track the implied ones."""
    out = {}
    for st, (k1, k2) in STATION_PAIRS.items():
        both = np.isfinite(d["pit"][0][:, k1]) & np.isfinite(d["pit"][0][:, k2])
        imp = d["implied"][:, 0 if st == "BRW" else 1]
        ok = both & np.isfinite(imp)
        edges = np.quantile(imp[ok], np.linspace(0, 1, n_bins + 1))
        group = np.clip(np.searchsorted(edges, imp, side="right") - 1, 0, n_bins - 1)
        rows = []
        for g in range(n_bins):
            cells = ok & (group == g)
            obs = [correlation(p[:, k1][cells], p[:, k2][cells]) for p in d["pit"]]
            rows.append({"implied_mean": float(imp[cells].mean()), "observed": float(np.mean(obs)) if None not in obs else None,
                         "cells": int(cells.sum()), "implied_range": [float(edges[g]), float(edges[g + 1])]})
        out[st] = rows
    return out


def fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def main():
    data = {m: load(m) for m in METHODS if METHOD_FILTER is None or m in METHOD_FILTER}
    ready = [m for m, d in data.items() if d is not None]
    summary = {"methods_complete": ready, "n_boot": N_BOOT,
               "note": "no conformal calibration and no empirical copula; single cells in original units (CO2 ppm, CH4 ppb); joint scores on deviations from the TimesFM median divided by fixed per-slot constants"}
    if not ready:
        write_json(ROOT / f"07_报告/{TAG}_summary.json", summary)
        return
    ref = data[ready[0]]
    for m in ready[1:]:
        assert np.array_equal(data[m]["origins"], ref["origins"]), m
    boot = Bootstrap(ref["origins"])
    summary.update(test_windows=int(len(ref["origins"])), month_blocks=int(len(boot.blocks)),
                   windows_by_year={str(y): int((ref["year"] == y).sum()) for y in RUN_FOLDS})
    metric_names = [k for k in ref["metrics"] if all(k in data[m]["metrics"] for m in ready)]
    summary["pooled"] = {m: {k: {"value": ratio(data[m]["metrics"][k]), "ci95": boot.ci(data[m]["metrics"][k])}
                             for k in metric_names} for m in ready}
    summary["by_year"] = {m: {str(y): {k: ratio(data[m]["metrics"][k], ref["year"] == y) for k in metric_names}
                              for y in RUN_FOLDS} for m in ready}
    seasons = np.array([SEASON[int(x)] for x in ref["months"]])
    summary["by_season"] = {m: {s: {k: ratio(data[m]["metrics"][k], seasons == s) for k in metric_names}
                                for s in SEASONS} for m in ready}
    pairs = [(a, b) for a, b in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "D"), ("C", "B"),
                                  ("E", "B"), ("E", "F"), ("E", "A"), ("E", "D"), ("F", "B"), ("F", "A"),
                                  ("G", "H"), ("G", "B"), ("G", "E"), ("H", "F"), ("G", "A"), ("G", "D"))
             if a in ready and b in ready]
    summary["differences"] = {f"{a}-{b}": {k: boot.diff(data[a]["metrics"][k], data[b]["metrics"][k]) for k in metric_names}
                              for a, b in pairs}
    summary["differences_by_season"] = {
        f"{a}-{b}": {s: {k: boot.diff(data[a]["metrics"][k], data[b]["metrics"][k], seasons == s)
                         for k in metric_names if k.split("_")[0] in ("BRW", "MLO", "gas", "both", "variogram", "energy", "CO2", "CH4")}
                     for s in SEASONS} for a, b in pairs if (a, b) in (("A", "B"), ("A", "C"), ("E", "B"), ("E", "F"), ("E", "A"), ("G", "H"), ("G", "B"), ("G", "E"))}
    summary["selected_epochs"] = {m: data[m]["selected_epochs"] for m in ready if m != "D"}
    if all(m in ready for m in "ABC"):
        def beats(a, b):
            return {k: summary["differences"][f"{a}-{b}"][k]["ci95"][1] < 0 for k in CROSS_CRITERIA}
        not_worse = {f"{g}_{k}": summary["differences"]["A-B"][f"{g}_{k}"]["ci95"][0] <= 0
                     for g in ("CO2", "CH4") for k in ("crps", "mis80")}
        ab, ac = beats("A", "B"), beats("A", "C")
        summary["success_criteria"] = {
            "rule": "A beats B on difference CRPS, cross-gas variogram and both-above Brier (95% intervals below 0), "
                    "single-cell CRPS and MIS80 of both gases not significantly worse than B, and A beats C on the same three",
            "A_beats_B": ab, "A_not_worse_than_B_single_cell": not_worse, "A_beats_C": ac,
            "met": all(ab.values()) and all(not_worse.values()) and all(ac.values())}
    if all(m in ready for m in "BEF"):
        def beats2(a, b):
            return {k: summary["differences"][f"{a}-{b}"][k]["ci95"][1] < 0 for k in CROSS_CRITERIA}
        not_worse2 = {f"{g}_{k}": summary["differences"]["E-B"][f"{g}_{k}"]["ci95"][0] <= 0
                      for g in ("CO2", "CH4") for k in ("crps", "mis80")}
        ef, eb = beats2("E", "F"), beats2("E", "B")
        summary["success_criteria_round2"] = {
            "rule": "E beats F on difference CRPS, cross-gas variogram and both-above Brier (95% intervals below 0); "
                    "E beats B on the same three; single-cell CRPS and MIS80 of both gases not significantly worse than B",
            "E_beats_F": ef, "E_beats_B": eb, "E_not_worse_than_B_single_cell": not_worse2,
            "met": all(ef.values()) and all(eb.values()) and all(not_worse2.values())}
    if all(m in ready for m in "BGH"):
        def beats3(a, b):
            return {k: summary["differences"][f"{a}-{b}"][k]["ci95"][1] < 0 for k in CROSS_CRITERIA}
        not_worse3 = {f"{g}_{k}": summary["differences"]["G-B"][f"{g}_{k}"]["ci95"][0] <= 0
                      for g in ("CO2", "CH4") for k in ("crps", "mis80")}
        gh, gb = beats3("G", "H"), beats3("G", "B")
        summary["success_criteria_round3"] = {
            "rule": "G beats H on difference CRPS, cross-gas variogram and both-above Brier (95% intervals below 0); "
                    "G beats B on the same three; single-cell CRPS and MIS80 of both gases not significantly worse than B",
            "G_beats_H": gh, "G_beats_B": gb, "G_not_worse_than_B_single_cell": not_worse3,
            "met": all(gh.values()) and all(gb.values()) and all(not_worse3.values())}
    # implied versus observed correlation
    diag_groups = {"all": np.ones(len(seasons), bool)}
    diag_groups.update({s: seasons == s for s in SEASONS})
    diag_groups.update({f"month{m:02d}": ref["months"] == m for m in range(1, 13)})
    summary["diagnostic"] = {}
    for m in ready:
        d = data[m]
        groups = dict(diag_groups)
        for i, st in enumerate(("BRW", "MLO")):
            h = d["history"][:, i]
            cut = np.quantile(h, [1 / 3, 2 / 3])
            ter = np.digitize(h, cut)
            for t, name in enumerate(("low", "mid", "high")):
                groups[f"{st}_history_{name}"] = ter == t
                for s in SEASONS:
                    groups[f"{st}_history_{name}_{s}"] = (ter == t) & (seasons == s)
        summary["diagnostic"][m] = diagnostic(d, groups)
    summary["dependence_reliability"] = {m: reliability(data[m]) for m in ready if m in ("A", "C", "E", "F", "G", "H")}
    summary["learned_r"] = {}
    for m in ready:
        if m not in ("E", "F", "G", "H"):
            continue
        d = data[m]
        per = {}
        for i, st in enumerate(("BRW", "MLO")):
            r = d["learned_r"][:, i]
            h = d["history"][:, i]
            ter = np.digitize(h, np.quantile(h, [1 / 3, 2 / 3]))
            per[st] = {"all": float(r.mean()), "sd_over_windows": float(r.std()),
                       **{s: float(r[seasons == s].mean()) for s in SEASONS},
                       **{f"month{k:02d}": float(r[d["months"] == k].mean()) for k in range(1, 13)},
                       **{f"history_{nm}": float(r[ter == t].mean()) for t, nm in enumerate(("low", "mid", "high"))}}
        summary["learned_r"][m] = per
    write_json(ROOT / f"07_报告/{TAG}_summary.json", summary)
    tables(summary, ready)
    print(json.dumps(summary.get("success_criteria"), ensure_ascii=False, indent=1))
    print(json.dumps(summary.get("success_criteria_round2"), ensure_ascii=False, indent=1))
    print(json.dumps(summary.get("success_criteria_round3"), ensure_ascii=False, indent=1))


def tables(s, ready):
    title = "第三步汇总表" if TAG == "step3" else "第三步确认（测试年 2014–2019）汇总表"
    L = [f"# {title}（自动生成，数字来自 {TAG}_summary.json）", "",
         f"测试窗口 {s['test_windows']} 个（{', '.join(f'{y}: {n}' for y, n in s['windows_by_year'].items())}），"
         f"按预报起点所在月份分成 {s['month_blocks']} 块做配对 bootstrap（{s['n_boot']} 次）。"
         "没有共形校准，没有经验 copula。单格指标用原始单位（CO₂ ppm，CH₄ ppb）；联合指标用“偏离 TimesFM 中位数 ÷ 固定常数”。", ""]
    L += ["## 1. 单格指标（每种气体，两站合并）", "",
          "| 方法 | CO₂ 覆盖率 | CO₂ 宽度 | CO₂ MIS80 | CO₂ CRPS | CO₂ MAE | CH₄ 覆盖率 | CH₄ 宽度 | CH₄ MIS80 | CH₄ CRPS | CH₄ MAE |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ready:
        p = s["pooled"][m]
        row = [NAMES[m]]
        for g, nd in (("CO2", 3), ("CH4", 2)):
            row += [f"{100 * p[f'{g}_coverage80']['value']:.1f}%", fmt(p[f"{g}_width80"]["value"], nd),
                    fmt(p[f"{g}_mis80"]["value"], nd), fmt(p[f"{g}_crps"]["value"], nd), fmt(p[f"{g}_abs_error"]["value"], nd)]
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## 2. 单格指标（每站每种气体）", "", "| 方法 | 格 | 覆盖率 | 宽度 | MIS80 | CRPS |", "|---|---|---|---|---|---|"]
    for m in ready:
        for slot in SLOTS:
            p = s["pooled"][m]
            nd = 3 if "CO2" in slot else 2
            L.append(f"| {NAMES[m]} | {slot} | {100 * p[f'{slot}_coverage80']['value']:.1f}% | {fmt(p[f'{slot}_width80']['value'], nd)} | "
                     f"{fmt(p[f'{slot}_mis80']['value'], nd)} | {fmt(p[f'{slot}_crps']['value'], nd)} |")
    L += ["", "## 3. 两种气体之间的指标（同一站同一天）", "",
          "| 方法 | 差值 CRPS | 差值覆盖率 | 差值宽度 | 同时偏高 Brier | 四象限 Brier | 跨气体 variogram | 两气体同时覆盖带覆盖率 | 带宽 | 和值 CRPS | Energy Score | 同气体跨天 variogram |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ready:
        p = s["pooled"][m]
        L.append(f"| {NAMES[m]} | {fmt(p['gas_difference_crps']['value'])} | {100 * p['gas_difference_coverage80']['value']:.1f}% | "
                 f"{fmt(p['gas_difference_width80']['value'], 3)} | {fmt(p['both_above_brier']['value'])} | {fmt(p['quadrant_brier']['value'])} | "
                 f"{fmt(p['variogram_cross_gas']['value'])} | {100 * p['band_two_gas_coverage80']['value']:.1f}% | {fmt(p['band_two_gas_width80']['value'], 3)} | "
                 f"{fmt(p['gas_sum_crps']['value'])} | {fmt(p['energy_score']['value'])} | {fmt(p['variogram_within_series']['value'])} |")
    L += ["", "## 4. 两两比较（前者减后者，负数＝前者更好；覆盖率除外），括号内为 95% 区间", ""]
    keys = [("gas_difference_crps", "差值 CRPS"), ("variogram_cross_gas", "跨气体 variogram"), ("both_above_brier", "同时偏高 Brier"),
            ("quadrant_brier", "四象限 Brier"), ("gas_sum_crps", "和值 CRPS"), ("energy_score", "Energy Score"),
            ("CO2_crps", "CO₂ CRPS"), ("CO2_mis80", "CO₂ MIS80"), ("CH4_crps", "CH₄ CRPS"), ("CH4_mis80", "CH₄ MIS80"),
            ("all4_normalized_crps", "四格标准化 CRPS")]
    for title, comps in (("第一轮", [c for c in s["differences"] if c[0] in "ABC"]),
                         ("第二轮（显式联动强弱）", [c for c in s["differences"] if c[0] in "EF"]),
                         ("第三轮（加二维 Energy Score 损失）", [c for c in s["differences"] if c[0] in "GH"])):
        if not comps:
            continue
        L += [f"**{title}**", "", "| 指标 | " + " | ".join(comps) + " |", "|---|" + "---|" * len(comps)]
        for k, label in keys:
            cells = []
            for c in comps:
                d = s["differences"][c][k]
                nd = 2 if k.startswith("CH4") else 4
                star = " *" if d["ci95"][1] < 0 or d["ci95"][0] > 0 else ""
                cells.append(f"{d['difference']:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}]{star}")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["\\* 区间不含 0。", ""]
    if "success_criteria" in s:
        sc = s["success_criteria"]
        yes = lambda v: "是" if v else "否"
        L += ["## 5. 事先定好的成功标准", "", f"- A 比 B 好（三项跨气体指标）：" + "，".join(f"{k} {yes(v)}" for k, v in sc["A_beats_B"].items()),
              f"- A 的单格指标不显著比 B 差：" + "，".join(f"{k} {yes(v)}" for k, v in sc["A_not_worse_than_B_single_cell"].items()),
              f"- A 比 C 好（三项跨气体指标）：" + "，".join(f"{k} {yes(v)}" for k, v in sc["A_beats_C"].items()),
              f"- **总判定：{'达标' if sc['met'] else '未达标'}**", ""]
    if "success_criteria_round2" in s:
        sc = s["success_criteria_round2"]
        yes = lambda v: "是" if v else "否"
        L += ["第二轮（显式联动强弱，开始训练前写进配置）：", "",
              f"- E 比 F 好（三项跨气体指标）：" + "，".join(f"{k} {yes(v)}" for k, v in sc["E_beats_F"].items()),
              f"- E 比 B 好（三项跨气体指标）：" + "，".join(f"{k} {yes(v)}" for k, v in sc["E_beats_B"].items()),
              f"- E 的单格指标不显著比 B 差：" + "，".join(f"{k} {yes(v)}" for k, v in sc["E_not_worse_than_B_single_cell"].items()),
              f"- **总判定：{'达标' if sc['met'] else '未达标'}**", ""]
    if "success_criteria_round3" in s:
        sc = s["success_criteria_round3"]
        yes = lambda v: "是" if v else "否"
        L += ["第三轮（加二维 Energy Score 损失，开始训练前写进配置）：", "",
              f"- G 比 H 好（三项跨气体指标）：" + "，".join(f"{k} {yes(v)}" for k, v in sc["G_beats_H"].items()),
              f"- G 比 B 好（三项跨气体指标）：" + "，".join(f"{k} {yes(v)}" for k, v in sc["G_beats_B"].items()),
              f"- G 的单格指标不显著比 B 差：" + "，".join(f"{k} {yes(v)}" for k, v in sc["G_not_worse_than_B_single_cell"].items()),
              f"- **总判定：{'达标' if sc['met'] else '未达标'}**", ""]
    L += ["## 6. 分站的跨气体指标", "", "| 方法 | 站 | 差值 CRPS | 同时偏高 Brier | 跨气体 variogram | 两气体带覆盖率 |", "|---|---|---|---|---|---|"]
    for m in ready:
        for st in ("BRW", "MLO"):
            p = s["pooled"][m]
            L.append(f"| {NAMES[m]} | {st} | {fmt(p[f'{st}_gas_difference_crps']['value'])} | {fmt(p[f'{st}_both_above_brier']['value'])} | "
                     f"{fmt(p[f'{st}_variogram_cross_gas']['value'])} | {100 * p[f'{st}_band_two_gas_coverage80']['value']:.1f}% |")
    if s.get("differences_by_season"):
        L += ["", "## 7. 分季节的两两比较（前者减后者，负数＝前者更好），95% 区间", ""]
        for comp, per in s["differences_by_season"].items():
            L += [f"**{comp}**", "", "| 季节 | BRW 差值 CRPS | MLO 差值 CRPS | BRW 同时偏高 Brier | MLO 同时偏高 Brier |", "|---|---|---|---|---|"]
            for se in SEASONS:
                ds = per[se]
                cells = [f"{ds[k]['difference']:+.4f} [{ds[k]['ci95'][0]:+.4f}, {ds[k]['ci95'][1]:+.4f}]"
                         for k in ("BRW_gas_difference_crps", "MLO_gas_difference_crps", "BRW_both_above_brier", "MLO_both_above_brier")]
                L.append(f"| {se} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["", "## 8. 模型给出的联动 vs 实际联动（CO₂ 与 CH₄ 偏差的相关）", "",
          "“给出”＝模型样本里两种气体的相关；“实际”＝观测值在模型自己的分布里所处位置（PIT 正态分数）的相关。模型对，两者应接近。", "",
          "| 方法 | 站 | 全年 给出/实际 | DJF | MAM | JJA | SON |", "|---|---|---|---|---|---|---|"]
    for m in ready:
        for st in ("BRW", "MLO"):
            dg = s["diagnostic"][m]
            cells = [f"{fmt(dg[f'{st}|{g}']['implied'], 2)} / {fmt(dg[f'{st}|{g}']['observed'], 2)}" for g in ("all",) + SEASONS]
            L.append(f"| {NAMES[m]} | {st} | " + " | ".join(cells) + " |")
    L += ["", "按近 14 天两种气体走势的相关分三档（低、中、高）：", "", "| 方法 | 站 | 低 给出/实际 | 中 | 高 |", "|---|---|---|---|---|"]
    for m in ready:
        for st in ("BRW", "MLO"):
            dg = s["diagnostic"][m]
            cells = [f"{fmt(dg[f'{st}|{st}_history_{t}']['implied'], 2)} / {fmt(dg[f'{st}|{st}_history_{t}']['observed'], 2)}"
                     for t in ("low", "mid", "high")]
            L.append(f"| {NAMES[m]} | {st} | " + " | ".join(cells) + " |")
    if s.get("dependence_reliability"):
        L += ["", "按模型给出的联动强弱把格子分成五档（每站分开），每档里实际联动是多少：", "",
              "| 方法 | 站 | " + " | ".join(f"第{i + 1}档 给出/实际" for i in range(5)) + " |", "|---|---|" + "---|" * 5]
        for m, per in s["dependence_reliability"].items():
            for st, rows in per.items():
                L.append(f"| {NAMES[m]} | {st} | " + " | ".join(f"{fmt(r['implied_mean'], 2)} / {fmt(r['observed'], 2)}" for r in rows) + " |")
    if s.get("learned_r"):
        L += ["", "## 8b. 第二、三轮学到的联动强弱 r（测试窗口，三种子平均）", "",
              "| 方法 | 站 | 全年 | DJF | MAM | JJA | SON | 走势一致 低 / 中 / 高 | 各窗口的标准差 |", "|---|---|---|---|---|---|---|---|---|"]
        for m, per in s["learned_r"].items():
            for st, v in per.items():
                L.append(f"| {NAMES[m]} | {st} | {v['all']:+.2f} | " + " | ".join(f"{v[se]:+.2f}" for se in SEASONS) +
                         f" | {v['history_low']:+.2f} / {v['history_mid']:+.2f} / {v['history_high']:+.2f} | {v['sd_over_windows']:.2f} |")
        L += ["", "| 方法 | 站 | " + " | ".join(f"{k}月" for k in range(1, 13)) + " |", "|---|---|" + "---|" * 12]
        for m, per in s["learned_r"].items():
            for st, v in per.items():
                L.append(f"| {NAMES[m]} | {st} | " + " | ".join(f"{v[f'month{k:02d}']:+.2f}" for k in range(1, 13)) + " |")
    L += ["", "## 9. 每个测试年（三种子平均）", "", "| 方法 | 指标 | " + " | ".join(str(y) for y in RUN_FOLDS) + " |", "|---|---|" + "---|" * len(RUN_FOLDS)]
    for k, label in (("gas_difference_crps", "差值 CRPS"), ("both_above_brier", "同时偏高 Brier"), ("CO2_mis80", "CO₂ MIS80"),
                     ("CH4_mis80", "CH₄ MIS80"), ("CO2_coverage80", "CO₂ 覆盖率"), ("CH4_coverage80", "CH₄ 覆盖率")):
        for m in ready:
            vals = [s["by_year"][m][str(y)][k] for y in RUN_FOLDS]
            nd = 2 if k.startswith("CH4_mis") else 4 if "coverage" not in k else 3
            L.append(f"| {NAMES[m]} | {label} | " + " | ".join(fmt(v, nd) for v in vals) + " |")
    (ROOT / f"07_报告/{TAG}_tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["confirm"]:
        RUN_FOLDS, TAG, METHOD_FILTER = CONFIRM_FOLDS, "step3_confirm", ("B", "D", "G", "H")
    main()
