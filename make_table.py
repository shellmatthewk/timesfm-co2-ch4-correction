"""The ten schemes on the large-sample test (two stations, BRW and MLO, CO2): one table.

Reads only files of this repository (results/):
  rolling_corrections/{tfm,res}_fold*.npz      TimesFM and median-locked Engression, without correction and + LSTM
                                              (three seeds already averaged)
  rolling_noise/{T1,N1}_fold*_seed*.npz        T1 and N1, without correction and + LSTM
  rolling_basic/basic_fold*_seed*.npz          basic Engression with 1280-dim noise (ID 5 setting)
  timesfm_ch4/windows*.npz                     TimesFM + CH4 covariate (quantile output)
Per window: sums over the window's cells of each score and the number of cells; seeds are averaged per window, then
the test years of a period are put together. 95% intervals of differences: calendar-month blocks resampled with the
Bootstrap below, a copy of mg.summary3.Bootstrap (multigas/src/mg/summary3.py; same seed, same 4000 draws as the
large-sample reports in results/reports/), so that this script needs numpy only.

Usage: python make_table.py      -> 总表.md, 总表.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
N_BOOT = 4000

PERIODS = {"2020–2025": ((2020, 2021, 2022, 2023, 2024, 2025), ""), "2014–2019": ((2014, 2015, 2016, 2017, 2018, 2019), "_confirm")}
SEEDS = (17, 29, 43)
STATIONS = {"BRW": "BRW_CO2", "MLO": "MLO_CO2"}
METRICS = ("mis80", "coverage80", "width80", "crps", "abs_error")
# number, name, uses the CH4 observed in the 14 forecast days, (source, version)
SCHEMES = [(1, "TimesFM 不修正", False, ("tfm", "none")),
           (2, "基础 Engression（1280 维噪声）", False, ("basic", "none")),
           (3, "锁中位数 Engression", False, ("res", "none")),
           (4, "T1：TimesFM + 随情况变的宽度", False, ("T1", "none")),
           (5, "N1：Engression + 噪声大小随情况变", False, ("N1", "none")),
           (6, "TimesFM + CH₄ 协变量", True, ("tfm_ch4", None)),
           (7, "TimesFM + 双向 LSTM", True, ("tfm", "bilstm")),
           (8, "锁中位数 Engression + 双向 LSTM", True, ("res", "bilstm")),
           (9, "T1 + 双向 LSTM", True, ("T1", "lstm")),
           (10, "N1 + 双向 LSTM", True, ("N1", "lstm"))]


def ratio(pair, sel=None):
    num, den = pair
    if sel is not None:
        num, den = num[sel], den[sel]
    return float(num.sum() / den.sum()) if den.sum() > 0 else None


class Bootstrap:
    """Paired resampling of calendar-month blocks (same draws for every method and metric); copy of mg.summary3.Bootstrap."""

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


def load(years, tag):
    cat = lambda parts: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data = {base: cat([dict(np.load(RES / "rolling_corrections" / f"{base}_fold{y}.npz")) for y in years]) for base in ("tfm", "res")}
    for base, prefix in (("T1", "rolling_noise/T1"), ("N1", "rolling_noise/N1"), ("basic", "rolling_basic/basic")):
        per_year = []
        for y in years:
            runs = [dict(np.load(RES / f"{prefix}_fold{y}_seed{s}.npz")) for s in SEEDS]
            per_year.append({k: runs[0][k] if k in ("origins", "months") else np.mean([r[k] for r in runs], 0) for k in runs[0]})
        data[base] = cat(per_year)
    data["tfm_ch4"] = dict(np.load(RES / "timesfm_ch4" / f"windows{tag}.npz"))
    for base in data:
        assert np.array_equal(data[base]["origins"], data["tfm"]["origins"]), base
    return data


def get(data, source, label, metric, kind="both"):
    base, version = source
    if base == "tfm_ch4":
        return data[base].get(f"{label}__both__{metric}") if kind == "both" else None
    return data[base][f"{label}__{kind}__{version}__{metric}"]


def main():
    fmt = lambda d: f"{d['difference']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    mark = lambda d: " *" if d["ci95"][1] < 0 else " †" if d["ci95"][0] > 0 else ""
    result, detail, overview = {}, [], {}
    for period, (years, tag) in PERIODS.items():
        data = load(years, tag)
        boot = Bootstrap(data["tfm"]["origins"])
        result[period] = {}
        for st, label in STATIONS.items():
            ref_none = (get(data, SCHEMES[0][3], label, "mis80"), get(data, SCHEMES[0][3], label, "n"))
            ref_ch4 = (get(data, SCHEMES[5][3], label, "mis80"), get(data, SCHEMES[5][3], label, "n"))
            rows = {}
            detail += [f"### {st}，{period}", "", "| 编号 | 方案 | MIS80 | 减 TimesFM 不修正 | 减 TimesFM + CH₄ | 覆盖率 | 区间宽度 | CRPS | MAE |",
                       "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
            for num, name, _, source in SCHEMES:
                n = get(data, source, label, "n")
                r = {m: (None if get(data, source, label, m) is None else float(get(data, source, label, m).sum() / n.sum())) for m in METRICS}
                r["n_cells"] = int(n.sum())
                pair = (get(data, source, label, "mis80"), n)
                r["minus_timesfm"] = None if num == 1 else boot.diff(pair, ref_none)
                r["minus_timesfm_ch4"] = None if num == 6 else boot.diff(pair, ref_ch4)
                if source[0] != "tfm_ch4":
                    n2 = get(data, source, label, "n", "co2only")
                    r["co2only_cells"] = {"n_cells": int(n2.sum()), **{m: float(get(data, source, label, m, "co2only").sum() / n2.sum()) for m in METRICS}}
                rows[num] = r
                cell = lambda v, pct=False: "—" if v is None else (f"{100 * v:.1f}%" if pct else f"{v:.3f}")
                detail.append(f"| {num} | {name} | {r['mis80']:.3f} | {'—' if r['minus_timesfm'] is None else fmt(r['minus_timesfm']) + mark(r['minus_timesfm'])} | "
                              f"{'—' if r['minus_timesfm_ch4'] is None else fmt(r['minus_timesfm_ch4']) + mark(r['minus_timesfm_ch4'])} | "
                              f"{cell(r['coverage80'], True)} | {cell(r['width80'])} | {cell(r['crps'])} | {cell(r['abs_error'])} |")
                overview[(num, st, period)] = f"{r['mis80']:.3f}" + ("" if r["minus_timesfm_ch4"] is None else mark(r["minus_timesfm_ch4"]))
            detail.append("")
            result[period][st] = rows
    cols = [(st, p) for p in PERIODS for st in STATIONS]
    L = ["# 十个方案：两站大样本总表", "",
         "自动生成（`python make_table.py`）。两种气体当天都有观测的格子，单位 ppm，三个种子平均，不做共形校准。",
         "* 表示比 TimesFM + CH₄ 显著更好，† 表示显著更差：差值的 95% 区间不含 0，区间按月分块重抽测试年的窗口得到。", "",
         "## 1. MIS80 总览（越低越好）", "",
         "| 编号 | 方案 | 用到未来 14 天的 CH₄ | " + " | ".join(f"{st} {p}" for st, p in cols) + " |",
         "|---:|---|:---:|" + "---:|" * len(cols)]
    for num, name, uses, _ in SCHEMES:
        L.append(f"| {num} | {name} | {'是' if uses else '否'} | " + " | ".join(overview[(num, st, p)] for st, p in cols) + " |")
    L += ["", "## 2. 细节（覆盖率越接近 80% 越好，其余越低越好）", "",
          "“减 TimesFM 不修正”“减 TimesFM + CH₄”是 MIS80 的差值和 95% 区间，负数表示更好。TimesFM + CH₄ 只给分位数，没有样本，算不了 CRPS。", ""]
    L += detail
    L += ["## 3. 当天没测到 CH₄ 的格子（MIS80）", "",
          "这些格子上，只看当天的修正不起作用，双向 LSTM 仍可以借前后几天的 CH₄。TimesFM + CH₄ 的结果只保存了两种气体都有观测的格子。", "",
          "| 编号 | 方案 | " + " | ".join(f"{st} {p}" for st, p in cols) + " |", "|---:|---|" + "---:|" * len(cols)]
    for num, name, _, source in SCHEMES:
        if source[0] == "tfm_ch4":
            continue
        L.append(f"| {num} | {name} | " + " | ".join(f"{result[p][st][num]['co2only_cells']['mis80']:.3f}" for st, p in cols) + " |")
    L.append("| | 格子数 | " + " | ".join(str(result[p][st][1]["co2only_cells"]["n_cells"]) for st, p in cols) + " |")
    (HERE / "总表.md").write_text("\n".join(L) + "\n")
    (HERE / "总表.json").write_text(json.dumps({p: {st: {str(k): v for k, v in rows.items()} for st, rows in r.items()} for p, r in result.items()},
                                               ensure_ascii=False, indent=1) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
