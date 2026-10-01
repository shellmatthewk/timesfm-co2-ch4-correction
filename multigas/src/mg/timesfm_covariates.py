"""TimesFM 3 with CH4 as a covariate of CO2, to compare with G + correction (user request 2026-09-30).

All variants use the 4-station CO2 context as targets, like the project's native TimesFM:
  none        : no covariates (must reproduce the saved native quantiles)
  past        : CH4 at BRW and MLO over the past 14 days (past_only_covariates)
  past_future : CH4 at BRW and MLO over the past 14 days and the 14 forecast days as observed
                (past_future_covariates; days without a CH4 observation are masked) - the same
                information the correction uses
Scored (no calibration) on the 14 test windows of the comparison table (BRW + MLO, every CO2 cell) and on
the rolling test windows 2020-2025 per station on the cells where CO2 and CH4 are both observed (the cells of
co2_given_ch4). TimesFM gives quantiles, not samples, so there is no CRPS or ES. MPS, batches of 8.

Usage (PYTHONPATH=src): python -m mg.timesfm_covariates
"""
from __future__ import annotations

import json
import numpy as np
import torch

from .encode import backbone                      # first: puts the base project's uq14 (with the TimesFM code) on the path
from . import multigas as mg                      # noqa: E402  (its uq14.metrics and conformal are byte-identical)
from .data import ROOT, write_json                # noqa: E402
from .old_split_g import splits, evaluate_predictions   # noqa: E402

OUT = ROOT / "06_结果" / "timesfm_covariates"
VARIANTS = ("none", "past", "past_future")
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def inputs(rows):
    d = mg.data()
    co2, ch4 = np.load(ROOT / "02_数据/daily_co2_windows.npz"), np.load(ROOT / "02_数据/daily_ch4_windows.npz")
    pc = {o: i for i, o in enumerate(co2["origins"])}
    ph = {o: i for i, o in enumerate(ch4["origins"])}
    ic = np.array([pc[o] for o in d["origins"][rows]])
    ih = np.array([ph[o] for o in d["origins"][rows]])
    assert np.array_equal(co2["y"][ic][:, :2], d["y"][rows][:, :2], equal_nan=True)
    return co2["x"][ic][:, :2], ch4["x"][ih], np.nan_to_num(ch4["y"][ih]), ch4["mask"][ih].astype(bool)  # two-station rerun 2026-10-01: BRW, MLO only


@torch.no_grad()
def quantiles(variant, co2x, ch4x, ch4y, ch4m, device="mps", batch=8):
    body = backbone(device)
    out = []
    for s in range(0, len(co2x), batch):
        sl = slice(s, s + batch)
        target = torch.from_numpy(np.ascontiguousarray(co2x[sl], dtype=np.float32)).to(device)
        kw = {}
        if variant == "past":
            kw["past_only_covariates"] = torch.from_numpy(np.ascontiguousarray(ch4x[sl], dtype=np.float32)).to(device)
        elif variant == "past_future":
            values = np.concatenate([ch4x[sl], np.where(ch4m[sl], ch4y[sl], 0.0)], axis=-1).astype(np.float32)
            masked = np.concatenate([np.zeros_like(ch4m[sl]), ~ch4m[sl]], axis=-1)
            kw["past_future_covariates"] = torch.from_numpy(values).to(device)
            kw["past_future_mask"] = torch.from_numpy(masked).to(device)
        q = body.decode(target=target, horizon=14, **kw)
        assert q.shape[1] >= target.shape[1] and q.shape[2] == 14, tuple(q.shape)
        out.append(q[:, :target.shape[1]].sort(dim=-1).values.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def cell_scores(q, y):
    lo, med, hi = q[..., 0], q[..., 4], q[..., 8]
    width = hi - lo
    mis = width + 10.0 * np.maximum(lo - y, 0) + 10.0 * np.maximum(y - hi, 0)
    return {"coverage80": ((y >= lo) & (y <= hi)).astype(float), "width80": width, "mis80": mis, "abs_error": np.abs(med - y)}


def main():
    d = mg.data()
    te14 = splits("grid")[3]
    roll = np.concatenate([mg.fold(y)["test"] for y in mg.FOLDS])
    rows = np.unique(np.concatenate([te14, roll]))
    where = {r: i for i, r in enumerate(rows)}
    x = inputs(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    q = {}
    for v in VARIANTS:
        q[v] = quantiles(v, *x)
        np.savez_compressed(OUT / f"{v}.npz", rows=rows, origins=d["origins"][rows], quantiles=q[v][:, :2])
        print("encoded", v, q[v].shape, flush=True)
    check = float(np.abs(q["none"][:, :2] - d["quantiles"][rows][:, :2]).max())

    y, mask = d["y"], d["mask"].astype(bool)
    i14 = np.array([where[r] for r in te14])
    table14 = {}
    for v in VARIANTS:
        res = evaluate_predictions(y[te14][:, :2], mask[te14][:, :2], quantiles=q[v][i14][:, :2], station_names=["BRW", "MLO"])
        table14[v] = {k: res["overall"][k] for k in ("mae", "rmse", "wis9", "coverage80", "mean_width80", "mis80")}
        table14[v].update({f"{s}_mis80": res["by_station"][s]["mis80"] for s in ("BRW", "MLO")})
        table14[v].update({f"{s}_coverage80": res["by_station"][s]["coverage80"] for s in ("BRW", "MLO")})

    iroll = np.array([where[r] for r in roll])
    seasons = np.array([SEASON[int(m)] for m in d["months"][roll]])
    rolling = {}
    for k, label in ((0, "BRW_CO2"), (1, "MLO_CO2")):
        b, t = np.nonzero(mask[roll][:, k] & mask[roll][:, k + 2])
        rolling[label] = {}
        for sel_name in ("all", "DJF", "MAM", "JJA", "SON"):
            keep = np.ones(len(b), bool) if sel_name == "all" else seasons[b] == sel_name
            rolling[label][sel_name] = {v: {m: float(val[keep].mean()) for m, val in
                                            cell_scores(q[v][iroll][b, k, t], y[roll][b, k, t]).items()} for v in VARIANTS}
            rolling[label][sel_name]["n_cells"] = int(keep.sum())
    write_json(ROOT / "07_报告/timesfm_covariates.json", {"reproduction_check_none_vs_saved": check, "same14": table14,
                                                        "rolling_2020_2025": rolling, "note": __doc__.split("Usage")[0].strip()})
    print("largest difference, no-covariate run vs saved native quantiles:", check)
    for v in VARIANTS:
        r = table14[v]
        print(f"14 windows {v:11s} " + " ".join(f"{m}={r[m]:.3f}" for m in r))
    for label in rolling:
        for sel in ("all", "DJF", "MAM", "JJA", "SON"):
            r = rolling[label][sel]
            print(f"rolling {label} {sel:3s} n={r['n_cells']:5d} " + " | ".join(
                f"{v}: mis {r[v]['mis80']:.3f} cov {r[v]['coverage80']:.3f} w {r[v]['width80']:.3f} mae {r[v]['abs_error']:.3f}" for v in VARIANTS))


def rolling():
    """The same variants (no covariates, CH4 past and future) on the 2020-2025 rolling test windows: main() without the
    14-window part, which needs data that are not in this repository."""
    d = mg.data()
    rows = np.concatenate([mg.fold(y)["test"] for y in mg.FOLDS])
    x = inputs(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for v in ("none", "past_future"):
        q = quantiles(v, *x)
        np.savez_compressed(OUT / f"{v}.npz", rows=rows, origins=d["origins"][rows], quantiles=q[:, :2])
        if v == "none":
            print("largest difference vs saved native quantiles:", float(np.abs(q[:, :2] - d["quantiles"][rows][:, :2]).max()), flush=True)
        print("encoded", v, q.shape, flush=True)


def confirm():
    """The same variants (no covariates, CH4 past and future) on the 2014-2019 confirmation test windows."""
    d = mg.data()
    rows = np.concatenate([mg.fold(y)["test"] for y in mg.CONFIRM_FOLDS])
    x = inputs(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for v in ("none", "past_future"):
        q = quantiles(v, *x)
        np.savez_compressed(OUT / f"{v}_confirm.npz", rows=rows, origins=d["origins"][rows], quantiles=q[:, :2])
        if v == "none":
            print("largest difference vs saved native quantiles:", float(np.abs(q[:, :2] - d["quantiles"][rows][:, :2]).max()), flush=True)
        print("encoded", v, q.shape, flush=True)


def compare():
    """G + correction minus TimesFM + CH4 (past and future) on the rolling cells, month-block bootstrap."""
    from .conditional_co2 import load
    from .summary3 import Bootstrap
    d = mg.data()
    roll = np.concatenate([mg.fold(y)["test"] for y in mg.FOLDS])
    g = load("G", mg.FOLDS)
    assert np.array_equal(g["origins"], d["origins"][roll])
    boot = Bootstrap(g["origins"])
    seasons = np.array([SEASON[int(m)] for m in d["months"][roll]])
    y, mask = d["y"][roll], d["mask"][roll].astype(bool)
    res = {}
    for v in ("none", "past_future"):
        z = np.load(OUT / f"{v}.npz")
        where = {r: i for i, r in enumerate(z["rows"])}
        q = z["quantiles"][[where[r] for r in roll]]
        res[v] = {}
        for k, label in ((0, "BRW_CO2"), (1, "MLO_CO2")):
            both = mask[:, k] & mask[:, k + 2]
            assert np.allclose(both.sum(1), g[f"{label}__cond__n"])
            sc = cell_scores(q[:, k], np.nan_to_num(y[:, k]))
            res[v][label] = {}
            for sel_name in ("all", "DJF", "MAM", "JJA", "SON"):
                sel = None if sel_name == "all" else seasons == sel_name
                res[v][label][sel_name] = {m: boot.diff((g[f"{label}__cond__{m}"], g[f"{label}__cond__n"]),
                                                        (np.where(both, sc[m], 0.0).sum(1), both.sum(1).astype(float)), sel)
                                           for m in ("mis80", "width80", "coverage80", "abs_error")}
    write_json(ROOT / "07_报告/timesfm_covariates_vs_correction.json", {"note": "G + correction minus TimesFM variant, rolling 2020-2025 cells with CO2 and CH4 observed; month-block bootstrap", **res})
    for label in ("BRW_CO2", "MLO_CO2"):
        for sel in ("all", "DJF", "MAM", "JJA", "SON"):
            r = res["past_future"][label][sel]["mis80"]
            print(f"{label} {sel:3s} MIS80 G+correction minus TimesFM+CH4: {r['difference']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    import sys
    {"compare": compare, "confirm": confirm, "rolling": rolling}.get(sys.argv[1] if sys.argv[1:] else "", main)()
