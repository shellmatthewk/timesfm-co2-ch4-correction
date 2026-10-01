"""Other ways to get the linkage r used by the correction (user request 2026-09-30).

G's marginal samples are kept; only the correlation used to correct CO2 with the same-day observed CH4 changes.
Correction for every variant: Gaussian-copula conditional in rank space,
    z_i = rho * v_obs + sqrt(1 - rho^2) * u_i   (u_i: normal scores of the CO2 sample ranks),
then back through the CO2 sample quantiles. v_obs is the normal score of the observed CH4 among the CH4 samples.

Variants (all but "oracle" use only the fold's training years 2010..Y-2, validation Y-1 for early stopping):
  current : the reported correction (regression adjustment with G's within-sample correlation)
  g_gauss : G's within-sample correlation, Gaussian formula above
  fixed   : one correlation per station
  month   : one correlation per station and month of the forecast origin
  doy     : per station, kernel average over the day of year of the forecast day (bandwidth 10 days)
  mlp     : small neural network, inputs day of year (sin, cos), 14-day co-movement, station, lead;
            trained on the bivariate normal likelihood of the observed normal scores
  oracle  : per station and month, estimated on the test year itself - uses test data, upper bound only
Observed normal scores for fitting come from G (seed 17 of the fold, 256 samples). Test: seeds 17, 29, 43 with the
evaluation samples. Cells: CO2 and CH4 both observed at the station and day (the cells of co2_given_ch4).

Confirmation (user request 2026-09-30, after the 2020-2025 results): "mlp" was chosen as the method and is the
one pre-specified comparison on 2014-2019 (never used for design); the other variants are shown for reference.

Usage (PYTHONPATH=src): python -m mg.r_variants run|report [confirm]
"""
from __future__ import annotations

import sys
import numpy as np
import torch
from scipy.stats import norm

from . import multigas as mg, linkage as lk
from .data import ROOT, write_json
from .conditional_co2 import SEEDS, SEASON, condition, scores, samples

OUT = ROOT / "06_结果" / "r_variants"
VARIANTS = ("none", "current", "g_gauss", "fixed", "month", "doy", "mlp", "oracle")
METRICS = ("coverage80", "width80", "mis80", "crps", "abs_error", "n")
LABELS = ("BRW_CO2", "MLO_CO2")
CLIP = 0.97


def model_of(year, seed):
    ck = ROOT / "04_训练" / f"multigas_{lk.NAMES['G']}" / f"fold{year}_seed{seed}"
    e0 = mg.enc("A", mg.fold(year)["test"][:1])
    model = lk.build(seed, e0["features"].shape[-1])
    model.head.set_params(np.load(ck / "best_head.npz"))
    rhead = lk.RHead(5, lk.R_CFG["hidden"], np.random.default_rng(0))
    rhead.set_params(np.load(ck / "best_rhead.npz"))
    return model, rhead


def obs_score(xs, y):
    s = xs.shape[1]
    below = (xs < y[:, None]).sum(1) + 0.5 * (xs == y[:, None]).sum(1)
    return norm.ppf((below + 0.5) / (s + 1))


def day_of_year(dates):
    dd = np.asarray(dates).astype("datetime64[D]")
    return (dd - dd.astype("datetime64[Y]")).astype(int) + 1


def cells(idx, x=None, model=None, rhead=None, n_samples=256, seed=0, block=400):
    """Per (window, station, day) cell with both gases observed: features and the observed normal scores."""
    d = mg.data()
    rows = {k: [] for k in ("w", "s", "t", "month", "doy", "hist", "lead", "u", "v")}
    for start in range(0, len(idx), block):
        part = idx[start:start + block]
        xs = x[start:start + block] if x is not None else lk.predict(model, rhead, mg.enc("A", part), lk.z_inputs("E", part), n_samples, seed + start)[0]
        for s in (0, 1):
            b, t = np.nonzero(d["mask"][part][:, s] & d["mask"][part][:, s + 2])
            rows["w"].append(start + b)
            rows["s"].append(np.full(len(b), s))
            rows["t"].append(t)
            rows["month"].append(d["months"][part][b])
            rows["doy"].append(day_of_year(d["forecast_dates"][part][b, t]))
            rows["hist"].append(d["history_comovement"][part][b, s])
            rows["lead"].append(t)
            rows["u"].append(obs_score(xs[b, :, s, t], d["y"][part][b, s, t]))
            rows["v"].append(obs_score(xs[b, :, s + 2, t], d["y"][part][b, s + 2, t]))
    return {k: np.concatenate(v) for k, v in rows.items()}


def corr(u, v):
    return float(np.clip((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum()), -CLIP, CLIP))


def features(c):
    a = 2 * np.pi * c["doy"] / 365.25
    return np.stack([np.sin(a), np.cos(a), c["hist"], c["s"] == 0, c["s"] == 1, c["lead"] / 13.0], 1).astype(np.float32)


def fit_mlp(tr, va, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1))
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.zero_()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    t = {k: (torch.from_numpy(features(c)), torch.from_numpy(c["u"]).float(), torch.from_numpy(c["v"]).float()) for k, c in (("tr", tr), ("va", va))}

    def nll(X, u, v):
        rho = CLIP * torch.tanh(net(X).squeeze(-1))
        one = 1 - rho ** 2
        return (0.5 * torch.log(one) + (u * u - 2 * rho * u * v + v * v) / (2 * one) - (u * u + v * v) / 2).mean()

    best, state, bad = np.inf, None, 0
    for _ in range(800):
        opt.zero_grad()
        loss = nll(*t["tr"])
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = float(nll(*t["va"]))
        if val < best - 1e-6:
            best, state, bad = val, {k: p.clone() for k, p in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 60:
                break
    net.load_state_dict(state)
    return lambda c: CLIP * np.tanh(net(torch.from_numpy(features(c))).detach().numpy()[:, 0])


def fit(tr, va, seed):
    """Every trained rho variant as a function of the test cells."""
    fixed = {s: corr(tr["u"][tr["s"] == s], tr["v"][tr["s"] == s]) for s in (0, 1)}
    month = {(s, m): corr(tr["u"][(tr["s"] == s) & (tr["month"] == m)], tr["v"][(tr["s"] == s) & (tr["month"] == m)])
             for s in (0, 1) for m in range(1, 13)}
    doy = {}
    for s in (0, 1):
        sel = tr["s"] == s
        uv, uu, vv, dd = (tr["u"] * tr["v"])[sel], (tr["u"] ** 2)[sel], (tr["v"] ** 2)[sel], tr["doy"][sel]
        table = np.zeros(367)
        for day in range(1, 367):
            gap = np.minimum(np.abs(dd - day), 365 - np.abs(dd - day))
            w = np.exp(-0.5 * (gap / 10.0) ** 2)
            table[day] = np.clip((w * uv).sum() / np.sqrt((w * uu).sum() * (w * vv).sum()), -CLIP, CLIP)
        doy[s] = table
    mlp = fit_mlp(tr, va, seed)
    return {"fixed": lambda c: np.array([fixed[s] for s in c["s"]]),
            "month": lambda c: np.array([month[(s, m)] for s, m in zip(c["s"], c["month"])]),
            "doy": lambda c: np.array([doy[s][day] for s, day in zip(c["s"], c["doy"])]),
            "mlp": mlp}, {"fixed": fixed, "month": {f"{s}_{m}": v for (s, m), v in month.items()}}


def gauss_correct(xa, xb, yb, rho):
    s = xa.shape[1]
    ordered = np.sort(xa, axis=1)
    v_obs = obs_score(xb, yb)
    u = norm.ppf(np.arange(1, s + 1) / (s + 1))
    rho = np.clip(rho, -CLIP, CLIP)[:, None]
    pos = np.clip(norm.cdf(rho * v_obs[:, None] + np.sqrt(1 - rho ** 2) * u[None]) * (s + 1) - 1, 0, s - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, s - 1)
    rows = np.arange(len(xa))[:, None]
    return ordered[rows, lo] * (1 - (pos - lo)) + ordered[rows, hi] * (pos - lo)


def within_sample(xa, xb):
    s = xa.shape[1]
    from scipy.stats import rankdata
    u = norm.ppf(rankdata(xa, axis=1) / (s + 1))
    v = norm.ppf(rankdata(xb, axis=1) / (s + 1))
    uc, vc = u - u.mean(1, keepdims=True), v - v.mean(1, keepdims=True)
    den = np.sqrt((uc ** 2).sum(1) * (vc ** 2).sum(1))
    return np.where(den > 1e-12, (uc * vc).sum(1) / np.where(den > 1e-12, den, 1.0), 0.0)


def run(folds, tag):
    d = mg.data()
    OUT.mkdir(parents=True, exist_ok=True)
    fitted_all = {}
    for year in folds:
        f = mg.fold(year)
        model, rhead = model_of(year, 17)
        tr = cells(f["train"], model=model, rhead=rhead, seed=700_000)
        va = cells(f["validation"], model=model, rhead=rhead, seed=800_000)
        rho_fns, fitted = fit(tr, va, year)
        fitted_all[year] = fitted
        te = f["test"]
        sums = {}
        for seed in SEEDS:
            _, x = samples("G", year, seed)
            c = cells(te, x=x)
            oracle = {(s, m): corr(c["u"][(c["s"] == s) & (c["month"] == m)], c["v"][(c["s"] == s) & (c["month"] == m)])
                      for s in (0, 1) for m in range(1, 13) if ((c["s"] == s) & (c["month"] == m)).any()}
            rhos = {k: fn(c) for k, fn in rho_fns.items()}
            rhos["oracle"] = np.array([oracle[(s, m)] for s, m in zip(c["s"], c["month"])])
            for s, label in enumerate(LABELS):
                sel = c["s"] == s
                w, t = c["w"][sel], c["t"][sel]
                xa, xb = x[w, :, s, t], x[w, :, s + 2, t]
                ya, yb = d["y"][te][w, s, t], d["y"][te][w, s + 2, t]
                versions = {"none": xa, "current": condition(xa, xb, yb)[0], "g_gauss": gauss_correct(xa, xb, yb, within_sample(xa, xb))}
                versions.update({k: gauss_correct(xa, xb, yb, r[sel]) for k, r in rhos.items()})
                for v, xs in versions.items():
                    for m, val in scores(xs, ya).items():
                        a = np.zeros(len(te))
                        np.add.at(a, w, val)
                        sums.setdefault(f"{label}__{v}__{m}", []).append(a)
        np.savez_compressed(OUT / f"fold{year}.npz", origins=d["origins"][te], months=d["months"][te],
                            **{k: np.mean(v, 0) for k, v in sums.items()})
        print(f"fold {year} done; fixed r {fitted['fixed']}", flush=True)
    write_json(OUT / f"fitted{tag}.json", {str(k): v for k, v in fitted_all.items()})


def report(folds, tag):
    from .summary3 import Bootstrap
    parts = [dict(np.load(OUT / f"fold{y}.npz")) for y in folds]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    boot = Bootstrap(data["origins"])
    seasons = np.array([SEASON[int(m)] for m in data["months"]])
    # TimesFM + CH4 (past and future) on the same cells
    tf = np.load(ROOT / "06_结果" / "timesfm_covariates" / f"past_future{tag}.npz")
    d = mg.data()
    roll = np.concatenate([mg.fold(y)["test"] for y in folds])
    where = {r: i for i, r in enumerate(tf["rows"])}
    q = tf["quantiles"][[where[r] for r in roll]]
    y, mask = d["y"][roll], d["mask"][roll].astype(bool)
    result = {}
    period = "2014–2019 确认（事先只定了 mlp 一种）" if tag else "2020–2025 滚动检验"
    lines = [f"# r 的其他取法：修正后的 CO₂（{period}，两种气体当天都有观测的格子）", "",
             "自动生成（`python -m mg.r_variants report`）。G 的样本不变，只换修正时用的相关系数。单位 ppm，三个种子平均，不校准。", ""]
    for k, label in enumerate(LABELS):
        both = mask[:, k] & mask[:, k + 2]
        lo, med, hi = q[:, k, :, 0], q[:, k, :, 4], q[:, k, :, 8]
        yy = np.nan_to_num(y[:, k])
        tf_mis = np.where(both, (hi - lo) + 10 * np.maximum(lo - yy, 0) + 10 * np.maximum(yy - hi, 0), 0).sum(1)
        tf_n = both.sum(1).astype(float)
        result[label] = {}
        lines += [f"## {label}", "", "| 季节 | " + " | ".join(VARIANTS) + " | TimesFM+CH₄ |", "|---|" + "---:|" * (len(VARIANTS) + 1)]
        for sel_name in ("all", "DJF", "MAM", "JJA", "SON"):
            sel = None if sel_name == "all" else seasons == sel_name
            w = np.ones(len(seasons)) if sel is None else sel.astype(float)
            row = {}
            for v in VARIANTS:
                n = (data[f"{label}__{v}__n"] * w).sum()
                row[v] = {m: float((data[f"{label}__{v}__{m}"] * w).sum() / n) for m in METRICS[:-1]}
            row["timesfm_ch4"] = {"mis80": float((tf_mis * w).sum() / (tf_n * w).sum())}
            for v in ("month", "doy", "mlp", "fixed"):
                row[f"{v}_minus_current_mis80"] = boot.diff((data[f"{label}__{v}__mis80"], data[f"{label}__{v}__n"]),
                                                            (data[f"{label}__current__mis80"], data[f"{label}__current__n"]), sel)
                row[f"{v}_minus_timesfm_mis80"] = boot.diff((data[f"{label}__{v}__mis80"], data[f"{label}__{v}__n"]), (tf_mis, tf_n), sel)
            result[label][sel_name] = row
            lines.append(f"| {sel_name} | " + " | ".join(f"{row[v]['mis80']:.3f}" for v in VARIANTS) + f" | {row['timesfm_ch4']['mis80']:.3f} |")
        lines.append("")
    write_json(ROOT / f"07_报告/r_variants{tag}.json", result)
    (ROOT / f"07_报告/r_variants{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for label in LABELS:
        r = result[label]["all"]
        for v in ("fixed", "month", "doy", "mlp"):
            a, b = r[f"{v}_minus_current_mis80"], r[f"{v}_minus_timesfm_mis80"]
            print(f"{label} {v:6s} vs current {a['difference']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]"
                  f" | vs TimesFM+CH4 {b['difference']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
                  f" | cov {r[v]['coverage80']:.3f} width {r[v]['width80']:.3f} crps {r[v]['crps']:.3f} mae {r[v]['abs_error']:.3f}")


if __name__ == "__main__":
    folds, tag = (mg.CONFIRM_FOLDS, "_confirm") if sys.argv[2:] == ["confirm"] else (mg.FOLDS, "")
    run(folds, tag) if sys.argv[1] == "run" else report(folds, tag)
