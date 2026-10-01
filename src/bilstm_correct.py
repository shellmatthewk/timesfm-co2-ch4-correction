"""Corrections that look at all 14 days of CH4, not only the same day (user request 2026-10-01).

For each window and station, the observed CH4's position in its predictive distribution (normal score, missing days
masked) forms a 14-day sequence. A model reads it together with date, lead day, station and the 14-day co-movement
and outputs, for every day, the shift mu and spread sigma of the CO2 normal score; the CO2 samples are moved to
mu + sigma * u (u: normal scores of the sample ranks), as in r_variants.gauss_correct. Every CO2 cell is corrected,
also those whose same-day CH4 is missing.
  same_day : r_variants' small network r, mu = r v_t, sigma = sqrt(1 - r^2) on days with CH4 (the current correction)
  linear   : linear in the CH4 scores of days t-3..t+3 and the features (no neural network)
  bilstm   : bidirectional LSTM over the 14 days (one layer, 16 units per direction)
Base distributions as residual_rnet.py: the trained residual Engression for CO2 (row 15) and for CH4. linear and
bilstm are trained on the old split's daily windows 2017-2022 by the Gaussian likelihood of the observed CO2 normal
scores (validation 2023 every 7th day for early stopping), from the seed-17 base samples (256 per cell).
Scored on the 14 test windows like co2_same14 (BRW + MLO, raw), mean over the three base seeds.

Usage: python src/bilstm_correct.py

In this repository only the parts used by the large-sample scripts run (arrays, sequences, BiLSTM, fit, shift_correct,
same_day_params); main() (14 windows) needs files of the earlier projects that are not included.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_rnet as rr                                                   # noqa: E402
from residual_rnet import mg, splits, evaluate_predictions, cells, fit_mlp, heads, joint, predict, enc   # noqa: E402
from mg.r_variants import obs_score, day_of_year                            # noqa: E402

LAGS = range(-3, 4)


def arrays(idx, x):
    """Per window: CO2 scores u (NaN if unobserved), CH4 scores v with mask m, and the day features."""
    d = mg.data()
    y, mask = d["y"][idx], d["mask"][idx].astype(bool)
    u, v = np.full((len(idx), 2, 14), np.nan), np.zeros((len(idx), 2, 14))
    for s in (0, 1):
        b, t = np.nonzero(mask[:, s])
        u[b, s, t] = obs_score(x[b, :, s, t], y[b, s, t])
        b, t = np.nonzero(mask[:, s + 2])
        v[b, s, t] = obs_score(x[b, :, s + 2, t], y[b, s + 2, t])
    a = 2 * np.pi * day_of_year(d["forecast_dates"][idx].ravel()).reshape(len(idx), 14) / 365.25
    return {"u": u, "v": v, "m": mask[:, 2:4].astype(float), "sin": np.sin(a), "cos": np.cos(a),
            "hist": d["history_comovement"][idx]}


def sequences(A):
    """[windows * 2 stations, 14, features] and the CO2 targets."""
    W = len(A["u"])
    st = np.zeros((W, 2, 14, 2))
    st[:, 0, :, 0], st[:, 1, :, 1] = 1, 1
    day = lambda k: np.broadcast_to(A[k][:, None, :], (W, 2, 14))
    feats = np.stack([A["v"] * A["m"], A["m"], day("sin"), day("cos"), np.broadcast_to(np.arange(14) / 13.0, (W, 2, 14)),
                      np.broadcast_to(A["hist"][:, :, None], (W, 2, 14))], -1)
    X = np.concatenate([feats, st], -1).reshape(W * 2, 14, -1)
    lag = []
    for k in LAGS:
        vm, mm = np.zeros((W, 2, 14)), np.zeros((W, 2, 14))
        lo, hi = max(0, -k), min(14, 14 - k)
        vm[:, :, lo:hi] = (A["v"] * A["m"])[:, :, lo + k:hi + k]
        mm[:, :, lo:hi] = A["m"][:, :, lo + k:hi + k]
        lag += [vm, mm]
    L = np.concatenate([np.stack(lag, -1), feats[..., 2:], st], -1).reshape(W * 2, 14, -1)
    return X.astype(np.float32), L.astype(np.float32), A["u"].reshape(W * 2, 14)


class BiLSTM(torch.nn.Module):
    def __init__(self, f, h=16):
        super().__init__()
        self.lstm = torch.nn.LSTM(f, h, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(2 * h, 2)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        o = self.out(self.lstm(x)[0])
        return o[..., 0], o[..., 1].clamp(-3, 1)


class Linear(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.out = torch.nn.Linear(f, 2)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        o = self.out(x)
        return o[..., 0], o[..., 1].clamp(-3, 1)


def nll(model, X, u):
    mu, ls = model(X)
    ok = ~torch.isnan(u)
    z = torch.where(ok, u, torch.zeros_like(u))
    return ((ls + 0.5 * (z - mu) ** 2 * torch.exp(-2 * ls)) * ok).sum() / ok.sum()


def fit(kind, tr, va, seed=2017):
    torch.manual_seed(seed)
    X, Xv = (tr[0], va[0]) if kind == "bilstm" else (tr[1], va[1])
    model = BiLSTM(X.shape[-1]) if kind == "bilstm" else Linear(X.shape[-1])
    X, Xv = torch.from_numpy(X), torch.from_numpy(Xv)
    u, uv = torch.from_numpy(tr[2]).float(), torch.from_numpy(va[2]).float()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best, state, bad, hist = np.inf, None, 0, []
    for step in range(3000):
        opt.zero_grad()
        loss = nll(model, X, u)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = float(nll(model, Xv, uv))
        hist.append(val)
        if val < best - 1e-6:
            best, state, bad = val, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 150:
                break
    model.load_state_dict(state)
    return model, {"best_validation_nll": best, "steps": step + 1, "validation_nll_at_start": hist[0]}


def shift_correct(xa, mu, sigma):
    s = xa.shape[1]
    ordered = np.sort(xa, axis=1)
    u = norm.ppf(np.arange(1, s + 1) / (s + 1))
    pos = np.clip(norm.cdf(mu[:, None] + sigma[:, None] * u[None]) * (s + 1) - 1, 0, s - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, s - 1)
    rows = np.arange(len(xa))[:, None]
    return ordered[rows, lo] * (1 - (pos - lo)) + ordered[rows, hi] * (pos - lo)


def same_day_params(A, idx, x, rho_fn):
    """mu, sigma of the current correction: network r on days with CH4, nothing elsewhere."""
    W = len(idx)
    mu, sg = np.zeros((W, 2, 14)), np.ones((W, 2, 14))
    c = cells(idx, x=x)
    rho = rho_fn(c)
    mu[c["w"], c["s"], c["t"]] = rho * A["v"][c["w"], c["s"], c["t"]]
    sg[c["w"], c["s"], c["t"]] = np.sqrt(1 - rho ** 2)
    return mu, sg


def score_params(A, mu, sg):
    ok = ~np.isnan(A["u"])
    return float((np.log(sg) + 0.5 * ((A["u"] - mu) / sg) ** 2)[ok].mean())


def main():
    d = mg.data()
    tr_idx, va_idx = splits("daily")[:2]
    te = splits("grid")[3]
    base = np.load(rr.NEW / "方法/基础_20260920/02_数据/test_windows.npz")["origins"]
    keep = base != "2025-12-16"
    co2, ch4 = heads(17)
    xs = {"tr": [], "va": []}
    A = {}
    for name, idx, seed in (("tr", tr_idx, 700_000), ("va", va_idx, 800_000)):
        parts = [arrays(idx[s:s + 400], joint(idx[s:s + 400], co2, ch4, 256, seed + s)) for s in range(0, len(idx), 400)]
        A[name] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rho_fn = fit_mlp(rr.fit_cells(tr_idx, co2, ch4, 700_000), rr.fit_cells(va_idx, co2, ch4, 800_000), 2017)
    S = {k: sequences(A[k]) for k in ("tr", "va")}
    models, info = {}, {}
    for kind in ("linear", "bilstm"):
        models[kind], info[kind] = fit(kind, S["tr"], S["va"])
        print(kind, info[kind], flush=True)
    keys = ("mae", "rmse", "crps", "energy_score_observed_normalized", "wis9", "coverage80", "mean_width80", "mis80")
    runs = {k: [] for k in ("none", "same_day", "linear", "bilstm")}
    test_nll = {k: [] for k in runs}
    y, mask = d["y"][te][:, :2], d["mask"][te][:, :2]
    for seed in rr.SEEDS:
        co2, ch4 = heads(seed)
        saved = np.load(rr.R15 / "06_结果" / f"residual_centered_station_z1280_seed{seed}" / "test_predictions.npz")["samples"][keep].astype(np.float64)
        x = np.concatenate([saved, predict(ch4, enc(te, [2, 3]), 512, seed + mg.OFFSETS["test"])], axis=2)
        At = arrays(te, x)
        Xt, Lt, _ = sequences(At)
        params = {"none": (np.zeros((len(te), 2, 14)), np.ones((len(te), 2, 14))), "same_day": same_day_params(At, te, x, rho_fn)}
        with torch.no_grad():
            for kind, inp in (("linear", Lt), ("bilstm", Xt)):
                mu, ls = models[kind](torch.from_numpy(inp))
                params[kind] = (mu.numpy().reshape(len(te), 2, 14).astype(np.float64), np.exp(ls.numpy()).reshape(len(te), 2, 14).astype(np.float64))
        for kind, (mu, sg) in params.items():
            xc = x.copy()
            for s in (0, 1):
                for t in range(14):
                    xc[:, :, s, t] = shift_correct(x[:, :, s, t], mu[:, s, t], sg[:, s, t])
            res = evaluate_predictions(y, mask, samples=xc[:, :, :2], station_names=["BRW", "MLO"])
            r = {k: res["overall"][k] for k in keys}
            r.update({f"{st}_{k}": res["by_station"][st][k] for st in ("BRW", "MLO") for k in ("coverage80", "mean_width80", "mis80", "crps", "mae")})
            runs[kind].append(r)
            test_nll[kind].append(score_params(At, mu, sg))
    result = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in runs.items()}
    # likelihood of the observed CO2 scores on the training and validation years (lower is better)
    fit_nll = {}
    for name, idx in (("train", tr_idx), ("validation", va_idx)):
        a = A["tr" if name == "train" else "va"]
        Sx = S["tr" if name == "train" else "va"]
        co2, ch4 = heads(17)
        xs_ = None
        out = {"none": score_params(a, np.zeros_like(a["v"]), np.ones_like(a["v"]))}
        with torch.no_grad():
            for kind, inp in (("linear", Sx[1]), ("bilstm", Sx[0])):
                mu, ls = models[kind](torch.from_numpy(inp))
                out[kind] = score_params(a, mu.numpy().reshape(a["v"].shape), np.exp(ls.numpy()).reshape(a["v"].shape))
        fit_nll[name] = out
    result.update({"test_nll": {k: float(np.mean(v)) for k, v in test_nll.items()}, "fit_nll": fit_nll, "training": info})
    (rr.REPORT / "bilstm_correct_same14.json").write_text(json.dumps(result, indent=1) + "\n")
    for k in ("none", "same_day", "linear", "bilstm"):
        r = result[k]
        print(f"{k:9s} " + " ".join(f"{m}={r[m]:.3f}" for m in keys) + f" | BRW mis {r['BRW_mis80']:.3f} | MLO mis {r['MLO_mis80']:.3f} | test nll {result['test_nll'][k]:.4f}")
    print("fit nll:", json.dumps(fit_nll))


if __name__ == "__main__":
    main()
