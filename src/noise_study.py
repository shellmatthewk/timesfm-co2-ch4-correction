"""Learned rules for putting the Engression noise together (user request 2026-10-01): can a learned noise structure
make the residual Engression beat TimesFM? Pilot only; no test year of 2014-2025 is used:
  train   : windows whose 14 forecast days lie in 2000-2008
  select  : 2009 (validation)
  holdout : 2010-2013 (evaluation)
CO2 at BRW and MLO, two stations. PyTorch re-implementation of the residual Engression of row 15: head
g([h, eps]) 1280+1280 -> 64 -> 64 -> 14 (SiLU), paired-SiLU warm start g = 0 + 0.05 * eps[:14], one noise vector per
station, sample = TimesFM median + RevIN scale * (g - median over samples of g) (median locked to TimesFM's).

Arms (all settings written down before any result of this study was seen):
  E0  current Engression: Gaussian noise; Energy Score in ppm (28 dims, pair weight 1/2), 16 training samples;
      selection by validation ES on every 7th day of 2009
  N0  Gaussian noise with the common objective of N1-N5: per-cell fair CRPS / sigma_station + 0.5 * ES on
      station-normalized values, 64 training samples; selection on all 2009 windows by the mean over stations of
      (CRPS / CRPS_TimesFM + MIS80 / MIS80_TimesFM) / 2 (the earlier pilot's "crpses64" objective)
  N1  amplitude:  eps = exp(gamma(c)) * z                      (gamma limited to [-3, 3])
  N2  sources:    eps_s = alpha(c) * z_shared + sqrt(1 - alpha^2) * z_s   (z_shared common to the two stations)
  N3  days:       the first 14 noise dims (mapped to the 14 days by the warm start) are an AR(1) over the days, rho(c)
  N4  history:    first 14 noise dims = lambda(c) * a past standardized TimesFM error trajectory of a similar training
                  window + sqrt(1 - lambda^2) * Gaussian (30 nearest by the context; windows within 28 days excluded)
  N5  shape:      sinh-arcsinh transform of the Gaussian noise, skew and tail per block of 16 dims
N1-N4 parameters come from a small network of the context, N5's are free; all start at the identity (= N0 model).
Context c per window and station, standardized on the training windows: season (sin, cos of the first forecast day),
log mean TimesFM 80% width, log RevIN scale, log std of the day-to-day changes of the 14-day history (non-imputed days),
station one-hot. Head: AdamW lr 2e-4, decay 0.01; structure parameters: AdamW lr 1e-3, no decay; batch 32; at most
80 epochs, patience 12; gradient norms clipped at 1 (head and structure separately). Seeds 17, 29, 43.
Each trained run is also corrected with the bidirectional LSTM of bilstm_correct.py (CH4 positions from TimesFM's
quantiles, fitted on 2000-2008 with early stopping on 2009) and scored again, on the same holdout samples.
Holdout scores: three sampling seeds of 512 samples, averaged. Joint scores (uncorrected samples only; the LSTM
correction is cell by cell): Energy Score in ppm over the 14 days of each station and over both stations (128 samples),
correlation of the samples between adjacent days and between the stations.

Supplementary control T1 (added after the first N1 run, seed 17, had been seen): TimesFM's own distribution with the
same amplitude rule, residual sample = exp(gamma(c)) * (TimesFM sample - TimesFM median), samples drawn cell by cell
from TimesFM's quantiles (as timesfm_rnet.tf_samples); no Engression head. Same context network, objective, selection,
seeds and LSTM correction as N1. It tells whether N1's gain needs the Engression head or only the learned amplitude.

Usage: python noise_study.py tfmval | tfm | run ARM SEED | report        (NOISE_SMOKE=1: tiny quick check)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
sys.path.insert(0, str(NEW / "src"))
import rolling_residual as rres                                          # noqa: E402  (puts the multigas copy on the path)
import bilstm_correct as bc                                              # noqa: E402
from residual_rnet import enc                                            # noqa: E402
from timesfm_rnet import tf_samples                                      # noqa: E402
from mg.r_variants import day_of_year                                    # noqa: E402
from cop.joint_metrics import crps_units                                 # noqa: E402
from cop.marginal import icdf_from_z                                     # noqa: E402

torch.set_num_threads(2)
mg = rres.mg
SMOKE = os.environ.get("NOISE_SMOKE") == "1"
OUT = NEW / "results" / ("noise_pilot_smoke" if SMOKE else "noise_pilot")
ARMS = ("E0", "N0", "N1", "N2", "N3", "N4", "N5", "T1")
NAMES = {"E0": "现在的 Engression", "N0": "普通噪声（新训练目标）", "N1": "噪声大小随情况变", "N2": "两站共享一部分噪声",
         "N3": "天与天相连的噪声", "N4": "拼接历史误差", "N5": "学噪声的形状", "T1": "TimesFM + 同样的大小规则（追加对照）"}
SEEDS = (17, 29, 43)
SPLITS = {"train": ("2000-01-01", "2008-12-31"), "val": ("2009-01-01", "2009-12-31"), "hold": ("2010-01-01", "2013-12-31")}
D, Z, H, K = 1280, 1280, 64, 30
STN = ("BRW", "MLO")


def span(a, b):
    d = mg.data()
    f, l = d["forecast_dates"][:, 0], d["forecast_dates"][:, -1]
    idx = np.flatnonzero((f >= a) & (l <= b))
    return idx[::25] if SMOKE else idx


def raw_context(idx):
    d = mg.data()
    q = d["quantiles"][idx][:, :2]
    width = np.log((q[..., 8] - q[..., 0]).mean(-1) + 1e-6)
    scale = np.log(d["scale"][idx][:, :2, 0] + 1e-6)
    x, imp = d["x"][idx][:, :2].astype(np.float64), d["imputed_mask"][idx][:, :2].astype(bool)
    dx, ok = np.diff(x, axis=-1), ~(imp[..., 1:] | imp[..., :-1])
    vol = np.full((len(idx), 2), np.nan)
    for w in range(len(idx)):
        for s in range(2):
            if ok[w, s].sum() >= 3:
                vol[w, s] = dx[w, s][ok[w, s]].std()
    for s in range(2):
        vol[np.isnan(vol[:, s]), s] = np.nanmedian(vol[:, s])
    ang = 2 * np.pi * day_of_year(d["forecast_dates"][idx][:, 0]) / 365.25
    season = np.broadcast_to(np.stack([np.sin(ang), np.cos(ang)], -1)[:, None], (len(idx), 2, 2))
    return np.concatenate([season, width[..., None], scale[..., None], np.log(vol + 1e-3)[..., None]], -1)   # [W,2,5]


def load_split(name, norm=None):
    d = mg.data()
    idx = span(*SPLITS[name])
    e = enc(idx, [0, 1])
    rc = raw_context(idx)
    if norm is None:
        flat = rc.reshape(-1, rc.shape[-1])
        norm = (flat.mean(0), flat.std(0) + 1e-9)
    cs = (rc - norm[0]) / norm[1]
    c = np.concatenate([cs, np.broadcast_to(np.eye(2)[None], (len(idx), 2, 2))], -1)
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    y, m = d["y"][idx][:, :2], d["mask"][idx][:, :2].astype(bool)
    return idx, {"h": t(e["features"]), "scale": t(e["scale"]), "c": t(c), "ry": t(np.where(m, y - e["q50"], 0.0)), "m": torch.tensor(m),
                 "q50": e["q50"], "mask": m, "ctx_std": cs, "q": torch.as_tensor(np.ascontiguousarray(d["quantiles"][idx][:, :2]))}, norm


# ---------------------------------------------------------------- history library for N4
def library(tr_idx, ctx_tr):
    d = mg.data()
    y, m = d["y"][tr_idx][:, :2], d["mask"][tr_idx][:, :2].astype(bool)
    e = (y - d["quantiles"][tr_idx][:, :2, :, 4]) / d["scale"][tr_idx][:, :2]
    dates = d["forecast_dates"][tr_idx][:, 0].astype("datetime64[D]")
    lib = []
    for s in range(2):
        keep = np.flatnonzero(m[:, s].sum(1) >= 12)
        traj = np.stack([np.interp(np.arange(14), np.flatnonzero(m[w, s]), e[w, s][m[w, s]]) for w in keep])
        traj = traj / traj.std()
        lib.append({"traj": torch.tensor(traj, dtype=torch.float32), "ctx": ctx_tr[keep, s], "date": dates[keep]})
    return lib


def neighbours(lib, idx, ctx):
    d = mg.data()
    dates = d["forecast_dates"][idx][:, 0].astype("datetime64[D]")
    nb = np.zeros((len(idx), 2, K), dtype=np.int64)
    for s in range(2):
        L = lib[s]
        k = min(K, len(L["traj"]) - 1)
        for w in range(len(idx)):
            dist = ((L["ctx"] - ctx[w, s]) ** 2).sum(1)
            dist[np.abs((L["date"] - dates[w]).astype(int)) <= 28] = np.inf
            nb[w, s] = np.resize(np.argpartition(dist, k)[:k], K)
    return torch.tensor(nb)


# ---------------------------------------------------------------- model
class Net(torch.nn.Module):
    def __init__(self, arm, seed, cdim=7):
        super().__init__()
        torch.manual_seed(seed)
        self.arm = arm
        self.l1, self.l2, self.l3 = torch.nn.Linear(D + Z, H), torch.nn.Linear(H, H), torch.nn.Linear(H, 14)
        with torch.no_grad():                         # paired-SiLU warm start (rq.engression_np.Head.warm_start): g = 0 + 0.05 * eps[:14]
            A = torch.zeros(14, Z)
            A[:, :14] = torch.eye(14) * 0.05
            W1 = self.l1.weight
            W1[:28] = 0
            W1[:14, D:] = A
            W1[14:28, D:] = -A
            self.l1.bias[:28] = 0
            I = torch.eye(14)
            self.l2.weight[:28] = 0
            self.l2.weight[:14, :14], self.l2.weight[:14, 14:28] = I, -I
            self.l2.weight[14:28, :14], self.l2.weight[14:28, 14:28] = -I, I
            self.l2.bias[:28] = 0
            self.l3.weight.zero_()
            self.l3.weight[:, :14], self.l3.weight[:, 14:28] = I, -I
            self.l3.bias.zero_()
        if arm in ("N1", "N2", "N3", "N4", "T1"):
            self.ctx = torch.nn.Sequential(torch.nn.Linear(cdim, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1))
            torch.nn.init.zeros_(self.ctx[2].weight)
            torch.nn.init.zeros_(self.ctx[2].bias)
        if arm == "N5":
            self.skew = torch.nn.Parameter(torch.zeros(Z // 16))
            self.logtail = torch.nn.Parameter(torch.zeros(Z // 16))

    def head_params(self):
        return [p for m in (self.l1, self.l2, self.l3) for p in m.parameters()]

    def structure_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("l1", "l2", "l3"))]

    def structure(self, c):
        """Learned noise parameter per window and station [B,2]: gamma (N1), alpha (N2), rho (N3) or lambda (N4)."""
        v = self.ctx(c)[..., 0]
        return v.clamp(-3, 3) if self.arm in ("N1", "T1") else 0.97 * torch.tanh(v)

    def noise(self, B, S, c, gen, analog=None):
        z = torch.randn(B, S, 2, Z, generator=gen)
        a = self.arm
        if a in ("E0", "N0"):
            return z
        if a == "N5":
            zb = z.view(B, S, 2, Z // 16, 16)
            out = torch.sinh(torch.exp(self.logtail)[:, None] * torch.asinh(zb) - self.skew[:, None])
            return out.view(B, S, 2, Z)
        p = self.structure(c)[:, None, :, None]                                  # [B,1,2,1]
        if a == "N1":
            return torch.exp(p) * z
        if a == "N2":
            zs = torch.randn(B, S, 1, Z, generator=gen)
            return p * zs + torch.sqrt(1 - p ** 2) * z
        if a == "N3":
            xi, steps = z[..., :14], [z[..., 0]]
            for t in range(1, 14):
                steps.append(p[..., 0] * steps[-1] + torch.sqrt(1 - p[..., 0] ** 2) * xi[..., t])
            return torch.cat([torch.stack(steps, -1), z[..., 14:]], -1)
        if a == "N4":
            mix = p * analog + torch.sqrt(1 - p ** 2) * z[..., :14]
            return torch.cat([mix, z[..., 14:]], -1)
        raise ValueError(a)

    def residual(self, data, ids, S, gen, lib=None):
        """Samples minus the TimesFM median, ppm [B,S,2,14]."""
        ids = torch.as_tensor(ids)
        if self.arm == "T1":
            q = data["q"][ids]
            z = torch.randn(len(ids), S, 2, 14, generator=gen, dtype=torch.float64)
            r = (icdf_from_z(q, z) - q[..., 4][:, None]).float()
            return torch.exp(self.structure(data["c"][ids]))[:, None, :, None] * r
        h, c = data["h"][ids], data["c"][ids]
        analog = None
        if self.arm == "N4":
            pick = torch.randint(0, K, (len(ids), S, 2), generator=gen)
            rows = data["nb"][ids][:, None].expand(-1, S, -1, -1).gather(3, pick[..., None])[..., 0]     # [B,S,2]
            analog = torch.stack([lib[s]["traj"][rows[:, :, s]] for s in range(2)], 2)                    # [B,S,2,14]
        eps = self.noise(len(ids), S, c, gen, analog)
        W = self.l1.weight
        a1 = (h @ W[:, :D].T)[:, None] + eps @ W[:, D:].T + self.l1.bias
        g = self.l3(torch.nn.functional.silu(self.l2(torch.nn.functional.silu(a1))))
        return data["scale"][ids][:, None] * (g - torch.quantile(g, 0.5, dim=1, keepdim=True))


# ---------------------------------------------------------------- losses and scores
def crps_loss(r, ry, m, sig):
    S = r.shape[1]
    rn, yn = r / sig[None, None, :, None], ry / sig[None, :, None]
    xs = torch.sort(rn, dim=1).values
    w = (2 * torch.arange(1, S + 1, dtype=torch.float32) - S - 1).view(1, S, 1, 1)
    crps = (rn - yn[:, None]).abs().mean(1) - (xs * w).sum(1) / (S * (S - 1))
    return (crps * m).sum() / m.sum()


def es_loss(r, ry, m, sig):
    """As rq.engression_np.energy_score_and_grad (pair weight 1/2, each window / sqrt(observed dims), window mean)."""
    B, S = r.shape[:2]
    mf = m.float()
    xs = (r / sig[None, None, :, None] * mf[:, None]).reshape(B, S, -1)
    ys = (ry / sig[None, :, None] * mf).reshape(B, 1, -1)
    n = mf.reshape(B, -1).sum(1)
    first = torch.sqrt(((xs - ys) ** 2).sum(-1) + 1e-12).mean(1)
    pdist = torch.sqrt(((xs[:, :, None] - xs[:, None]) ** 2).sum(-1) + 1e-12)
    pairs = (pdist.sum((1, 2)) - S * 1e-6) / 2
    per = (first - pairs / (S * (S - 1))) / torch.sqrt(n.clamp(min=1))
    return per[n > 0].mean()


def station_metrics(r, data, q=None):
    """r numpy [W,S,2,14] residual samples; per station over observed CO2 cells (ppm)."""
    out = {}
    ry = data["ry"].numpy().astype(np.float64)
    for s, st in enumerate(STN):
        w, t = np.nonzero(data["mask"][:, s])
        xs, ya = r[w, :, s, t].astype(np.float64), ry[w, s, t]
        lo, hi = np.quantile(xs, [0.1, 0.9], axis=1)
        mis = (hi - lo) + 10 * np.maximum(lo - ya, 0) + 10 * np.maximum(ya - hi, 0)
        res = {"mis80": float(mis.mean()), "crps": float(crps_units(xs, ya).mean()), "cov80": float(((lo <= ya) & (ya <= hi)).mean()),
               "width80": float((hi - lo).mean()), "mae": float(np.abs(np.median(xs, 1) - ya).mean()), "n": int(len(ya))}
        if q is not None:
            ql, qh = q[w, s, t, 0] - q[w, s, t, 4], q[w, s, t, 8] - q[w, s, t, 4]
            res["mis80_exact"] = float(((qh - ql) + 10 * np.maximum(ql - ya, 0) + 10 * np.maximum(ya - qh, 0)).mean())
            res["cov80_exact"] = float(((ql <= ya) & (ya <= qh)).mean())
            res["width80_exact"] = float((qh - ql).mean())
        out[st] = res
    return out


def window_sums(r, data, q=None):
    """Per window and station: sums of MIS80 and CRPS over the observed days and the number of days (for the bootstrap)."""
    out = {}
    ry = data["ry"].numpy().astype(np.float64)
    for s, st in enumerate(STN):
        w, t = np.nonzero(data["mask"][:, s])
        xs, ya = r[w, :, s, t].astype(np.float64), ry[w, s, t]
        lo, hi = np.quantile(xs, [0.1, 0.9], axis=1)
        vals = {"mis80": (hi - lo) + 10 * np.maximum(lo - ya, 0) + 10 * np.maximum(ya - hi, 0), "crps": crps_units(xs, ya), "n": np.ones(len(ya))}
        if q is not None:
            ql, qh = q[w, s, t, 0] - q[w, s, t, 4], q[w, s, t, 8] - q[w, s, t, 4]
            vals["mis80_exact"] = (qh - ql) + 10 * np.maximum(ql - ya, 0) + 10 * np.maximum(ya - qh, 0)
        for k, v in vals.items():
            out[f"{st}__{k}"] = np.bincount(w, weights=v, minlength=r.shape[0])
    return out


def joint_stats(r, data, S=128):
    """Energy Score in ppm (observed days only; each station's 14 days, and both stations' 28) and sample correlations."""
    x = torch.as_tensor(np.ascontiguousarray(r[:, :S]), dtype=torch.float64)
    ry, m = data["ry"].double(), data["m"].double()
    out = {}
    for name, st in (("es_BRW", [0]), ("es_MLO", [1]), ("es_both", [0, 1])):
        mm = m[:, st].reshape(len(m), -1)
        xs = x[:, :, st].reshape(len(x), S, -1) * mm[:, None]
        ys = ry[:, st].reshape(len(x), -1) * mm
        vals = []
        for i in range(0, len(xs), 64):
            a, b = xs[i:i + 64], ys[i:i + 64]
            vals.append(torch.linalg.norm(a - b[:, None], dim=-1).mean(1) - 0.5 * torch.cdist(a, a).sum((1, 2)) / (S * (S - 1)))
        out[name] = float(torch.cat(vals)[mm.sum(1) > 0].mean())
    lag, cross = {st: [] for st in STN}, []
    with np.errstate(invalid="ignore", divide="ignore"):
        for w in range(0, r.shape[0], 7):
            c = np.corrcoef(np.concatenate([r[w, :, 0].T, r[w, :, 1].T]))
            for s, st in enumerate(STN):
                lag[st].append(np.nanmean(np.diag(c[14 * s:14 * s + 14, 14 * s:14 * s + 14], 1)))
            cross.append(np.nanmean(np.diag(c[:14, 14:])))
    out.update({f"lag1_{st}": float(np.nanmean(lag[st])) for st in STN})
    out["cross_station"] = float(np.nanmean(cross))
    return out


def generate(net, data, ids, S, seed, lib=None, chunk=16):
    gen = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), chunk):
            out.append(net.residual(data, ids[i:i + chunk], S, gen, lib).numpy())
    return np.concatenate(out)


def with_ch4(co2_abs, idx, ch4_seed, S):
    """[W,S,4,14]: CO2 samples of the model + CH4 samples from TimesFM's quantiles."""
    return np.concatenate([co2_abs, tf_samples(idx, S, ch4_seed)[:, :, 2:4]], axis=2)


def fit_lstm(sample_fn, tr_idx, tr, va_idx, va, seed):
    """Bidirectional LSTM of bilstm_correct.py fitted on the model's own samples of the training years."""
    A = {}
    for name, idx, data, sd in (("tr", tr_idx, tr, 700_000), ("va", va_idx, va, 800_000)):
        parts = []
        for i in range(0, len(idx), 400):
            ids = np.arange(i, min(i + 400, len(idx)))
            co2 = data["q50"][ids][:, None] + sample_fn(data, ids, 256, sd + i)
            parts.append(bc.arrays(idx[ids], with_ch4(co2, idx[ids], sd + 1 + i, 256)))
        A[name] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return bc.fit("bilstm", bc.sequences(A["tr"]), bc.sequences(A["va"]), seed=seed)


def holdout(sample_fn, ho_idx, ho, lstm, q=None):
    """Three sampling seeds; uncorrected and LSTM-corrected scores on the same samples, averaged."""
    runs = {"none": [], "lstm": [], "joint": [], "w_none": [], "w_lstm": []}
    for sd in SEEDS:
        rs, xcs = [], []
        for i in range(0, len(ho_idx), 400):
            ids = np.arange(i, min(i + 400, len(ho_idx)))
            r = sample_fn(ho, ids, 512, sd + 300_000 + i).astype(np.float64)
            A = bc.arrays(ho_idx[ids], with_ch4(ho["q50"][ids][:, None] + r, ho_idx[ids], sd + 310_000 + i, 512))
            with torch.no_grad():
                mu, ls = lstm(torch.from_numpy(bc.sequences(A)[0]))
            n = len(ids)
            mu, sg = mu.numpy().reshape(n, 2, 14).astype(np.float64), np.exp(ls.numpy()).reshape(n, 2, 14).astype(np.float64)
            xc = np.empty_like(r)
            for s in range(2):
                for t in range(14):
                    xc[:, :, s, t] = bc.shift_correct(r[:, :, s, t], mu[:, s, t], sg[:, s, t])
            rs.append(r)
            xcs.append(xc)
        r, xc = np.concatenate(rs), np.concatenate(xcs)
        runs["none"].append(station_metrics(r, ho, q))
        runs["lstm"].append(station_metrics(xc, ho))
        runs["joint"].append(joint_stats(r, ho))
        runs["w_none"].append(window_sums(r, ho, q))
        runs["w_lstm"].append(window_sums(xc, ho))
        del rs, xcs, r, xc
    avg = lambda rs: {st: {k: float(np.mean([x[st][k] for x in rs])) for k in rs[0][st]} for st in STN}
    windows = {f"{v}__{k}": np.mean([x[k] for x in runs[f"w_{v}"]], 0) for v in ("none", "lstm") for k in runs[f"w_{v}"][0]}
    return avg(runs["none"]), avg(runs["lstm"]), {k: float(np.mean([x[k] for x in runs["joint"]])) for k in runs["joint"][0]}, windows


def tf_residual_fn(idx_of):
    """TimesFM's own distribution as residual samples (sample_fn interface)."""
    return lambda data, ids, S, sd: tf_samples(idx_of[id(data)][ids], S, sd)[:, :, :2] - data["q50"][ids][:, None]


# ---------------------------------------------------------------- jobs
def tfmval():
    OUT.mkdir(parents=True, exist_ok=True)
    va_idx, va, _ = load_split("val", load_split("train")[2])
    fn = tf_residual_fn({id(va): va_idx})
    val = station_metrics(fn(va, np.arange(len(va_idx)), 128, 900_001), va, mg.data()["quantiles"][va_idx][:, :2])
    (OUT / "tfm_val.json").write_text(json.dumps(val, indent=1) + "\n")
    print("DONE tfmval", json.dumps(val), flush=True)


def tfm():
    OUT.mkdir(parents=True, exist_ok=True)
    tr_idx, tr, norm = load_split("train")
    va_idx, va, _ = load_split("val", norm)
    ho_idx, ho, _ = load_split("hold", norm)
    fn = tf_residual_fn({id(tr): tr_idx, id(va): va_idx, id(ho): ho_idx})
    print("STAGE lstm", flush=True)
    lstm, info = fit_lstm(fn, tr_idx, tr, va_idx, va, 2017)
    print("STAGE holdout", flush=True)
    none, corr, joint, windows = holdout(fn, ho_idx, ho, lstm, mg.data()["quantiles"][ho_idx][:, :2])
    np.savez_compressed(OUT / "tfm_windows.npz", origins=mg.data()["origins"][ho_idx], **windows)
    # how strongly the observed CO2 positions (normal scores under TimesFM) go together: adjacent days, and the two stations
    u = bc.arrays(ho_idx, tf_samples(ho_idx, 512, 300_017))["u"]
    obs = {}
    for s, st in enumerate(STN):
        a, b = u[:, s, :-1].ravel(), u[:, s, 1:].ravel()
        ok = ~np.isnan(a) & ~np.isnan(b)
        obs[f"lag1_{st}"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    a, b = u[:, 0].ravel(), u[:, 1].ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    obs["cross_station"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    (OUT / "tfm.json").write_text(json.dumps({"none": none, "lstm": corr, "joint": joint, "observed_score_corr": obs, "lstm_training": info,
                                              "windows": {"train": int(len(tr_idx)), "validation": int(len(va_idx)), "holdout": int(len(ho_idx))}},
                                             indent=1) + "\n")
    print("DONE tfm", json.dumps({st: round(none[st]["mis80_exact"], 3) for st in STN}), json.dumps(obs), flush=True)


def run(arm, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    tr_idx, tr, norm = load_split("train")
    va_idx, va, _ = load_split("val", norm)
    ho_idx, ho, _ = load_split("hold", norm)
    sig = torch.tensor(mg.norm_constants(tr_idx)[:2], dtype=torch.float32)
    lib = None
    if arm == "N4":
        lib = library(tr_idx, tr["ctx_std"])
        for data, idx in ((tr, tr_idx), (va, va_idx), (ho, ho_idx)):
            data["nb"] = neighbours(lib, idx, data["ctx_std"])
    tf_val = json.loads((OUT / "tfm_val.json").read_text())
    net = Net(arm, seed)
    opt_h = torch.optim.AdamW(net.head_params(), lr=2e-4, weight_decay=0.01)
    sp = net.structure_params()
    opt_s = torch.optim.AdamW(sp, lr=1e-3, weight_decay=0.0) if sp else None
    old = arm == "E0"
    S_train = 16 if old else 64
    va_ids = np.arange(len(va_idx))[::7] if old else np.arange(len(va_idx))

    def validate():
        if old:
            gen = torch.Generator().manual_seed(seed + 100000)
            tot = 0.0
            with torch.no_grad():
                for i in range(0, len(va_ids), 16):
                    ids = va_ids[i:i + 16]
                    r = net.residual(va, ids, 256, gen, lib)
                    tot += float(es_loss(r, va["ry"][ids], va["m"][ids], torch.ones(2))) * len(ids)
            return tot / len(va_ids)
        sc = station_metrics(generate(net, va, va_ids, 128, seed + 100000, lib), va)
        return float(np.mean([(sc[k]["crps"] / tf_val[k]["crps"] + sc[k]["mis80"] / tf_val[k]["mis80_exact"]) / 2 for k in STN]))

    max_epochs = 2 if SMOKE else 80
    best = validate()
    best_state, best_epoch, bad, t0 = {k: v.clone() for k, v in net.state_dict().items()}, 0, 0, time.time()
    print(f"{arm} s{seed} ep 0 val {best:.4f} best 0 (0s)", flush=True)
    for epoch in range(1, max_epochs + 1):
        net.train()
        order = np.random.default_rng(seed + epoch).permutation(len(tr_idx))
        gen = torch.Generator().manual_seed(seed * 10000 + epoch)
        for i in range(0, len(order), 32):
            ids = order[i:i + 32]
            r = net.residual(tr, ids, S_train, gen, lib)
            ry, m = tr["ry"][torch.as_tensor(ids)], tr["m"][torch.as_tensor(ids)]
            loss = es_loss(r, ry, m, torch.ones(2)) if old else crps_loss(r, ry, m, sig) + 0.5 * es_loss(r, ry, m, sig)
            opt_h.zero_grad()
            if opt_s:
                opt_s.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.head_params(), 1.0)
            opt_h.step()
            if opt_s:
                torch.nn.utils.clip_grad_norm_(sp, 1.0)
                opt_s.step()
        net.eval()
        val = validate()
        if val < best - 1e-9:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"{arm} s{seed} ep {epoch} val {val:.4f} best {best_epoch} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= 12:
            break
    net.load_state_dict(best_state)
    seconds = time.time() - t0
    (OUT / "models").mkdir(exist_ok=True)
    torch.save(best_state, OUT / "models" / f"{arm}_s{seed}.pt")
    learned = {}
    if arm in ("N1", "N2", "N3", "N4", "T1"):
        with torch.no_grad():
            p = net.structure(ho["c"]).numpy()
        month = mg.data()["forecast_dates"][ho_idx][:, 0].astype("datetime64[M]").astype(int) % 12 + 1
        season = np.array([{12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏", 7: "夏", 8: "夏"}.get(int(m), "秋") for m in month])
        learned = {st: {"all": float(p[:, s].mean()), **{se: float(p[season == se, s].mean()) for se in ("冬", "春", "夏", "秋")},
                        "p10": float(np.quantile(p[:, s], 0.1)), "p90": float(np.quantile(p[:, s], 0.9)),
                        "corr_with_context": {k: float(np.corrcoef(p[:, s], ho["ctx_std"][:, s, j])[0, 1]) if p[:, s].std() > 0 else 0.0
                                              for j, k in enumerate(("sin", "cos", "log_width", "log_scale", "log_vol"))}}
                   for s, st in enumerate(STN)}
    if arm == "N5":
        tail, skew = torch.exp(net.logtail).detach(), net.skew.detach()
        learned = {"skew_abs_mean": float(skew.abs().mean()), "skew_first_block": float(skew[0]), "tail_mean": float(tail.mean()),
                   "tail_first_block": float(tail[0]), "tail_min": float(tail.min()), "tail_max": float(tail.max())}
    fn = lambda data, ids, S, sd: generate(net, data, ids, S, sd, lib)
    print("STAGE lstm", flush=True)
    lstm, info = fit_lstm(fn, tr_idx, tr, va_idx, va, seed)
    torch.save(lstm.state_dict(), OUT / "models" / f"{arm}_s{seed}_lstm.pt")
    print("STAGE holdout", flush=True)
    none, corr, joint, windows = holdout(fn, ho_idx, ho, lstm)
    np.savez_compressed(OUT / f"{arm}_s{seed}_windows.npz", origins=mg.data()["origins"][ho_idx], **windows)
    res = {"arm": arm, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch, "best_val": best, "seconds": seconds,
           "none": none, "lstm": corr, "joint": joint, "lstm_training": info, "learned": learned}
    (OUT / f"{arm}_s{seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(f"DONE {arm} s{seed}: holdout MIS80 BRW {none['BRW']['mis80']:.3f} MLO {none['MLO']['mis80']:.3f} | "
          f"+LSTM BRW {corr['BRW']['mis80']:.3f} MLO {corr['MLO']['mis80']:.3f}", flush=True)


def differences():
    """Paired differences with 95% intervals (calendar-month blocks of 2010-2013 resampled; per-window scores averaged
    over the three training seeds, so the interval shows the spread over windows, not over training seeds)."""
    from mg.summary3 import Bootstrap
    load = lambda name: dict(np.load(OUT / f"{name}_windows.npz")) if (OUT / f"{name}_windows.npz").exists() else None
    tw = load("tfm")
    arms = {}
    for a in ARMS:
        parts = [load(f"{a}_s{s}") for s in SEEDS]
        if all(p is not None for p in parts):
            arms[a] = {k: np.mean([p[k] for p in parts], 0) for k in parts[0] if k != "origins"}
    if tw is None or not arms:
        return ["", "（每个窗口的分数还没齐，差值和区间暂缺。）"]
    boot = Bootstrap(tw["origins"])
    pair = lambda W, v, st, m: (W[f"{v}__{st}__{m}"], W[f"{v}__{st}__n"])

    def fmt(d):
        lo, hi = d["ci95"]
        return f"{d['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if hi < 0 else " †" if lo > 0 else "")
    L = ["", "## 1b. 差值和 95% 区间", "",
         "负数表示比参照好。* 表示显著更好，† 表示显著更差（区间不含 0）。区间按月分块重抽 2010–13 年的窗口得到，三个种子的分数先平均，"
         "所以它反映的是窗口的波动，不含种子之间的差别（种子之间的差别看上表括号）。", ""]
    for metric, mname in (("mis80", "MIS80"), ("crps", "CRPS")):
        L += [f"**和 TimesFM 比（{mname}）**：不修正的减 TimesFM；加 LSTM 的减 TimesFM + LSTM。", "",
              "| 方法 | BRW 不修正 | MLO 不修正 | BRW + LSTM | MLO + LSTM |", "|---|---:|---:|---:|---:|"]
        for a, W in arms.items():
            ref = "mis80_exact" if metric == "mis80" else "crps"
            cells = [fmt(boot.diff(pair(W, "none", st, metric), pair(tw, "none", st, ref))) for st in STN]
            cells += [fmt(boot.diff(pair(W, "lstm", st, metric), pair(tw, "lstm", st, metric))) for st in STN]
            L.append(f"| {a} {NAMES[a]} | " + " | ".join(cells) + " |")
        L.append("")
        if "N0" in arms:
            L += [f"**和 N0（普通噪声、同样的训练目标）比（{mname}）**：看噪声的拼接方式本身有没有用。", "",
                  "| 方法 | BRW 不修正 | MLO 不修正 | BRW + LSTM | MLO + LSTM |", "|---|---:|---:|---:|---:|"]
            for a, W in arms.items():
                if a not in ("E0", "N0", "T1"):
                    L.append(f"| {a} {NAMES[a]} | " + " | ".join(fmt(boot.diff(pair(W, v, st, metric), pair(arms["N0"], v, st, metric)))
                                                            for v in ("none", "lstm") for st in STN) + " |")
            L.append("")
        if "N1" in arms and "T1" in arms:
            L += [f"**N1 减 T1（{mname}）**：同样的大小规则，用 Engression 的样本还是直接用 TimesFM 的分布。", "",
                  "| | BRW 不修正 | MLO 不修正 | BRW + LSTM | MLO + LSTM |", "|---|---:|---:|---:|---:|",
                  "| N1 减 T1 | " + " | ".join(fmt(boot.diff(pair(arms["N1"], v, st, metric), pair(arms["T1"], v, st, metric)))
                                                  for v in ("none", "lstm") for st in STN) + " |", ""]
    return L


def report():
    tf = json.loads((OUT / "tfm.json").read_text())
    rows = {a: [json.loads((OUT / f"{a}_s{s}.json").read_text()) for s in SEEDS if (OUT / f"{a}_s{s}.json").exists()] for a in ARMS}
    mean = lambda rs, part, st, k: float(np.mean([r[part][st][k] for r in rs]))
    sd = lambda rs, part, st, k: float(np.std([r[part][st][k] for r in rs]))
    jm = lambda rs, k: float(np.mean([r["joint"][k] for r in rs]))
    tn, tl, tj = tf["none"], tf["lstm"], tf["joint"]
    L = ["# 噪声拼接方法的预试（2010–2013 年评估，两站，CO₂）", "",
         "自动生成（`python noise_study.py report`）。训练 2000–08，按 2009 选轮次，2010–13 评估；没有用到 2014–2025 的任何测试年。"
         "单位 ppm，越低越好（覆盖率越接近 80% 越好）；三个训练种子平均，括号里是种子间的标准差。TimesFM 的 MIS80、覆盖率、宽度用精确分位数。", "",
         "## 1. 不修正：Engression 本身", "",
         "| 方法 | BRW MIS80 | MLO MIS80 | BRW CRPS | MLO CRPS | BRW 覆盖率 | MLO 覆盖率 | BRW 宽度 | MLO 宽度 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
         f"| TimesFM | {tn['BRW']['mis80_exact']:.3f} | {tn['MLO']['mis80_exact']:.3f} | {tn['BRW']['crps']:.3f} | {tn['MLO']['crps']:.3f} | "
         f"{100 * tn['BRW']['cov80_exact']:.1f}% | {100 * tn['MLO']['cov80_exact']:.1f}% | {tn['BRW']['width80_exact']:.3f} | {tn['MLO']['width80_exact']:.3f} |"]
    for a in ARMS:
        rs = rows[a]
        if rs:
            L.append(f"| {a} {NAMES[a]} | {mean(rs, 'none', 'BRW', 'mis80'):.3f} ({sd(rs, 'none', 'BRW', 'mis80'):.3f}) | "
                     f"{mean(rs, 'none', 'MLO', 'mis80'):.3f} ({sd(rs, 'none', 'MLO', 'mis80'):.3f}) | "
                     f"{mean(rs, 'none', 'BRW', 'crps'):.3f} | {mean(rs, 'none', 'MLO', 'crps'):.3f} | {100 * mean(rs, 'none', 'BRW', 'cov80'):.1f}% | "
                     f"{100 * mean(rs, 'none', 'MLO', 'cov80'):.1f}% | {mean(rs, 'none', 'BRW', 'width80'):.3f} | {mean(rs, 'none', 'MLO', 'width80'):.3f} |")
    L += ["", "## 2. 加上双向 LSTM 修正（同一批样本；CH₄ 的位置用 TimesFM 的分布算）", "",
          "| 方法 | BRW MIS80 | MLO MIS80 | BRW CRPS | MLO CRPS | BRW 覆盖率 | MLO 覆盖率 | BRW MAE | MLO MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
          f"| TimesFM + LSTM | {tl['BRW']['mis80']:.3f} | {tl['MLO']['mis80']:.3f} | {tl['BRW']['crps']:.3f} | {tl['MLO']['crps']:.3f} | "
          f"{100 * tl['BRW']['cov80']:.1f}% | {100 * tl['MLO']['cov80']:.1f}% | {tl['BRW']['mae']:.3f} | {tl['MLO']['mae']:.3f} |"]
    for a in ARMS:
        rs = rows[a]
        if rs:
            L.append(f"| {a} {NAMES[a]} + LSTM | {mean(rs, 'lstm', 'BRW', 'mis80'):.3f} ({sd(rs, 'lstm', 'BRW', 'mis80'):.3f}) | "
                     f"{mean(rs, 'lstm', 'MLO', 'mis80'):.3f} ({sd(rs, 'lstm', 'MLO', 'mis80'):.3f}) | "
                     f"{mean(rs, 'lstm', 'BRW', 'crps'):.3f} | {mean(rs, 'lstm', 'MLO', 'crps'):.3f} | {100 * mean(rs, 'lstm', 'BRW', 'cov80'):.1f}% | "
                     f"{100 * mean(rs, 'lstm', 'MLO', 'cov80'):.1f}% | {mean(rs, 'lstm', 'BRW', 'mae'):.3f} | {mean(rs, 'lstm', 'MLO', 'mae'):.3f} |")
    L += differences()
    o = tf["observed_score_corr"]
    L += ["", "## 3. 多天、两站一起看（不修正的样本）", "",
          "Energy Score（ppm，越低越好）：BRW、MLO 各自 14 天一起评，以及两站 28 个数一起评。相关：同一窗口里样本在相邻两天、两站之间的相关。"
          f"实际观测值（在 TimesFM 分布里的位置）的相关作参考：相邻两天 BRW {o['lag1_BRW']:.2f}、MLO {o['lag1_MLO']:.2f}，两站之间 {o['cross_station']:.2f}。", "",
          "| 方法 | ES BRW 14 天 | ES MLO 14 天 | ES 两站 | 相邻两天相关 BRW | 相邻两天相关 MLO | 两站相关 |", "|---|---:|---:|---:|---:|---:|---:|",
          f"| TimesFM（各天独立抽样） | {tj['es_BRW']:.3f} | {tj['es_MLO']:.3f} | {tj['es_both']:.3f} | {tj['lag1_BRW']:.2f} | {tj['lag1_MLO']:.2f} | {tj['cross_station']:.2f} |"]
    for a in ARMS:
        rs = rows[a]
        if rs:
            L.append(f"| {a} {NAMES[a]} | {jm(rs, 'es_BRW'):.3f} | {jm(rs, 'es_MLO'):.3f} | {jm(rs, 'es_both'):.3f} | {jm(rs, 'lag1_BRW'):.2f} | "
                     f"{jm(rs, 'lag1_MLO'):.2f} | {jm(rs, 'cross_station'):.2f} |")
    L += ["", "## 4. 学到的规律（2010–13 窗口上的平均值）", ""]
    labels = {"N1": "噪声放大倍数的对数 γ（0 = 不变，+0.1 约放大 10%）", "N2": "两站共享的比例 α", "N3": "相邻两天的相关 ρ", "N4": "历史误差的比例 λ",
              "T1": "TimesFM 区间放大倍数的对数 γ"}
    for a in ("N1", "N2", "N3", "N4", "T1"):
        for st in STN:
            if rows[a]:
                v = {k: np.mean([r["learned"][st][k] for r in rows[a]]) for k in ("all", "冬", "春", "夏", "秋", "p10", "p90")}
                cw = {k: np.mean([r["learned"][st]["corr_with_context"][k] for r in rows[a]]) for k in rows[a][0]["learned"][st]["corr_with_context"]}
                top = max(cw, key=lambda k: abs(cw[k]))
                ctx_name = {"sin": "季节（sin）", "cos": "季节（cos）", "log_width": "TimesFM 的区间宽度", "log_scale": "最近的数值尺度",
                            "log_vol": "过去 14 天的日波动"}[top]
                L.append(f"- {a} {labels[a]}，{st}：全部 {v['all']:+.2f}；冬 {v['冬']:+.2f}，春 {v['春']:+.2f}，夏 {v['夏']:+.2f}，秋 {v['秋']:+.2f}；"
                         f"10%–90% 范围 {v['p10']:+.2f} 到 {v['p90']:+.2f}；和它关系最大的输入是{ctx_name}（相关 {cw[top]:+.2f}）")
    if rows["N5"]:
        v = {k: np.mean([r["learned"][k] for r in rows["N5"]]) for k in rows["N5"][0]["learned"]}
        L.append(f"- N5 噪声形状：管前 14 天的那一组，偏斜 {v['skew_first_block']:+.3f}、尾部 {v['tail_first_block']:.3f}（1 = 正态，大于 1 尾巴更重）；"
                 f"全部组平均偏斜绝对值 {v['skew_abs_mean']:.3f}，尾部 {v['tail_min']:.2f}–{v['tail_max']:.2f}")
    L += ["", "## 5. 训练情况", "", "| 方法 | 选中的轮次 | 跑了几轮 | 训练用时（分钟） |", "|---|---|---|---:|"]
    for a in ARMS:
        if rows[a]:
            L.append(f"| {a} | {', '.join(str(r['best_epoch']) for r in rows[a])} | {', '.join(str(r['epochs_run']) for r in rows[a])} | "
                     f"{np.mean([r['seconds'] for r in rows[a]]) / 60:.1f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    (OUT / "report.json").write_text(json.dumps({"tfm": tf, "runs": rows}, ensure_ascii=False, indent=1) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"tfmval": tfmval, "tfm": tfm, "run": lambda: run(a[1], int(a[2])), "report": report}[a[0]]()
