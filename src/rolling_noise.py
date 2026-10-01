"""Large-sample test (two stations) of the two best schemes of the noise study (user request 2026-10-01, "前两名大数据跑一下";
settings fixed before any large-sample result of them was seen, all copied unchanged from src/noise_study.py):

  N1 : residual Engression (CO2, locked median, per-station noise) whose Gaussian noise is multiplied by exp(gamma(c)),
       gamma from a small network of the context c (season, TimesFM 80% width, RevIN scale, history volatility, station)
  T1 : TimesFM's own distribution with the same learned rule, residual sample = exp(gamma(c)) * (TimesFM sample - median)
  each without correction and with the bidirectional LSTM correction (CH4 positions from TimesFM's quantiles)

Per test year (2020-2025: training from 2010; confirmation 2014-2019: training from 2000) and seed (17, 29, 43), as
mg.fold: training windows up to year-2, validation year-1 (all daily windows, selection rule of the noise study: mean
over stations of (CRPS / CRPS_TimesFM + MIS80 / MIS80_TimesFM) / 2), test year. Context standardized on the fold's
training windows; loss normalization mg.norm_constants of the training windows; objective per-cell fair CRPS + 0.5 *
Energy Score (station-normalized), 64 training samples, AdamW (head lr 2e-4 decay 0.01; rule lr 1e-3), batch 32, at most
80 epochs, patience 12. The LSTM is fitted per fold and seed on the model's own samples of the training years (256 per
window), validation year for early stopping (bilstm_correct.fit, unchanged).
Test: 512 samples per seed (seed + 300000), cells scored as rolling_residual.fold (conditional_co2.scores): cells with
both gases observed (compared with TimesFM + CH4 covariate) and cells whose same-day CH4 is missing. Scores of the three
seeds are averaged per window. References from the earlier large-sample run (results/rolling_corrections/tfm_fold*.npz):
TimesFM without correction and TimesFM + LSTM.

Usage: python src/rolling_noise.py run N1|T1 YEAR SEED | report [confirm]       (NOISE_SMOKE=1: tiny quick check)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

NEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW / "src"))
import noise_study as ns                                                 # noqa: E402  (model, losses, scores of the noise study)
import bilstm_correct as bc                                              # noqa: E402
from residual_rnet import enc, GCOPY                                     # noqa: E402
from timesfm_rnet import tf_samples                                      # noqa: E402
from mg.conditional_co2 import scores                                    # noqa: E402
from mg.summary3 import Bootstrap                                        # noqa: E402

mg = ns.mg
SMOKE = os.environ.get("NOISE_SMOKE") == "1"
OUT = NEW / "results" / ("rolling_noise_smoke" if SMOKE else "rolling_noise")
MODELS = NEW / "training" / ("rolling_noise_smoke" if SMOKE else "rolling_noise")
REPORT = NEW / "results" / "reports"
SEEDS = (17, 29, 43)
LABELS = ("BRW_CO2", "MLO_CO2")
STN = ns.STN
METRICS = ("coverage80", "width80", "mis80", "crps", "abs_error", "n")


def load_idx(idx, norm=None):
    """As noise_study.load_split, for given windows."""
    d = mg.data()
    e = enc(idx, [0, 1])
    rc = ns.raw_context(idx)
    if norm is None:
        flat = rc.reshape(-1, rc.shape[-1])
        norm = (flat.mean(0), flat.std(0) + 1e-9)
    cs = (rc - norm[0]) / norm[1]
    c = np.concatenate([cs, np.broadcast_to(np.eye(2)[None], (len(idx), 2, 2))], -1)
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    y, m = d["y"][idx][:, :2], d["mask"][idx][:, :2].astype(bool)
    return {"h": t(e["features"]), "scale": t(e["scale"]), "c": t(c), "ry": t(np.where(m, y - e["q50"], 0.0)), "m": torch.tensor(m),
            "q50": e["q50"], "mask": m, "ctx_std": cs, "q": torch.as_tensor(np.ascontiguousarray(d["quantiles"][idx][:, :2]))}, norm


def run(arm, year, seed):
    d = mg.data()
    f = mg.fold(year)
    tr_idx, va_idx, te_idx = f["train"], f["validation"], f["test"]
    if SMOKE:
        tr_idx, va_idx = tr_idx[::25], va_idx[::5]
    tr, norm = load_idx(tr_idx)
    va, _ = load_idx(va_idx, norm)
    te, _ = load_idx(te_idx, norm)
    sig = torch.tensor(mg.norm_constants(tr_idx)[:2], dtype=torch.float32)
    # TimesFM on the validation year: reference of the selection rule (as noise_study.tfmval)
    tf_val = ns.station_metrics(tf_samples(va_idx, 128, 900_001)[:, :, :2] - va["q50"][:, None], va, d["quantiles"][va_idx][:, :2])
    net = ns.Net(arm, seed)
    opt_h = torch.optim.AdamW(net.head_params(), lr=2e-4, weight_decay=0.01)
    sp = net.structure_params()
    opt_s = torch.optim.AdamW(sp, lr=1e-3, weight_decay=0.0)
    va_ids = np.arange(len(va_idx))

    def validate():
        sc = ns.station_metrics(ns.generate(net, va, va_ids, 128, seed + 100000), va)
        return float(np.mean([(sc[k]["crps"] / tf_val[k]["crps"] + sc[k]["mis80"] / tf_val[k]["mis80_exact"]) / 2 for k in STN]))

    best = validate()
    best_state, best_epoch, bad, t0 = {k: v.clone() for k, v in net.state_dict().items()}, 0, 0, time.time()
    print(f"{arm} {year} s{seed} ep 0 val {best:.4f} best 0 (0s)", flush=True)
    for epoch in range(1, (2 if SMOKE else 80) + 1):
        net.train()
        order = np.random.default_rng(seed + epoch).permutation(len(tr_idx))
        gen = torch.Generator().manual_seed(seed * 10000 + epoch)
        for i in range(0, len(order), 32):
            ids = order[i:i + 32]
            r = net.residual(tr, ids, 64, gen)
            ry, m = tr["ry"][torch.as_tensor(ids)], tr["m"][torch.as_tensor(ids)]
            loss = ns.crps_loss(r, ry, m, sig) + 0.5 * ns.es_loss(r, ry, m, sig)
            opt_h.zero_grad()
            opt_s.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.head_params(), 1.0)
            opt_h.step()
            torch.nn.utils.clip_grad_norm_(sp, 1.0)
            opt_s.step()
        net.eval()
        val = validate()
        if val < best - 1e-9:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"{arm} {year} s{seed} ep {epoch} val {val:.4f} best {best_epoch} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= 12:
            break
    net.load_state_dict(best_state)
    seconds = time.time() - t0
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS / f"{arm}_fold{year}_seed{seed}.pt")
    with torch.no_grad():
        gamma = net.structure(te["c"]).numpy()
    months = d["months"][te_idx]
    season = np.array([mg_season(m) for m in months])
    learned = {st: {"all": float(gamma[:, s].mean()), **{se: float(gamma[season == se, s].mean()) if (season == se).any() else None
                                                         for se in ("DJF", "MAM", "JJA", "SON")}} for s, st in enumerate(STN)}
    fn = lambda data, ids, S, sd: ns.generate(net, data, ids, S, sd)
    print("STAGE lstm", flush=True)
    lstm, info = ns.fit_lstm(fn, tr_idx, tr, va_idx, va, seed)
    torch.save(lstm.state_dict(), MODELS / f"{arm}_fold{year}_seed{seed}_lstm.pt")
    print("STAGE test", flush=True)
    r = ns.generate(net, te, np.arange(len(te_idx)), 512, seed + mg.OFFSETS["test"]).astype(np.float64)
    x = np.concatenate([te["q50"][:, None] + r, tf_samples(te_idx, 512, seed + mg.OFFSETS["test"] + 1)[:, :, 2:4]], axis=2)
    A = bc.arrays(te_idx, x)
    with torch.no_grad():
        mu, ls = lstm(torch.from_numpy(bc.sequences(A)[0]))
    mu_l, sg_l = mu.numpy().reshape(len(te_idx), 2, 14).astype(np.float64), np.exp(ls.numpy()).reshape(len(te_idx), 2, 14).astype(np.float64)
    y, mask = d["y"][te_idx], d["mask"][te_idx].astype(bool)
    sums = {}
    for s, label in enumerate(LABELS):
        for kind, sel in (("both", mask[:, s] & mask[:, s + 2]), ("co2only", mask[:, s] & ~mask[:, s + 2])):
            w, t = np.nonzero(sel)
            xa, ya = x[w, :, s, t], y[w, s, t]
            for v, xs in (("none", xa), ("lstm", bc.shift_correct(xa, mu_l[w, s, t], sg_l[w, s, t]))):
                for m, val in scores(xs, ya).items():
                    a = np.zeros(len(te_idx))
                    np.add.at(a, w, val)
                    sums[f"{label}__{kind}__{v}__{m}"] = a
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{arm}_fold{year}_seed{seed}.npz", origins=d["origins"][te_idx], months=months, **sums)
    (OUT / f"{arm}_fold{year}_seed{seed}.json").write_text(json.dumps({"arm": arm, "fold": year, "seed": seed, "best_epoch": best_epoch,
        "epochs_run": epoch, "best_val": best, "seconds": seconds, "windows": {"train": int(len(tr_idx)), "validation": int(len(va_idx)),
        "test": int(len(te_idx))}, "lstm_training": info, "learned_gamma_test": learned}, indent=1) + "\n")
    mis = lambda v, lab: sums[f"{lab}__both__{v}__mis80"].sum() / sums[f"{lab}__both__{v}__n"].sum()
    print(f"DONE {arm} {year} s{seed}: MIS80 (both gases observed) BRW {mis('none', 'BRW_CO2'):.3f} MLO {mis('none', 'MLO_CO2'):.3f} | "
          f"+LSTM BRW {mis('lstm', 'BRW_CO2'):.3f} MLO {mis('lstm', 'MLO_CO2'):.3f}", flush=True)


def mg_season(month):
    from mg.conditional_co2 import SEASON
    return SEASON[int(month)]


def report(tag, folds=None):
    folds = folds or (mg.CONFIRM_FOLDS if tag else mg.FOLDS)
    d = mg.data()
    parts = [dict(np.load(NEW / "results" / "rolling_corrections" / f"tfm_fold{y}.npz")) for y in folds]
    data = {"tfm": {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}}
    for arm in ("N1", "T1"):
        per_fold = []
        for y in folds:
            runs = [dict(np.load(OUT / f"{arm}_fold{y}_seed{s}.npz")) for s in SEEDS]
            per_fold.append({k: runs[0][k] if k in ("origins", "months") else np.mean([r[k] for r in runs], 0) for k in runs[0]})
        data[arm] = {k: np.concatenate([p[k] for p in per_fold]) for k in per_fold[0]}
        assert np.array_equal(data[arm]["origins"], data["tfm"]["origins"])
    boot = Bootstrap(data["tfm"]["origins"])
    tf = np.load(GCOPY / "06_结果" / "timesfm_covariates" / f"past_future{tag}.npz")
    roll = np.concatenate([mg.fold(y)["test"] for y in folds])
    where = {r: i for i, r in enumerate(tf["rows"])}
    q = tf["quantiles"][[where[r] for r in roll]]
    y, mask = d["y"][roll], d["mask"][roll].astype(bool)
    rows = [("tfm", "none", "TimesFM 不修正"), ("tfm", "bilstm", "TimesFM + LSTM"), ("T1", "none", "T1 TimesFM + 随情况变的宽度"),
            ("T1", "lstm", "T1 + LSTM"), ("N1", "none", "N1 Engression + 噪声大小随情况变"), ("N1", "lstm", "N1 + LSTM")]
    pairs = [(("N1", "lstm"), ("tfm", "bilstm"), "N1 + LSTM 减 TimesFM + LSTM"), (("T1", "lstm"), ("tfm", "bilstm"), "T1 + LSTM 减 TimesFM + LSTM"),
             (("N1", "lstm"), ("T1", "lstm"), "N1 + LSTM 减 T1 + LSTM"), (("N1", "none"), ("tfm", "none"), "N1 减 TimesFM（都不修正）"),
             (("T1", "none"), ("tfm", "none"), "T1 减 TimesFM（都不修正）"), (("N1", "none"), ("T1", "none"), "N1 减 T1（都不修正）")]
    fmt = lambda dd: f"{dd['difference']:+.3f} [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]" + (" *" if dd["ci95"][1] < 0 else " †" if dd["ci95"][0] > 0 else "")
    result = {}
    period = "2014–2019 确认" if tag else "2020–2025 滚动检验"
    L = [f"# 大样本（两站）：噪声拼接预试的前两名（{period}）", "",
         "自动生成（`python src/rolling_noise.py report`）。N1、T1 的设置和预试完全相同，在看到大样本结果之前定好；每个测试年、每个种子重新训练，三个种子的分数按窗口平均。"
         "TimesFM 不修正和 TimesFM + LSTM 来自之前的大样本结果。单位 ppm，越低越好。差值后的方括号是 95% 区间（按月分块的自助法），"
         "* 表示显著更好，† 表示显著更差。", ""]
    for k, label in enumerate(LABELS):
        both = mask[:, k] & mask[:, k + 2]
        lo, hi = q[:, k, :, 0], q[:, k, :, 8]
        yy = np.nan_to_num(y[:, k])
        tf_mis = np.where(both, (hi - lo) + 10 * np.maximum(lo - yy, 0) + 10 * np.maximum(yy - hi, 0), 0).sum(1)
        tf_cov = np.where(both, (yy >= lo) & (yy <= hi), 0).sum(1)
        tf_n = both.sum(1).astype(float)
        assert np.allclose(tf_n, data["tfm"][f"{label}__both__none__n"])
        g = lambda base, v, m, kind="both": data[base][f"{label}__{kind}__{v}__{m}"]
        res = {"timesfm_ch4": {"mis80": float(tf_mis.sum() / tf_n.sum()), "coverage80": float(tf_cov.sum() / tf_n.sum())}}
        L += [f"## {label}（两种气体当天都有观测的格子）", "",
              "| 方法 | MIS80 | 减 TimesFM + CH₄ | 覆盖率 | 区间宽度 | CRPS | MAE |", "|---|---:|---:|---:|---:|---:|---:|",
              f"| TimesFM + CH₄ 协变量 | {res['timesfm_ch4']['mis80']:.3f} | — | {100 * res['timesfm_ch4']['coverage80']:.1f}% | — | — | — |"]
        for base, v, name in rows:
            n = g(base, v, "n").sum()
            r = {m: float(g(base, v, m).sum() / n) for m in METRICS[:-1]}
            r["minus_timesfm_ch4"] = boot.diff((g(base, v, "mis80"), g(base, v, "n")), (tf_mis, tf_n))
            n2 = g(base, v, "n", "co2only").sum()
            r["co2only_cells"] = {"n": int(n2), **{m: float(g(base, v, m, "co2only").sum() / n2) for m in METRICS[:-1]}}
            res[name] = r
            dd = r["minus_timesfm_ch4"]
            L.append(f"| {name} | {r['mis80']:.3f} | {fmt(dd)} | {100 * r['coverage80']:.1f}% | {r['width80']:.3f} | {r['crps']:.3f} | {r['abs_error']:.3f} |")
        L += ["", "两两比较（MIS80 差，同样的格子）：", "", "| 比较 | 两种气体都有观测的格子 | 当天没测到 CH₄ 的格子 |", "|---|---:|---:|"]
        res["pairs"] = {}
        for (b1, v1), (b2, v2), name in pairs:
            dd = {kind: boot.diff((g(b1, v1, "mis80", kind), g(b1, v1, "n", kind)), (g(b2, v2, "mis80", kind), g(b2, v2, "n", kind))) for kind in ("both", "co2only")}
            res["pairs"][name] = dd
            L.append(f"| {name} | {fmt(dd['both'])} | {fmt(dd['co2only'])} |")
        L += ["", "当天没测到 CH₄ 的格子：", "", "| 方法 | 格子数 | MIS80 | 覆盖率 | CRPS | MAE |", "|---|---:|---:|---:|---:|---:|"]
        for _, _, name in rows:
            x = res[name]["co2only_cells"]
            L.append(f"| {name} | {x['n']} | {x['mis80']:.3f} | {100 * x['coverage80']:.1f}% | {x['crps']:.3f} | {x['abs_error']:.3f} |")
        L.append("")
        result[label] = res
    learned = {arm: [json.loads((OUT / f"{arm}_fold{y}_seed{s}.json").read_text()) for y in folds for s in SEEDS] for arm in ("N1", "T1")}
    L += ["## 学到的规律和训练情况", ""]
    for arm in ("N1", "T1"):
        js = learned[arm]
        for st in STN:
            vals = {k: [j["learned_gamma_test"][st][k] for j in js if j["learned_gamma_test"][st][k] is not None] for k in ("DJF", "MAM", "JJA", "SON")}
            se = {k: f"{np.mean(v):+.2f}" if v else "—" for k, v in vals.items()}
            L.append(f"- {arm} 的 γ（{st}，测试窗口平均）：冬 {se['DJF']}，春 {se['MAM']}，夏 {se['JJA']}，秋 {se['SON']}"
                     + ("（T1 里 exp(γ) 就是区间放大倍数）" if arm == "T1" else ""))
        L.append(f"- {arm} 选中的轮次：{', '.join(str(j['best_epoch']) for j in js)}；平均训练 {np.mean([j['seconds'] for j in js]) / 60:.1f} 分钟")
    REPORT.mkdir(exist_ok=True)
    name = f"large_noise{tag}{'_smoke' if SMOKE else ''}"
    (REPORT / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    (REPORT / f"{name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        run(a[1], int(a[2]), int(a[3]))
    else:
        tag = "_confirm" if "confirm" in a[1:] else ""
        only = [int(x) for x in a[1:] if x.isdigit()]
        report(tag, tuple(only) if only else None)
