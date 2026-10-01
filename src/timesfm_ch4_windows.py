"""Per-window scores of TimesFM + CH4 covariate (two stations, quantile output) on the large-sample test windows, cells
with both gases observed, computed exactly as in rolling_residual.report (MIS80, coverage from the 10% / 90% quantiles)
plus the interval width and the error of the median (quantile 50%). Saved so that tables can be rebuilt without
multigas.npz (e.g. in the 2to2 repository).

Usage: python src/timesfm_ch4_windows.py      -> results/timesfm_ch4/windows.npz, windows_confirm.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_rnet import mg, GCOPY, NEW                                         # noqa: E402

OUT = NEW / "results" / "timesfm_ch4"
LABELS = ("BRW_CO2", "MLO_CO2")


def main():
    d = mg.data()
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, folds in (("", mg.FOLDS), ("_confirm", mg.CONFIRM_FOLDS)):
        tf = np.load(GCOPY / "06_结果" / "timesfm_covariates" / f"past_future{tag}.npz")
        roll = np.concatenate([mg.fold(y)["test"] for y in folds])
        where = {r: i for i, r in enumerate(tf["rows"])}
        q = tf["quantiles"][[where[r] for r in roll]]
        y, mask = d["y"][roll], d["mask"][roll].astype(bool)
        sums = {}
        for k, label in enumerate(LABELS):
            both = mask[:, k] & mask[:, k + 2]
            lo, mid, hi = q[:, k, :, 0], q[:, k, :, 4], q[:, k, :, 8]
            yy = np.nan_to_num(y[:, k])
            cell = {"mis80": (hi - lo) + 10 * np.maximum(lo - yy, 0) + 10 * np.maximum(yy - hi, 0), "coverage80": (yy >= lo) & (yy <= hi),
                    "width80": hi - lo, "abs_error": np.abs(mid - yy), "n": np.ones_like(yy)}
            for m, v in cell.items():
                sums[f"{label}__both__{m}"] = np.where(both, v, 0).sum(1).astype(np.float64)
        np.savez_compressed(OUT / f"windows{tag}.npz", origins=d["origins"][roll], months=d["months"][roll], **sums)
        print(tag or "main", {lab: round(sums[f"{lab}__both__mis80"].sum() / sums[f"{lab}__both__n"].sum(), 3) for lab in LABELS})


if __name__ == "__main__":
    main()
