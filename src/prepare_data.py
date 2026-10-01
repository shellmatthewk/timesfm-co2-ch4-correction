"""All inputs of the ten schemes from the raw NOAA files (needs the TimesFM 3 weights in timesfm_base/models/timesfm3):

  1. daily panel 1990-2025 and the daily sliding windows (past 14 days -> next 14 days, forecast targets 2000-2025) of
     CO2 (BRW, MLO, SMO, SPO: the window rules use all four stations) and CH4 (BRW, MLO)        (mg.data)
  2. TimesFM features and quantiles: CO2 with only BRW and MLO as input ("two stations predict two stations"), CH4
     with BRW and MLO as input                                                                  (mg.encode)
  3. multigas/02_数据/multigas.npz, the input of every scheme                                    (mg.multigas.build)

The original runs encoded on Apple MPS in batches of 8; on MPS the encodings are reproduced bit for bit, on other
devices a few windows can differ (by up to a few ppm in the quantiles).

Usage: python src/prepare_data.py [--device mps|cuda|cpu]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "multigas"
sys.path.insert(0, str(G / "src"))
from mg.encode import encode                           # noqa: E402  (first: puts timesfm_base/src with the TimesFM code on the path)
from mg.data import build_panel, build_windows         # noqa: E402


def main(device):
    (G / "02_数据").mkdir(exist_ok=True)
    (G / "03_模型/encoded").mkdir(parents=True, exist_ok=True)
    dates, values, _ = build_panel()
    for group in ("co2", "ch4"):
        t0 = time.time()
        w, reasons = build_windows(dates, values, group, "2000-01-01", "2025-12-31", 1)
        np.savez_compressed(G / "02_数据" / f"daily_{group}_windows.npz", **w)
        enc = encode(w["x"][:, :2] if group == "co2" else w["x"], device=device)
        torch.save(enc, G / "03_模型/encoded" / f"daily_{group}.pt")
        print(f"daily {group}: {len(w['x'])} windows, excluded {reasons}, {time.time() - t0:.0f}s", flush=True)
    from mg import multigas
    multigas.build()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    main(ap.parse_args().device)
