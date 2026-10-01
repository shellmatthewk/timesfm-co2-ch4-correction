"""TimesFM 3's native output head (`output_head.weight` 576 x 1280 and `output_head.bias` 576), the starting point of
every Engression head (rq.engression_np.native_q50_weights takes the median rows). It is part of the TimesFM weights,
which have a non-commercial license, so it is not in this repository; this script takes it from the downloaded
checkpoint after checking the checkpoint against timesfm_base/model.lock.json.

Usage: python src/extract_native_head.py      -> residual_engression/native_output_head.npz
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "timesfm_base"


def main():
    lock = json.loads((BASE / "model.lock.json").read_text())
    path = BASE / lock["local_directory"] / "model.safetensors"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != lock["files"]["model.safetensors"]["sha256"]:
        raise ValueError(f"{path} is not the locked TimesFM 3 checkpoint")
    with safe_open(str(path), "np") as f:
        weight, bias = f.get_tensor("output_head.weight"), f.get_tensor("output_head.bias")
    out = ROOT / "residual_engression" / "native_output_head.npz"
    np.savez(out, output_head_weight=weight, output_head_bias=bias)
    print("saved", out, weight.shape, bias.shape)


if __name__ == "__main__":
    main()
