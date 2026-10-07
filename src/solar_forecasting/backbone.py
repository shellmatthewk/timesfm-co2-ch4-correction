"""Use the existing locked TimesFM checkpoint, with configurable array sizes."""
from __future__ import annotations

import inspect
import json
import math
import sys

import numpy as np
import torch

from .config import ROOT, file_sha, run_directory, write_json


def device_name(requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_body(device):
    base = ROOT / "timesfm_base"
    lock = json.loads((base / "model.lock.json").read_text(encoding="utf-8"))
    directory = base / lock["local_directory"]
    for name, detail in lock["files"].items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"TimesFM checkpoint missing: {path}. Download the locked weights as described in README.md.")
        if path.stat().st_size != detail["bytes"] or file_sha(path) != detail["sha256"]:
            raise ValueError(f"TimesFM checkpoint does not match model.lock.json: {path}")
    sys.path[:0] = [str(base / "src"), str(base / "vendor")]
    try:
        from uq14.backbone import load_backbone
        body = load_backbone(directory, device=device)
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install requirements-solar.txt to load TimesFM and SunPy dependencies") from exc
    return body, lock


@torch.no_grad()
def encode_arrays(body, x, horizon, device, batch):
    base = ROOT / "timesfm_base"
    sys.path.insert(0, str(base / "src"))
    from uq14.coordinates import _HistoryFeaturesView, _historical_trend
    maximum = min(2 * body.input_patch_len, body.output_patch_len) if body.use_stitching else body.output_patch_len
    if horizon > maximum:
        raise ValueError(f"This learned-head adapter supports horizons <= {maximum} steps for the locked checkpoint; got {horizon}")
    if not np.allclose(body.quantiles, np.arange(1, 10) / 10):
        raise ValueError("The solar heads require the locked nine-quantile output")
    decode = inspect.unwrap(type(body).decode)
    if decode is type(body).decode:
        raise RuntimeError("Expected the pinned TimesFM decode wrapper")
    result = {k: [] for k in ("features", "center", "scale", "quantiles")}
    for start in range(0, len(x), batch):
        inputs = torch.as_tensor(np.ascontiguousarray(x[start:start + batch]), dtype=torch.float32, device=device)
        b, c, length = inputs.shape
        _, aux = decode(_HistoryFeaturesView(body), target=inputs, horizon=horizon, return_aux_outputs=True)
        patch = math.ceil(length / body.input_patch_len) - 1
        mean, std = aux["revin_stats"]
        trend = _historical_trend(body, inputs.reshape(-1, length), horizon).reshape(b, c, horizon)
        result["features"].append(aux["__call__:transformer_output"][:, :, patch].cpu().numpy())
        result["center"].append((mean[:, :, patch, None] + trend).cpu().numpy())
        result["scale"].append(std[:, :, patch, None].cpu().numpy())
        result["quantiles"].append(body.decode(target=inputs, horizon=horizon).sort(dim=-1).values.cpu().numpy())
        print(f"Encoded {min(start + batch, len(x))}/{len(x)} windows", flush=True)
    encoded = {k: np.concatenate(v).astype(np.float32) for k, v in result.items()}
    rows = np.arange(horizon) * 9 + 4
    encoded["native_weight"] = body.output_head.weight.detach().cpu().numpy()[rows].copy()
    encoded["native_bias"] = body.output_head.bias.detach().cpu().numpy()[rows].copy()
    if not all(np.isfinite(a).all() for a in encoded.values()):
        raise ValueError("Nonfinite TimesFM encoding")
    return encoded


def encode(cfg, windows, audit):
    device = device_name(cfg["model"]["device"])
    body, lock = load_body(device)
    encoded = encode_arrays(body, windows["x"], cfg["windows"]["horizon"], device, cfg["model"]["encode_batch"])
    out = run_directory(cfg)
    np.savez_compressed(out / "encoding.npz", **encoded)
    write_json(out / "encoding_identity.json", {"windows_sha256": audit["windows_sha256"],
               "encoding_sha256": file_sha(out / "encoding.npz"), "model_lock": lock, "device": device})
    return encoded


def load_encoding(cfg, windows, audit):
    out = run_directory(cfg)
    if not (out / "encoding.npz").is_file():
        return encode(cfg, windows, audit)
    meta = json.loads((out / "encoding_identity.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "timesfm_base/model.lock.json").read_text(encoding="utf-8"))
    if meta["windows_sha256"] != audit["windows_sha256"] or meta["model_lock"] != lock:
        raise ValueError("Encoding was made for different windows or weights; run encode again")
    if meta["encoding_sha256"] != file_sha(out / "encoding.npz"):
        raise ValueError("Encoding cache changed; run encode again")
    with np.load(out / "encoding.npz", allow_pickle=False) as cache:
        return dict(cache)
