"""Frozen TimesFM 3 (the project's locked checkpoint) on any group of series, on MPS.

Returns the same four arrays as the project's caches: features [N,u,1280],
center [N,u,14], scale [N,u,1] and sorted native quantiles [N,u,14,9].
The feature extraction is the project's Station14ForecastModel.encode written
for u series instead of exactly four.
"""
from __future__ import annotations

import inspect
import math
import sys
import numpy as np
import torch

from .data import BASE

sys.path[:0] = [str(BASE / "src"), str(BASE / "vendor")]
from uq14.coordinates import _HistoryFeaturesView, _historical_trend   # noqa: E402  (base project, read only)

_BODY = {}


def backbone(device="mps"):
    if device not in _BODY:
        from uq14.workflow import backbone as locked      # verifies the locked checkpoint hashes
        _BODY[device] = locked(device)
    return _BODY[device]


@torch.no_grad()
def encode(x, device="mps", batch=8):
    body = backbone(device)
    decode = inspect.unwrap(type(body).decode)
    parts = {"features": [], "center": [], "scale": [], "native_quantiles": []}
    for start in range(0, len(x), batch):
        t = torch.from_numpy(np.ascontiguousarray(x[start:start + batch], dtype=np.float32)).to(device)
        b, u, length = t.shape
        _, aux = decode(_HistoryFeaturesView(body), target=t, horizon=14, return_aux_outputs=True)
        patch = math.ceil(length / body.input_patch_len) - 1
        mean, std = aux["revin_stats"]
        trend = _historical_trend(body, t.reshape(-1, length), 14).reshape(b, u, 14)
        parts["features"].append(aux["__call__:transformer_output"][:, :, patch].cpu())
        parts["center"].append((mean[:, :, patch, None] + trend).cpu())
        parts["scale"].append(std[:, :, patch, None].cpu())
        parts["native_quantiles"].append(body.decode(target=t, horizon=14).sort(dim=-1).values.cpu())
    out = {k: torch.cat(v) for k, v in parts.items()}
    if not all(bool(torch.isfinite(v).all()) for v in out.values()):
        raise ValueError("nonfinite TimesFM output")
    return out
