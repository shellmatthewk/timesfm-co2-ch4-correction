"""The affine output coordinates paired with TimesFM's hidden features.

Pinned to TimesFM source e31dadd84cb26bd5153fde6687502b8312e918fb.
The context can be linearly detrended *before* RevIN and the Transformer.
Consequently a direct replacement head must undo that same transformation,
rather than apply the mean/std of the original, non-detrended context.

No native forecast is used to compute any returned value. In particular, the
future trend is extrapolated solely from the historical context. The native
output head can be replaced without changing these coordinates or features.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _HistoryFeaturesView:
    """Reuse decode preprocessing, omitting dormant future-CPM refinement.

    Within one native output patch only the last historical patch contributes
    to the forecast. Future-CPM refinement does not affect that patch and its
    unused sqrt(0) graph can produce non-finite gradients for constant histories.
    This proxy does not mutate the underlying model or its configuration.
    """

    def __init__(self, backbone: nn.Module) -> None:
        self.backbone = backbone

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backbone, name)

    def forward(self, inputs, freeze_after=None, patch_cpm_mask=None, return_aux_outputs=False):
        return self.backbone(
            inputs,
            freeze_after=freeze_after,
            patch_cpm_mask=None,
            return_aux_outputs=return_aux_outputs,
        )


def _historical_trend(backbone: nn.Module, context: torch.Tensor, horizon: int) -> torch.Tensor:
    """Reproduce the pinned decoder's masked, padded trend fit and decision.

    Operations and float32 time indices intentionally match upstream decode;
    even the padding affects the fitted slope's time coordinate. Using another
    OLS implementation or a separately rounded detrending decision can disagree
    for nearly constant series. No values after the forecast origin enter here.
    """
    if not backbone.use_linear_detrending:
        return context.new_zeros((context.shape[0], horizon))
    padding = (-context.shape[-1]) % backbone.input_patch_len
    values = F.pad(context[:, None, :], (padding, 0))
    length = values.shape[-1]
    masks = torch.zeros_like(values, dtype=torch.bool)
    if padding:
        masks[..., :padding] = True
    t = torch.arange(-(length - 1), 1, dtype=torch.float32, device=context.device)
    t = t[None, None, :] / length
    valid = ~masks
    n = valid.float().sum(dim=-1, keepdim=True)
    sum_t = torch.where(valid, t, 0.0).sum(dim=-1, keepdim=True)
    sum_t2 = torch.where(valid, t**2, 0.0).sum(dim=-1, keepdim=True)
    sum_y = torch.where(valid, values, 0.0).sum(dim=-1, keepdim=True)
    sum_ty = torch.where(valid, t * values, 0.0).sum(dim=-1, keepdim=True)
    det = n * sum_t2 - sum_t**2
    safe_det = torch.where(det == 0.0, 1.0, det)
    slope = torch.where(det == 0.0, 0.0, (n * sum_ty - sum_t * sum_y) / safe_det)
    intercept = torch.where(
        det == 0.0,
        torch.where(n > 0, sum_y / torch.clamp_min(n, 1.0), 0.0),
        (sum_y - slope * sum_t) / torch.clamp_min(n, 1.0),
    )
    detrended = values - (slope * t + intercept)
    mean_y = sum_y / torch.clamp_min(n, 1.0)
    sum_y2 = torch.where(valid, values**2, 0.0).sum(dim=-1, keepdim=True)
    var_original = torch.clamp_min(sum_y2 / torch.clamp_min(n, 1.0) - mean_y**2, 0.0)
    std_original = torch.sqrt(var_original)
    sum_yd = torch.where(valid, detrended, 0.0).sum(dim=-1, keepdim=True)
    mean_yd = sum_yd / torch.clamp_min(n, 1.0)
    sum_yd2 = torch.where(valid, detrended**2, 0.0).sum(dim=-1, keepdim=True)
    var_detrended = torch.clamp_min(sum_yd2 / torch.clamp_min(n, 1.0) - mean_yd**2, 0.0)
    std_detrended = torch.sqrt(var_detrended)
    apply = std_detrended < backbone.linear_detrending_threshold * std_original
    future_t = torch.arange(1, horizon + 1, dtype=torch.float32, device=context.device) / length
    trend = slope[:, :, 0, None] * future_t[None, None, :] + intercept[:, :, 0, None]
    return torch.where(apply[:, :, 0, None], trend, 0.0)[:, 0]


def extract_direct_features(
    backbone: nn.Module, context: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(hidden[B,D], output_center[B,H], output_scale[B,1])``.

    A replacement head predicts in the native head's coordinates. Its original
    units are ``output_center + output_scale * head(hidden)``. Hidden features
    are returned as-is, without an extra LayerNorm. The scale is the *exact*
    RevIN scale, including zero for constant histories; no new epsilon is added.

    This affine identity matches native unsorted outputs when native value
    clipping is inactive (the checkpoint's threshold is 1e20). Clipping itself
    is nonlinear and cannot be represented by an affine center/scale. Quantile
    sorting is a separate operation and is likewise not part of this mapping.

    Supports a finite single target and one forecast patch only. The caller
    controls gradient mode and which backbone parameters require gradients.
    """
    if context.ndim != 2 or context.shape[-1] < 2:
        raise ValueError("context must have shape [batch, length] with length >= 2.")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError("horizon must be an integer.")
    maximum = (
        min(2 * backbone.input_patch_len, backbone.output_patch_len)
        if backbone.use_stitching else backbone.output_patch_len
    )
    if not 1 <= horizon <= maximum:
        raise ValueError(f"horizon must be between 1 and {maximum} (one output patch).")
    if getattr(backbone, "input_transform", "identity") != "identity":
        raise ValueError("Only the pinned identity input transform is supported.")
    param = next(backbone.parameters())
    context = context.to(device=param.device, dtype=param.dtype)
    if not bool(torch.isfinite(context).all()):
        raise ValueError("context must contain only finite historical observations.")
    decode = inspect.unwrap(type(backbone).decode)
    if decode is type(backbone).decode:
        raise RuntimeError("Expected pinned TimesFM3 decode wrapper; re-audit upstream API.")
    # Decode's output forecast is deliberately discarded. Both aux tensors used
    # below are computed upstream of the native prediction head.
    _, aux = decode(
        _HistoryFeaturesView(backbone), target=context[:, None, :],
        horizon=horizon, return_aux_outputs=True,
    )
    patch = math.ceil(context.shape[-1] / backbone.input_patch_len) - 1
    hidden = aux["__call__:transformer_output"][:, 0, patch]
    mean, std = aux["revin_stats"]
    center = mean[:, 0, patch, None] + _historical_trend(backbone, context, horizon)
    scale = std[:, 0, patch, None]
    return hidden, center, scale
