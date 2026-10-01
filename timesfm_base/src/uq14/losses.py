"""Masked objectives for [batch, sample, station, lead] forecasts.

Each window receives equal weight, regardless of its number of observed future
cells. No missing label is imputed: ``where`` removes it before arithmetic.
The Energy Score uses only that window's observed subvector and divides its
Euclidean distances by sqrt(observed dimension). It therefore is not a score
for an unobserved complete 56-dimensional vector.

Energy score: Gneiting and Raftery (2007), Section 4.3:
https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
"""
from __future__ import annotations

import torch


def _validated(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if target.ndim != 3 or target.shape[0] == 0 or any(d == 0 for d in target.shape[1:]):
        raise ValueError("target must have nonempty shape [B, station, lead].")
    if mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be bool and have the same shape as target.")
    if mask.device != target.device:
        raise ValueError("mask and target must be on the same device.")
    if not target.is_floating_point():
        raise ValueError("target must be floating point.")
    counts = mask.flatten(1).sum(1)
    if bool((counts == 0).any()):
        raise ValueError("Every window must have at least one observed future cell.")
    if not bool(torch.isfinite(target.masked_select(mask)).all()):
        raise ValueError("Observed targets must be finite; missing targets require mask=False.")
    return counts


def _safe_predictions(prediction: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.device != target.device or not prediction.is_floating_point():
        raise ValueError("Predictions must be floating point on the target device.")
    if not bool(torch.isfinite(prediction.masked_select(mask.expand_as(prediction))).all()):
        raise ValueError("Predictions at observed future cells must be finite.")
    return (torch.where(mask, prediction, torch.zeros_like(prediction)),
            torch.where(mask, target, torch.zeros_like(target)))


def masked_mse(point: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of each window's mean squared error over observed cells.

    ``point`` is [B,C,H] or [B,1,C,H]. Multiple stochastic samples are rejected
    instead of being silently converted to a deterministic point forecast.
    """
    counts = _validated(target, mask)
    if point.ndim == 4 and point.shape[1] == 1:
        point = point[:, 0]
    if point.shape != target.shape:
        raise ValueError("MSE point must have target shape, or a singleton sample axis.")
    safe_point, safe_target = _safe_predictions(point, target, mask)
    return ((safe_point - safe_target).square().flatten(1).sum(1)
            / counts.to(point.dtype)).mean()


def masked_energy_score(samples: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Unbiased sample ES, without diagonal pairs; equal weighting of windows.

    E||X-y|| - (1/2)E||X-X'|| is estimated using S>=2 samples. The
    unordered-pair sum is divided by S(S-1); no S-by-S-by-D tensor is allocated.
    torch.linalg.vector_norm gives finite zero subgradients at zero distance.
    """
    counts = _validated(target, mask)
    if (samples.ndim != 4 or samples.shape[0] != target.shape[0]
            or samples.shape[2:] != target.shape[1:] or samples.shape[1] < 2):
        raise ValueError("samples must have shape [B,S,C,H] with S>=2.")
    safe_samples, safe_target = _safe_predictions(samples, target[:, None], mask[:, None])
    x = safe_samples.flatten(2)
    y = safe_target.flatten(2)
    first = torch.linalg.vector_norm(x - y, dim=-1).mean(1)
    pairs = first.new_zeros(first.shape)
    size = samples.shape[1]
    for index in range(size - 1):
        pairs = pairs + torch.linalg.vector_norm(x[:, index + 1:] - x[:, index:index + 1], dim=-1).sum(1)
    per_window = (first - pairs / (size * (size - 1))) / counts.to(first.dtype).sqrt()
    return per_window.mean()


# Descriptive aliases for callers migrating from a single-target experiment.
masked_mse_loss = masked_mse
energy_score = masked_energy_score
