"""Initialize a direct MLP to a pretrained linear forecast mapping.

Two paired SiLU units reproduce a linear value exactly in real arithmetic:
``SiLU(x) - SiLU(-x) = x``. Applying that identity twice initializes the existing
three-Linear-layer MLP to the native median head without adding a residual path
or reading native predictions during the new model's forward pass.

Optional integration: select the native median rows for the desired horizon,
construct the ordinary direct MLP, and call ``initialize_silu_head_from_linear``
before creating its optimizer. Feed it the native (unnormalized) hidden features
and undo RevIN/detrending with ``extract_direct_features``. Applying an extra
feature LayerNorm afterwards would invalidate this initialization's parity.
"""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def initialize_silu_head_from_linear(
    head: nn.Sequential,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    noise_dim: int = 0,
    noise_scale: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Initialize ``Linear,SiLU,Linear,SiLU,Linear`` to ``Wx+b+A*noise``.

    ``weight`` has shape [H,D], optional ``bias`` [H], and the first Linear must
    accept D+Z inputs, with noise concatenated *only at that input*. Each hidden
    layer needs at least 2H neurons. The first 2H neurons implement paired SiLU
    identities; the remaining neurons retain their initial random parameters.
    The final spare columns start at zero, but remain parameters that can learn.

    ``noise_scale`` is a nonnegative scalar or length-H tensor, in normalized
    output units. Nonzero scales require Z>=H. The returned A[H,Z] has those
    scales on its first H diagonal entries, so independent standard Gaussian
    input noise gives exactly the requested initial marginal standard deviation.
    Zero scale works with any nonnegative Z, including Z=0.

    This function performs no random draws and does not change requires_grad.
    The caller sets seeds before constructing the MLP to control spare weights.
    Inputs are checked before modifying parameters. Floating point arithmetic
    introduces rounding error, especially under reduced precision.
    """
    if not isinstance(head, nn.Sequential) or len(head) != 5:
        raise ValueError("head must be Sequential(Linear, SiLU, Linear, SiLU, Linear).")
    if not all(isinstance(head[index], nn.Linear) for index in (0, 2, 4)) or not all(
        isinstance(head[index], nn.SiLU) for index in (1, 3)
    ):
        raise ValueError("head must be Sequential(Linear, SiLU, Linear, SiLU, Linear).")
    if not isinstance(noise_dim, int) or isinstance(noise_dim, bool) or noise_dim < 0:
        raise ValueError("noise_dim must be a nonnegative integer.")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("weight must be a [horizon, hidden_features] tensor.")
    horizon, features = weight.shape
    if horizon < 1 or features < 1:
        raise ValueError("weight dimensions must be positive.")
    first, middle, last = head[0], head[2], head[4]
    if first.in_features != features + noise_dim or last.out_features != horizon:
        raise ValueError("head input/output dimensions do not match weight and noise_dim.")
    if first.out_features != middle.in_features or middle.out_features != last.in_features:
        raise ValueError("head's adjacent Linear dimensions must agree.")
    if first.out_features < 2 * horizon or middle.out_features < 2 * horizon:
        raise ValueError("Both hidden layers require at least 2*horizon neurons.")
    device, dtype = first.weight.device, first.weight.dtype
    if not all(p.device == device and p.dtype == dtype for p in head.parameters()):
        raise ValueError("All head parameters must have the same device and dtype.")
    native_weight = weight.detach().to(device=device, dtype=dtype)
    if bias is None:
        native_bias = native_weight.new_zeros(horizon)
    else:
        if not isinstance(bias, torch.Tensor) or bias.shape != (horizon,):
            raise ValueError("bias must have shape [horizon].")
        native_bias = bias.detach().to(device=device, dtype=dtype)
    scales = torch.as_tensor(noise_scale, device=device, dtype=dtype)
    if scales.ndim == 0:
        scales = scales.expand(horizon)
    elif scales.shape != (horizon,):
        raise ValueError("noise_scale must be a scalar or have shape [horizon].")
    if not bool(torch.isfinite(native_weight).all() and torch.isfinite(native_bias).all()):
        raise ValueError("weight and bias must be finite.")
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("noise_scale must be finite and nonnegative.")
    if bool((scales > 0).any()) and noise_dim < horizon:
        raise ValueError("Nonzero noise_scale requires noise_dim >= horizon for orthogonal noise.")
    if first.bias is None and bool((native_bias != 0).any()):
        raise ValueError("A nonzero native bias requires the first Linear to have a bias.")

    projection = native_weight.new_zeros((horizon, noise_dim))
    if noise_dim >= horizon:
        projection[:, :horizon].copy_(torch.diag(scales))
    coefficients = torch.cat([native_weight, projection], dim=-1)
    identity = torch.eye(horizon, device=device, dtype=dtype)

    # First paired preactivations: x=Wh+b+A*noise and -x. Spare output rows stay
    # as constructed, including their random connections to input noise.
    first.weight[:horizon].copy_(coefficients)
    first.weight[horizon : 2 * horizon].copy_(-coefficients)
    if first.bias is not None:
        first.bias[:horizon].copy_(native_bias)
        first.bias[horizon : 2 * horizon].copy_(-native_bias)

    # Recover +/-x before the second SiLU. Unused input columns to these paired
    # rows must start at zero so spare first-layer units cannot change parity.
    middle.weight[: 2 * horizon].zero_()
    middle.weight[:horizon, :horizon].copy_(identity)
    middle.weight[:horizon, horizon : 2 * horizon].copy_(-identity)
    middle.weight[horizon : 2 * horizon, :horizon].copy_(-identity)
    middle.weight[horizon : 2 * horizon, horizon : 2 * horizon].copy_(identity)
    if middle.bias is not None:
        middle.bias[: 2 * horizon].zero_()

    # Recover x again. This is the ordinary final MLP layer, not a separate
    # native-forecast path. All zero columns remain fully trainable parameters.
    last.weight.zero_()
    last.weight[:, :horizon].copy_(identity)
    last.weight[:, horizon : 2 * horizon].copy_(-identity)
    if last.bias is not None:
        last.bias.zero_()
    return projection
