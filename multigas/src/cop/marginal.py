"""Quantile function built from the nine native TimesFM quantiles.

Between Q10 and Q90 the quantile function is linear in u between neighbouring
knots, so F^-1(0.1), ..., F^-1(0.9) equal the native quantiles. Outside, one of
three tails continues from Q10 / Q90:

* ``normal``       x = Q50 + sigma_L z (z < -z90), x = Q50 + sigma_U z (z > z90),
                   sigma_L = (Q50-Q10)/z90, sigma_U = (Q90-Q50)/z90
* ``exponential``  x = Q10 + (Q20-Q10) log(u/0.1), mirrored above Q90
                   (slope in u matches the neighbouring interior segment)
* ``linear``       the neighbouring interior segment is extended to u=0 / u=1

Everything is evaluated from the normal score z = Phi^-1(u), which keeps the
tails finite and differentiable. ``q`` is [..., 9], sorted, in ppm.
"""
from __future__ import annotations

import math
import torch

Z90 = 1.2815515655446004  # Phi^-1(0.9)
TAILS = ("normal", "exponential", "linear")
_MIN_SCALE = 1e-6
_LOG01 = math.log(0.1)


def _check(q, tail):
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}.")
    if q.shape[-1] != 9 or not q.is_floating_point():
        raise ValueError("q must be floating point with a trailing dimension of 9.")
    if not bool(torch.isfinite(q).all()) or bool((q.diff(dim=-1) < 0).any()):
        raise ValueError("q must be finite and nondecreasing.")


def icdf_from_z(q: torch.Tensor, z: torch.Tensor, tail: str = "normal") -> torch.Tensor:
    """x = F^-1(Phi(z)). q is [B,C,H,9]; z is [B,S,C,H]; returns [B,S,C,H]."""
    _check(q, tail)
    qe = q[:, None]                                         # [B,1,C,H,9]
    u = torch.special.ndtr(z)
    pos = ((u - 0.1) / 0.1).clamp(0.0, 8.0)
    idx = pos.detach().floor().clamp(max=7.0).long()
    frac = pos - idx
    shape = z.shape + (9,)
    lo = torch.gather(qe.expand(shape), -1, idx[..., None])[..., 0]
    hi = torch.gather(qe.expand(shape), -1, (idx + 1)[..., None])[..., 0]
    middle = lo + frac * (hi - lo)
    q10, q20, q50, q80, q90 = (qe[..., i] for i in (0, 1, 4, 7, 8))
    if tail == "normal":
        lower = q50 + ((q50 - q10) / Z90).clamp_min(_MIN_SCALE) * z
        upper = q50 + ((q90 - q50) / Z90).clamp_min(_MIN_SCALE) * z
    elif tail == "exponential":
        lower = q10 + (q20 - q10).clamp_min(_MIN_SCALE) * (torch.special.log_ndtr(z) - _LOG01)
        upper = q90 - (q90 - q80).clamp_min(_MIN_SCALE) * (torch.special.log_ndtr(-z) - _LOG01)
    else:
        lower = q10 + (q20 - q10).clamp_min(_MIN_SCALE) * (u - 0.1) / 0.1
        upper = q90 + (q90 - q80).clamp_min(_MIN_SCALE) * (u - 0.9) / 0.1
    return torch.where(z < -Z90, lower, torch.where(z > Z90, upper, middle))


def icdf(q: torch.Tensor, u: torch.Tensor, tail: str = "normal") -> torch.Tensor:
    """x = F^-1(u) for u in (0,1), interpolating directly in u inside [0.1, 0.9]."""
    _check(q, tail)
    if bool(((u <= 0) | (u >= 1)).any()):
        raise ValueError("u must lie strictly inside (0,1).")
    x = icdf_from_z(q, torch.special.ndtri(u), tail)
    qe = q[:, None]
    pos = ((u - 0.1) / 0.1).clamp(0.0, 8.0)
    idx = pos.round().long()
    knot = torch.gather(qe.expand(u.shape + (9,)), -1, idx[..., None])[..., 0]
    on_knot = ((pos - idx).abs() < 1e-12) & (u >= 0.1 - 1e-13) & (u <= 0.9 + 1e-13)
    return torch.where(on_knot, knot, x)


def normal_score(q: torch.Tensor, y: torch.Tensor, tail: str = "normal") -> torch.Tensor:
    """z = Phi^-1(F(y)), the inverse of :func:`icdf_from_z`. q [B,C,H,9], y [B,C,H].

    NaN in y stays NaN. Zero-width interior segments return their left end.
    """
    _check(q, tail)
    q10, q20, q50, q80, q90 = (q[..., i] for i in (0, 1, 4, 7, 8))
    safe = torch.where(torch.isfinite(y), y, q50)
    idx = (torch.searchsorted(q.contiguous(), safe[..., None].contiguous(), right=True)[..., 0] - 1).clamp(0, 7)
    lo = torch.gather(q, -1, idx[..., None])[..., 0]
    hi = torch.gather(q, -1, (idx + 1)[..., None])[..., 0]
    width = hi - lo
    frac = torch.where(width > 0, (safe - lo) / torch.where(width > 0, width, torch.ones_like(width)),
                       torch.zeros_like(width)).clamp(0.0, 1.0)
    u_mid = (0.1 * (idx + frac + 1)).clamp(0.1, 0.9)
    z_mid = torch.special.ndtri(u_mid)
    tiny = 1e-300
    if tail == "normal":
        z_lo = (safe - q50) / ((q50 - q10) / Z90).clamp_min(_MIN_SCALE)
        z_hi = (safe - q50) / ((q90 - q50) / Z90).clamp_min(_MIN_SCALE)
    elif tail == "exponential":
        u_lo = 0.1 * torch.exp((safe - q10).clamp(max=0.0) / (q20 - q10).clamp_min(_MIN_SCALE))
        u_hi = 0.1 * torch.exp(-(safe - q90).clamp(min=0.0) / (q90 - q80).clamp_min(_MIN_SCALE))
        z_lo = torch.special.ndtri(u_lo.clamp_min(tiny))
        z_hi = -torch.special.ndtri(u_hi.clamp_min(tiny))
    else:
        u_lo = 0.1 + 0.1 * (safe - q10) / (q20 - q10).clamp_min(_MIN_SCALE)
        u_hi = 0.1 - 0.1 * (safe - q90) / (q90 - q80).clamp_min(_MIN_SCALE)
        z_lo = torch.special.ndtri(u_lo.clamp(tiny, 0.1))
        z_hi = -torch.special.ndtri(u_hi.clamp(tiny, 0.1))
    z = torch.where(safe < q10, z_lo, torch.where(safe > q90, z_hi, z_mid))
    return torch.where(torch.isfinite(y), z, torch.full_like(z, float("nan")))
