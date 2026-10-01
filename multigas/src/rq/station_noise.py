"""Residual Engression with one noise vector per station.

The 9/23 head (``rq.engression_np.Head``) draws one noise vector per trajectory,
eps [B,S,Z], and feeds the same vector to all four stations. Here every station
gets its own vector, eps [B,S,4,Z]. Nothing else changes: the same weights are
shared by the four stations, and the warm start, the median alignment and the
Energy Score are those of ``rq.engression_np``.

With independent noise the four stations are independent given the inputs;
the 14 days of one station still share that station's noise, so dependence
between days is learned as before.
"""
from __future__ import annotations

import numpy as np

from .engression_np import Head, silu

STATIONS = 4


class StationHead(Head):
    """Same parameters as ``Head``; forward/backward take eps [B,S,4,Z]."""

    def forward(self, h, eps):
        if eps.ndim != 4 or eps.shape[2] != h.shape[1] or eps.shape[3] != self.noise_dim:
            raise ValueError("eps must have shape [B,S,station,noise_dim].")
        a1 = (h @ self.Wh.T)[:, None] + eps @ self.We.T + self.b1
        z1, s1 = silu(a1)
        a2 = z1 @ self.W2.T + self.b2
        z2, s2 = silu(a2)
        out = z2 @ self.W3.T + self.b3
        return out, (h, eps, a1, z1, s1, a2, z2, s2)

    def backward(self, dout, cache):
        h, eps, a1, z1, s1, a2, z2, s2 = cache
        g = {}
        g["W3"] = np.einsum("bsco,bsch->oh", dout, z2)
        g["b3"] = dout.sum((0, 1, 2))
        dz2 = dout @ self.W3
        da2 = dz2 * (s2 * (1 + a2 * (1 - s2)))
        g["W2"] = np.einsum("bsco,bsch->oh", da2, z1)
        g["b2"] = da2.sum((0, 1, 2))
        dz1 = da2 @ self.W2
        da1 = dz1 * (s1 * (1 + a1 * (1 - s1)))
        g["b1"] = da1.sum((0, 1, 2))
        g["Wh"] = np.einsum("bch,bcd->hd", da1.sum(1), h)
        g["We"] = np.einsum("bsch,bscz->hz", da1, eps)
        return g


def draw(rng, n_windows, n_samples, noise_dim, per_station):
    """Noise for one batch. The shared case is the 9/23 stream, number for number."""
    if per_station:
        return rng.standard_normal((n_windows, n_samples, STATIONS, noise_dim))
    return rng.standard_normal((n_windows, n_samples, noise_dim))
