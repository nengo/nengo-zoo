"""
Basis-function decoding from the 3D CPG state to 10D body tensions.

Spatial basis `phi(z, m)` is a set of 10 Gaussian bumps along the body
(z in [0, 1]). Temporal basis `Phi(z) = [1, sin(2πz), cos(2πz), sin(4πz)]`
captures the standing + traveling wave components. The `Gamma` matrix
projects between bases; `Z_mat` evaluates the spatial basis at 10 equally
spaced segment positions.

These matrices are computed once at import time and cached at module level.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate as sp_integrate

N_SEGMENTS = 10
N_TEMPORAL_MODES = 3  # we use the first three components of Phi


def phi(z: float, m: int) -> float:
    """Gaussian spatial basis function at body position z, mode m."""
    return float(np.exp(-np.square(z - (m / 10.0)) / np.square(0.5)))


def Phi(z: float) -> list[float]:
    """Temporal basis (constant, fundamental, harmonic) at body position z."""
    return [
        1.0,
        float(np.sin(2.0 * np.pi * z)),
        float(np.cos(2.0 * np.pi * z)),
        float(np.sin(4.0 * np.pi * z)),
    ]


def _coefficient(n: int, m: int) -> float:
    integrand = lambda x: Phi(x)[n] * phi(x, m)
    return sp_integrate.quad(integrand, 0, 1)[0]


def build_decoding_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Compute (Gamma_inv, Z_mat). Pure function — callers may cache."""
    Gamma = np.zeros((N_TEMPORAL_MODES, N_SEGMENTS))
    for m in range(N_SEGMENTS):
        for n in range(N_TEMPORAL_MODES):
            Gamma[n, m] = _coefficient(n, m)
    Gamma_inv = np.linalg.pinv(Gamma)

    Z_mat = np.zeros((N_SEGMENTS, N_SEGMENTS))
    for z_idx, z in enumerate(np.linspace(0, 1, N_SEGMENTS)):
        for m in range(N_SEGMENTS):
            Z_mat[z_idx, m] = phi(z, m)

    return Gamma_inv, Z_mat


# Computed once at import time. ~30 scipy.quad calls; fast.
_GAMMA_INV, _Z_MAT = build_decoding_matrices()


def T(x):
    """Map a 3D CPG state vector to a 10D segment-tension vector."""
    return np.dot(_Z_MAT, np.dot(_GAMMA_INV, x))
