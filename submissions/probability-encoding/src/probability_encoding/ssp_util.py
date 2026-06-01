"""
SSP utility helpers (integration, binding, pseudo-inverse).

Vendored verbatim from the source:
  https://github.com/ctn-waterloo/summerschool2025/blob/main/tutorials/vsa_probability/ssp_util.py

The upstream repository carries no LICENSE declaration; this submission
includes the code with the author's (Michael Furlong) permission, under
GPL-2.0-or-later to match the rest of NengoZoo.
"""

import numpy as np


def integrate_ssp(encoder, num_steps, x_min=-10, x_max=10):
    """Simple Riemann-sum integration of an SSP over a bounded domain."""
    xs = np.linspace(x_min, x_max, num_steps)
    dx = (x_max - x_min) / num_steps
    ssps = encoder.encode(np.atleast_2d(xs).T)
    return np.sum(ssps, axis=0) * dx


def invert(v: np.ndarray):
    """Pseudo-inverse of an SSP (FFT-domain conjugate)."""
    return v[:, -np.arange(v.shape[1])]


def bind(xs: np.ndarray, ys: np.ndarray):
    """Row-wise circular convolution of two collections of SSPs."""
    assert xs.shape[0] == ys.shape[0], "Must provide the same number of data points"
    assert xs.shape[1] == ys.shape[1], "Vectors must have the same dimensionality"
    x_fft = np.fft.fft(xs, axis=1)
    y_fft = np.fft.fft(ys, axis=1)
    xy_fft = np.multiply(x_fft, y_fft)
    return np.fft.ifft(xy_fft, axis=1).real
