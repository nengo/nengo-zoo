"""
Function Inverse Estimator (FIE) bandwidth selection and density fitting.

Vendored verbatim from the source:
  https://github.com/ctn-waterloo/summerschool2025/blob/main/tutorials/vsa_probability/fie_util.py

The upstream repository carries no LICENSE declaration; this submission
includes the code with the author's (Michael Furlong) permission, under
GPL-2.0-or-later to match the rest of NengoZoo.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def fie(query_x, xs, xi, ls=0.1):
    dist_mat = cdist(query_x, xs, "minkowski", p=1.0)
    return np.maximum(0, np.mean(np.sinc(dist_mat / ls), axis=1) / ls - xi)


def fit_dist(train_xs, low, high, ls=0.1, num_steps=1000):
    test_xs = np.linspace(low, high, num_steps).reshape((-1, 1))
    dx = (high - low) / num_steps

    def min_func(xi, test_xs=test_xs, train_xs=train_xs, dx=dx):
        return (1 - dx * np.sum(fie(test_xs, train_xs, xi, ls=ls))) ** 2

    xi_0 = 0
    soln = minimize(min_func, xi_0, method="L-BFGS-B")
    return soln.x, soln.fun


def bandwidth_ecf(samples, init_h=1):
    def min_func(h, samps=samples):
        n = len(samps)
        val = np.mean(np.exp(1j * samps / h))
        return (np.abs(val) - 1 / np.sqrt(n + 1)) ** 2

    root_n = np.sqrt(samples.shape[0])
    soln = minimize(
        min_func,
        x0=init_h,
        method="L-BFGS-B",
        bounds=[(1 / root_n, None)],
    )
    return soln.x


def bandwidth_silverman(samples):
    n = samples.shape[0]
    std = np.std(samples)

    Q1 = np.percentile(samples, 25)
    Q3 = np.percentile(samples, 75)
    IQR = Q3 - Q1

    m = np.minimum(std, IQR / 1.349)
    bw = 0.9 * m / np.power(n, 1 / 5)

    return bw
