"""
Vendored from https://github.com/maddybartlett/ImprovedRLContinuousStateReps
(network/rlnet/utils.py), commit fc75ae0 (2024-02-01).

This file is © Maddy Bartlett et al. and distributed under the Applied Brain
Research academic-use-only license — see LICENSE at the submission root.

Local edits relative to upstream:
  - Only `sparsity_to_x_intercept` (used by the actor-critic networks) and
    `softmax` (useful for action selection in examples) are vendored. The
    upstream `rend`, `save_gifs`, `next_power_of_2`, `get_ac_output`,
    `plot_policy`, and `plot_table` helpers are dropped — they depend on
    matplotlib + matplotlib.animation and hardcode RatBox/CartPole grid
    geometry, so they're not generally reusable.
  - Removed the matplotlib import (no longer needed).
"""

import math  # noqa: F401  (kept to preserve upstream module-level imports)

import numpy as np
import scipy.special
from scipy.special import log_softmax


## Convert sparsity parameter to neuron bias/intercept
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1 - scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))


## Softmax Function used for selecting next action
def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    filtered_x = np.nan_to_num(x-x.max())
    return np.exp(log_softmax(filtered_x, axis=axis))
