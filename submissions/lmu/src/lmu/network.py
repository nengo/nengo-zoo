"""
Legendre Memory Unit (LMU).

A linear recurrent layer whose state vector parameterises the last
`theta` seconds of a scalar input in a Legendre polynomial basis. Any
linear functional of the windowed input can be read off the state via a
fixed transform; any nonlinear functional (RMS, peak, variance, a
classifier output, ...) can be computed by routing the state through a
downstream nonlinear readout (typically a Nengo ensemble whose decoders
are learned by PES or solved offline).

Reference
---------
Voelker, A. R., Kajić, I., & Eliasmith, C. (2019). Legendre Memory Units:
Continuous-time representation in recurrent neural networks.
Advances in Neural Information Processing Systems 32.
"""

from __future__ import annotations

import nengo
import numpy as np
from nengo.utils.filter_design import cont2discrete


def _lmu_matrices(order: int, theta: float, dt: float):
    """Return the discretised LMU state-space matrices (A_d, B_d).

    Derivation: see Voelker et al. (2019), Sec. 2. The continuous-time
    matrices are the Padé approximation of the pure-delay transfer
    function, projected onto the Legendre basis; we discretise with
    zero-order hold at the simulator's `dt`.
    """
    Q = np.arange(order, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q[:, None] * R
    C = np.ones((1, order))
    D = np.zeros((1,))
    A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dt=dt, method="zoh")
    return A_d, B_d


class LMU(nengo.Network):
    """Legendre Memory Unit — a recurrent linear filter that compresses
    the last `theta` seconds of a scalar input into a fixed-dimensional
    Legendre state.

    Parameters
    ----------
    theta : float
        Length of the sliding window (seconds).
    order : int
        Number of Legendre polynomials used to represent the window.
        Higher orders give better fidelity at the cost of more state
        dimensions. Voelker et al. (2019) used 6–12 for most tasks.
    dt : float, optional
        Simulator timestep used to discretise the LMU A, B matrices.
        Must match the eventual `nengo.Simulator`'s `dt`. Default 0.001.
    label : str, optional
        Network label.
    seed : int, optional
        Network seed.

    Attributes
    ----------
    theta : float
        The window length.
    order : int
        The state dimensionality.
    A, B : np.ndarray
        Discretised LMU state-space matrices, shapes (order, order) and
        (order, 1) respectively.
    input : nengo.Node
        Scalar input. Connect upstream signals here.
    state : nengo.Node
        The Legendre coefficients of the windowed input — an
        `order`-dimensional vector. Connect this to a downstream
        ensemble (or read it directly via a fixed transform) to compute
        functions of the windowed input.

    Notes
    -----
    The recurrent connection on `state` uses `synapse=0`, i.e. no
    synaptic filter. This is intentional — the LMU implements a
    discrete-time linear filter and the `state -> state` connection is
    the discretised LTI dynamics, not a Nengo ensemble's neural decode.
    """

    def __init__(
        self,
        theta: float,
        order: int,
        dt: float = 0.001,
        label: str | None = None,
        seed: int | None = None,
    ):
        super().__init__(label=label or "LMU", seed=seed)
        if theta <= 0:
            raise ValueError(f"theta must be positive; got {theta}.")
        if order < 1:
            raise ValueError(f"order must be >= 1; got {order}.")

        A, B = _lmu_matrices(order, theta, dt)
        self.theta = theta
        self.order = order
        self.dt = dt
        self.A = A
        self.B = B

        with self:
            self.input = nengo.Node(size_in=1, label="input")
            self.state = nengo.Node(size_in=order, label="state")
            nengo.Connection(self.input, self.state, transform=B, synapse=None)
            nengo.Connection(self.state, self.state, transform=A, synapse=0)
