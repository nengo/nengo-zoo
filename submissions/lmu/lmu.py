"""
Legendre Memory Unit (LMU) in Nengo — Voelker, Kajić & Eliasmith (NeurIPS 2019).

This script builds the network at module level so NengoGUI can pick it up
directly (`nengo lmu.py`). It does *not* run a simulation here; for the
long-form training run that produces the published plots, see
``examples/run_training.py``. For the cell-by-cell walkthrough, see
``lmu.ipynb``.
"""

from collections import deque

import nengo
import numpy as np
from nengo.utils.filter_design import cont2discrete


# ---- LMU parameters ---------------------------------------------------------
theta = 1.0        # length of the sliding window (seconds)
order = 8          # number of Legendre polynomials representing the window

# ---- Input / task parameters -----------------------------------------------
freq  = 2          # band-limit of input signal (Hz)
rms   = 0.30       # amplitude of input (keeps values in [-1, 1])
delay = 0.5        # length of time delay the downstream ensemble learns

# ---- Simulation parameters -------------------------------------------------
dt    = 0.001      # simulation timestep
sim_t = 100        # nominal full-training length (seconds)
seed  = 0          # for deterministic results


# ---- LMU state-space matrices (from Voelker et al., 2019) ------------------
def _lmu_matrices(order: int, theta: float, dt: float):
    Q = np.arange(order, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q[:, None] * R
    C = np.ones((1, order))
    D = np.zeros((1,))
    A_d, B_d, _, _, _ = cont2discrete((A, B, C, D), dt=dt, method="zoh")
    return A_d, B_d


A, B = _lmu_matrices(order, theta, dt)


# ---- Ideal delay synapse (ground truth for training) -----------------------
class IdealDelay(nengo.synapses.Synapse):
    """A non-physical synapse that delays its input by exactly `delay` seconds."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        buffer = deque([0] * int(self.delay / dt))

        def delay_func(t, x):
            buffer.append(x.copy())
            return buffer.popleft()

        return delay_func


# ---- Network ---------------------------------------------------------------
with nengo.Network(seed=seed) as model:
    # Band-limited white-noise input.
    stim = nengo.Node(
        output=nengo.processes.WhiteSignal(
            high=freq, period=sim_t, rms=rms, y0=0, seed=seed
        )
    )

    # LMU: a recurrent linear filter with the analytically-derived A, B.
    lmu = nengo.Node(size_in=order)
    nengo.Connection(stim, lmu, transform=B, synapse=None)
    nengo.Connection(lmu, lmu, transform=A, synapse=0)

    # Downstream spiking ensemble that we'll train to read the LMU state.
    ens = nengo.Ensemble(1000, order, neuron_type=nengo.SpikingRectifiedLinear())
    nengo.Connection(lmu, ens, synapse=None)

    out = nengo.Node(size_in=1)

    # Error signal node: PES error = (target − actual), shut off after 80% of run
    # so the last 20% measures generalization.
    err_node = nengo.Node(lambda t, x: x if t < sim_t * 0.8 else 0, size_in=1)

    # Target: ideally-delayed input (negated; the ensemble's output is added).
    nengo.Connection(stim, err_node, synapse=IdealDelay(delay), transform=-1)
    nengo.Connection(out, err_node, synapse=None)

    learn_conn = nengo.Connection(
        ens, out, function=lambda x: 0, learning_rule_type=nengo.PES(2e-4)
    )
    nengo.Connection(err_node, learn_conn.learning_rule, synapse=None)

    # Probes for offline analysis.
    p_stim  = nengo.Probe(stim)
    p_ideal = nengo.Probe(stim, synapse=IdealDelay(delay))
    p_out   = nengo.Probe(out)
