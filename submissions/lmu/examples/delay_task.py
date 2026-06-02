"""
LMU delay-learning task — the original v0.1.0 demo, ported to the
v0.2.0 `network`-style LMU class.

Drives the LMU with a band-limited white-noise input, then trains a
downstream spiking ensemble (via PES) to output the input delayed by a
fixed amount. Learning is on for the first 70 % of the run and off for
the last 30 %, so the tail measures generalisation.

Note: a fixed time delay is actually a *linear* functional of the
windowed input, so a precomputed linear decode would also work. This
file is kept as a reference to the v0.1.0 demo; for a properly
nonlinear readout, see `example_usage.py` (windowed RMS).
"""

from collections import deque

import nengo
import numpy as np

from lmu import LMU


SEED = 0
THETA = 1.0
ORDER = 8
DT = 0.001
SIM_T = 15.0
LEARN_OFF_AT = 0.7 * SIM_T
DELAY = 0.5
N_NEURONS = 600
LEARNING_RATE = 1e-3


class IdealDelay(nengo.synapses.Synapse):
    """Non-physical synapse that delays its input by exactly `delay` s."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        n = int(self.delay / dt)
        buf = deque([0.0] * n, maxlen=n)

        def step(t, x):
            buf.append(float(x[0]))
            return np.array([buf.popleft()])

        return step


def main():
    with nengo.Network(seed=SEED) as model:
        stim = nengo.Node(
            nengo.processes.WhiteSignal(
                high=2.0, period=SIM_T, rms=0.3, y0=0, seed=SEED,
            )
        )

        lmu = LMU(theta=THETA, order=ORDER, dt=DT)
        nengo.Connection(stim, lmu.input, synapse=None)

        ens = nengo.Ensemble(
            N_NEURONS, dimensions=ORDER,
            neuron_type=nengo.SpikingRectifiedLinear(),
        )
        nengo.Connection(lmu.state, ens, synapse=None)

        out = nengo.Node(size_in=1, label="delayed_estimate")
        learn_conn = nengo.Connection(
            ens, out, function=lambda x: 0.0,
            learning_rule_type=nengo.PES(LEARNING_RATE),
        )

        target = nengo.Node(size_in=1, label="delayed_target")
        nengo.Connection(stim, target, synapse=IdealDelay(DELAY))

        err = nengo.Node(
            lambda t, x: x if t < LEARN_OFF_AT else np.zeros_like(x),
            size_in=1,
        )
        nengo.Connection(out, err, synapse=None)
        nengo.Connection(target, err, synapse=None, transform=-1)
        nengo.Connection(err, learn_conn.learning_rule, synapse=None)

        p_target = nengo.Probe(target, synapse=0.01)
        p_out = nengo.Probe(out, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(SIM_T)

    t = sim.trange()
    mask_final = t >= SIM_T - 1.0
    target_final = sim.data[p_target][mask_final].flatten()
    out_final = sim.data[p_out][mask_final].flatten()
    final_err = float(np.mean(np.abs(target_final - out_final)))
    print(f"Delay = {DELAY} s, mean |error| over final 1 s: {final_err:.4f}")


if __name__ == "__main__":
    main()
