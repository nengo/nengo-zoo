"""
Demo: learn windowed RMS of a band-limited input using PES on a
downstream spiking ensemble that reads the LMU state.

The LMU compresses the last `theta` seconds of a scalar input into a
fixed-dimensional vector. A spiking ensemble reads that vector and is
trained, via PES on its output decoders, to produce a target signal —
here, the root-mean-square of the input over the same window. After
70 % of the run, the learning rule is switched off and the remaining
time evaluates generalisation.

RMS is the simplest non-trivial nonlinear function of the windowed
input. The same shape of network learns any other windowed nonlinearity
— variance, peak, threshold-crossing rate, a small classifier — only
the target signal needs to change. See `delay_task.py` in this folder
for the (linear) delay-learning task the v0.1.0 version of this
submission used.

CI requirement: completes without error. Plotting is optional.
"""

import os
from collections import deque
from pathlib import Path

import nengo
import numpy as np

from lmu import LMU


SEED = 0
THETA = 1.0           # window length (s)
ORDER = 8             # LMU state dimensionality
DT = 0.001
SIM_T = 25.0          # total sim duration — keep CI-fast
LEARN_OFF_AT = 0.75 * SIM_T
N_NEURONS = 1000
LEARNING_RATE = 5e-4


class WindowedRMS(nengo.synapses.Synapse):
    """Synapse that emits the RMS of its input over a `window`-second buffer.

    Used here as a convenient way to compute the true RMS target online,
    inside the model, without having to pre-materialise the input signal.
    """

    def __init__(self, window):
        super().__init__()
        self.window = window

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        n = int(self.window / dt)
        buf = deque([0.0] * n, maxlen=n)

        def step(t, x):
            buf.append(float(x[0]))
            arr = np.fromiter(buf, dtype=float)
            return np.array([float(np.sqrt(np.mean(arr * arr)))])

        return step


def main():
    with nengo.Network(seed=SEED) as model:
        # Two band-limited noise sources: a fast "carrier" and a slow
        # envelope, multiplied to give an amplitude-modulated input
        # whose windowed RMS varies smoothly in time.
        carrier = nengo.Node(
            nengo.processes.WhiteSignal(
                high=2.0, period=SIM_T, rms=0.6, y0=0, seed=SEED,
            )
        )
        env_raw = nengo.Node(
            nengo.processes.WhiteSignal(
                high=0.2, period=SIM_T, rms=0.3, y0=0, seed=SEED + 1,
            )
        )
        # Modulator: 0.4 + |env_raw|, kept strictly positive.
        stim = nengo.Node(
            lambda t, x: float(x[0]) * (0.4 + abs(float(x[1]))),
            size_in=2,
            label="stim",
        )
        nengo.Connection(carrier, stim[0], synapse=None)
        nengo.Connection(env_raw, stim[1], synapse=None)

        # The LMU itself.
        lmu = LMU(theta=THETA, order=ORDER, dt=DT)
        nengo.Connection(stim, lmu.input, synapse=None)

        # Nonlinear readout: a spiking ensemble decoded via PES.
        ens = nengo.Ensemble(
            N_NEURONS,
            dimensions=ORDER,
            neuron_type=nengo.SpikingRectifiedLinear(),
        )
        nengo.Connection(lmu.state, ens, synapse=None)

        out = nengo.Node(size_in=1, label="rms_estimate")
        learn_conn = nengo.Connection(
            ens, out, function=lambda x: 0.0,
            learning_rule_type=nengo.PES(LEARNING_RATE),
        )

        # Target: true windowed RMS of the same input.
        target = nengo.Node(size_in=1, label="rms_target")
        nengo.Connection(stim, target, synapse=WindowedRMS(THETA))

        # Error = output − target, gated off after LEARN_OFF_AT.
        err = nengo.Node(
            lambda t, x: x if t < LEARN_OFF_AT else np.zeros_like(x),
            size_in=1,
            label="err",
        )
        nengo.Connection(out, err, synapse=None)
        nengo.Connection(target, err, synapse=None, transform=-1)
        nengo.Connection(err, learn_conn.learning_rule, synapse=None)

        p_stim = nengo.Probe(stim)
        p_target = nengo.Probe(target, synapse=0.01)
        p_out = nengo.Probe(out, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(SIM_T)

    t = sim.trange()
    stim_data = sim.data[p_stim].flatten()
    target_data = sim.data[p_target].flatten()
    out_data = sim.data[p_out].flatten()

    # Final-second error (post-learning-off, so this is generalisation).
    mask_final = t >= SIM_T - 1.0
    final_err = np.mean(np.abs(target_data[mask_final] - out_data[mask_final]))
    print(f"Mean |error| over final 1s: {final_err:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["text.usetex"] = False
        import matplotlib.pyplot as plt

        fig_dir = Path(__file__).resolve().parent.parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axs = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
        axs[0].plot(t, stim_data, color="gray", lw=0.8)
        axs[0].set_ylabel("input")
        axs[0].set_title("Amplitude-modulated band-limited input")

        axs[1].plot(t, target_data, color="gray", label="true RMS")
        axs[1].plot(t, out_data, "--k", label="decoded RMS")
        axs[1].axvline(LEARN_OFF_AT, ls=":", color="k", alpha=0.6)
        axs[1].set_ylabel("RMS")
        axs[1].legend(loc="upper right")
        axs[1].set_title(
            f"Windowed RMS (theta = {THETA:g} s)  —  learning off at t = "
            f"{LEARN_OFF_AT:g} s"
        )

        axs[2].plot(t, np.abs(target_data - out_data), color="C3", lw=0.8)
        axs[2].axvline(LEARN_OFF_AT, ls=":", color="k", alpha=0.6)
        axs[2].set_ylabel("|error|")
        axs[2].set_xlabel("time (s)")
        axs[2].set_title("Absolute error")

        fig.tight_layout()
        fig.savefig(fig_dir / "rms_learning.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'rms_learning.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
