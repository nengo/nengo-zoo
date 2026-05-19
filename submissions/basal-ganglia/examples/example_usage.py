"""
Demo: three competing actions with time-varying utilities, and a BG that
cleanly switches its selection as the leading utility changes.

CI requirement: this script must complete without error. Plotting is
optional (saves to figures/ if matplotlib is available) — the script
prints the selection sequence in either case.
"""

import os

import nengo
import numpy as np

from basal_ganglia import BasalGanglia


def utilities(t):
    """Three actions with utilities that swap every 0.3s."""
    if t < 0.3:
        return [0.8, 0.4, 0.2]
    if t < 0.6:
        return [0.3, 0.8, 0.2]
    return [0.2, 0.4, 0.8]


def build_model():
    with nengo.Network(seed=42) as model:
        bg = BasalGanglia(dimensions=3, n_neurons_per_ensemble=100)
        stim = nengo.Node(utilities)
        nengo.Connection(stim, bg.input)

        model.stim_probe = nengo.Probe(stim)
        model.out_probe = nengo.Probe(bg.output, synapse=0.01)
    return model


def selected_action(output_row):
    """The action whose GPi output is closest to zero (least inhibited)."""
    return int(np.argmax(output_row))


def main():
    model = build_model()
    with nengo.Simulator(model) as sim:
        sim.run(0.9)

    t = sim.trange()
    out = sim.data[model.out_probe]

    # Report selection at each 0.3s boundary.
    print("Action selection over time:")
    for label, target_t in [("0.25s (expect 0)", 0.25),
                            ("0.55s (expect 1)", 0.55),
                            ("0.85s (expect 2)", 0.85)]:
        idx = int(target_t / 0.001)
        print(f"  t={label}: selected action = {selected_action(out[idx])}")

    # Save figure if matplotlib is available and we have somewhere to put it.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, "selection.png")

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        axes[0].plot(t, sim.data[model.stim_probe])
        axes[0].set_ylabel("utility input")
        axes[0].legend(["action 0", "action 1", "action 2"], loc="upper right")
        axes[1].plot(t, out)
        axes[1].set_ylabel("BG output (GPi)")
        axes[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=110)
        print(f"Saved figure: {fig_path}")
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
