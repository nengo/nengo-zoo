# 2D Decision Integrator

# This is a model of perceptual decision making using a two dimensional
# integrator. As mentioned in the book, the goal is to construct a simple model
# of perceptual decision making without being concerned with establishing a
# detailed connection to neurobiology (we have done this elsewhere).

# Rather than having two different integrators for each dimension, you will
# build the model using a single two dimensional integrator. This integrator
# can be used irrespective of the task demands since it effectively integrates
# in every direction simultaneously. This is neurally more efficient due to the
# reasons explained in the book.

# The model has four ensembles: MT representing the motion area, LIP
# representing the lateral intraparietal area, input and output of the 2D
# integrator. The parameters used in the model are as described in the book.
# The 2D integrator resides in LIP. As discussed in the book an integrator
# requires two connections: here, the input from MT to LIP and the feedback
# connection from LIP to LIP.

# Here, you will provide a an input of (-0.5, 0.5) to the model spanning over a
# period of 6 seconds to observe the model behaviour. In order to inject noise
# while the simulation runs, you can use the 'noise' parameter when creating
# ensembles as shown. The reason for injecting noise is explained in the book.

# Press the play button to run the simulation.
# The output plot on the bottom-right shows the output of the 2D decision
# integrator which is represented by a single two dimensional output ensemble.
# You can see that as MT encodes the input over time, LIP slowly moves towards
# the same direction as it acuumulates evidence that there is sustained motion
# in that direction.

# Thus MT moves LIP in the right direction and once past a certain threshold,
# the output neurons start firing. To visualize this:
# 1) Select 'spikes' from the right-click menu of the output ensemble. This will
#    display a spike plot.
# 2) Run the simulation and then slide the blue box in the simulation control
#    bar backwards.
# 3) You will see that the spikes become stronger once past a certain threshold
#    (i.e., when LIP starts following MT)

# Setup the environment
import nengo
from nengo.dists import Uniform
from nengo.processes import WhiteNoise

# Create the network object to which we can add ensembles, connections, etc.
model = nengo.Network(label="2D Decision Integrator", seed=11)

with model:
    # Inputs
    input1 = nengo.Node(-0.5, label="Input 1")
    input2 = nengo.Node(0.5, label="Input 2")

    # Ensembles
    ens_inp = nengo.Ensemble(100, dimensions=2, label="Input")
    MT = nengo.Ensemble(100, dimensions=2, noise=WhiteNoise(dist=Uniform(-0.3, 0.3)))
    LIP = nengo.Ensemble(200, dimensions=2, noise=WhiteNoise(dist=Uniform(-0.3, 0.3)))
    ens_out = nengo.Ensemble(
        100,
        dimensions=2,
        intercepts=Uniform(0.3, 1),
        noise=WhiteNoise(dist=Uniform(-0.3, 0.3)),
        label="Output",
    )

    weight = 0.1
    # Connecting the input signal to the input ensemble
    nengo.Connection(input1, ens_inp[0], synapse=0.01)
    nengo.Connection(input2, ens_inp[1], synapse=0.01)

    # Providing input to MT ensemble
    nengo.Connection(ens_inp, MT, synapse=0.01)

    # Connecting MT ensemble to LIP ensemble
    nengo.Connection(MT, LIP, transform=weight, synapse=0.1)

    # Connecting LIP ensemble to itself
    nengo.Connection(LIP, LIP, synapse=0.1)

    # Connecting LIP population to output
    nengo.Connection(LIP, ens_out, synapse=0.01)

    # Probes for the headline figure (harmless for NengoGUI).
    p_input = nengo.Probe(ens_inp, synapse=0.03)
    p_MT = nengo.Probe(MT, synapse=0.03)
    p_LIP = nengo.Probe(LIP, synapse=0.03)
    p_out = nengo.Probe(ens_out, synapse=0.03)


if __name__ == "__main__":
    from pathlib import Path

    sim_T = 6.0
    with nengo.Simulator(model) as sim:
        sim.run(sim_T)

    t = sim.trange()
    inp = sim.data[p_input]
    mt = sim.data[p_MT]
    lip = sim.data[p_LIP]
    out = sim.data[p_out]

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig = plt.figure(figsize=(10, 4.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.3)
        ax_x = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[1, 0])
        ax_phase = fig.add_subplot(gs[:, 1])

        for ax, dim, label in [(ax_x, 0, "$x$"), (ax_y, 1, "$y$")]:
            ax.axhline(inp[-1, dim], color="gray", ls=":", lw=0.8,
                        label="input target")
            ax.plot(t, mt[:, dim], color="C0", lw=0.6, alpha=0.6, label="MT")
            ax.plot(t, lip[:, dim], color="C1", lw=1.4, label="LIP")
            ax.plot(t, out[:, dim], color="C3", lw=1.4, label="output")
            ax.set_ylabel(f"{label} component")
            ax.set_xlim(0, sim_T)
            ax.set_ylim(-1.05, 1.05)
        ax_x.legend(loc="upper right", fontsize=7, ncol=4)
        ax_x.set_title("Time series — `x` (top) and `y` (bottom) components")
        ax_y.set_xlabel("time (s)")

        # Phase portrait — LIP wanders from origin toward the input direction.
        ax_phase.plot(lip[:, 0], lip[:, 1], color="C1", lw=0.7, alpha=0.85,
                       label="LIP trajectory")
        ax_phase.plot(out[:, 0], out[:, 1], color="C3", lw=0.7, alpha=0.6,
                       label="output trajectory")
        ax_phase.scatter([inp[-1, 0]], [inp[-1, 1]], color="k", marker="*",
                          s=80, zorder=5, label=f"input direction "
                          f"({inp[-1, 0]:+.1f}, {inp[-1, 1]:+.1f})")
        ax_phase.scatter([0], [0], color="gray", marker="o", s=30, zorder=4,
                          label="start (0, 0)")
        # Output-ensemble firing-threshold circle (intercepts ≥ 0.3).
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_phase.plot(0.3 * np.cos(theta), 0.3 * np.sin(theta),
                       color="k", ls="--", lw=0.6, alpha=0.5,
                       label="output threshold |x| ≈ 0.3")
        ax_phase.set_xlim(-1.05, 1.05)
        ax_phase.set_ylim(-1.05, 1.05)
        ax_phase.set_aspect("equal")
        ax_phase.set_xlabel("$x$")
        ax_phase.set_ylabel("$y$")
        ax_phase.set_title("Phase portrait — LIP integrates toward input")
        ax_phase.legend(loc="lower right", fontsize=6)

        fig.savefig(fig_dir / "integrator_dynamics.png", dpi=110,
                     bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_dir / 'integrator_dynamics.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")
