# Tutorial 15: The Lorenz Chaotic Attractor

# Differential equations can also give chaotic behaviour.  The classic example
# of this is the Lorenz "butterfly" attractor.  The equations for it are
#
# dx0/dt = sigma * (x1 - x0)
# dx1/dt = - x0 * x2 - x1
# dx2/dt = x0 * x1 - beta * (x2 + rho) - rho
#
# Note: this is a slight transformation from the standard formulation so
#  as to centre the value around the origin. For further information, see
#  http://compneuro.uwaterloo.ca/publications/eliasmith2005b.html
#  "Chris Eliasmith. A unified approach to building and controlling
#   spiking attractor networks. Neural computation, 7(6):1276-1314, 2005."

# Since there are three dimensions, we can show three different XY plots
# combining the different values in different ways.

import nengo

model = nengo.Network(seed=3)
with model:

    x = nengo.Ensemble(n_neurons=600, dimensions=3, radius=30)

    synapse = 0.1

    def lorenz(x):
        sigma = 10
        beta = 8.0 / 3
        rho = 28

        dx0 = -sigma * x[0] + sigma * x[1]
        dx1 = -x[0] * x[2] - x[1]
        dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho

        return [dx0 * synapse + x[0], dx1 * synapse + x[1], dx2 * synapse + x[2]]

    nengo.Connection(x, x, synapse=synapse, function=lorenz)
    p_x = nengo.Probe(x, synapse=0.01)


if __name__ == "__main__":
    # Run the model and save the headline butterfly plot. Idempotent —
    # safe to re-run; just overwrites figures/butterfly.png.
    from pathlib import Path

    with nengo.Simulator(model) as sim:
        sim.run(10.0)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        traj = sim.data[p_x]
        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
        # Canonical "butterfly" view: x0 vs x2.
        axs[0].plot(traj[:, 0], traj[:, 2], color="k", lw=0.4)
        axs[0].set_xlabel("$x_0$")
        axs[0].set_ylabel("$x_2$")
        axs[0].set_title("Butterfly view ($x_0$ vs $x_2$)")
        axs[0].set_aspect("equal", adjustable="datalim")

        # State variables over time so spike noise is visible.
        t = sim.trange()
        axs[1].plot(t, traj[:, 0], lw=0.5, label="$x_0$")
        axs[1].plot(t, traj[:, 1], lw=0.5, label="$x_1$")
        axs[1].plot(t, traj[:, 2], lw=0.5, label="$x_2$")
        axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("state")
        axs[1].set_title("State variables over time")
        axs[1].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(fig_dir / "butterfly.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'butterfly.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")
