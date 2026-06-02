# Controlled Oscillator
#
# The same 2D oscillator dynamics as a plain oscillator, but with a
# third recurrent dimension `x[2]` that gates the angular speed `s`.
# A separate `speed` ensemble feeds that dimension via a slider, so the
# oscillation can be sped up, slowed down, or stopped by another neural
# population.
#
# Equations:
#   dx0/dt = -x1 * s + x0 * (r - x0**2 - x1**2)
#   dx1/dt =  x0 * s + x1 * (r - x0**2 - x1**2)
# with r = 1 and s = 10 * x[2].

import nengo

model = nengo.Network()
with model:

    x = nengo.Ensemble(n_neurons=400, dimensions=3)

    synapse = 0.1

    def oscillator(x):
        r = 1
        s = 10 * x[2]
        return [
            synapse * -x[1] * s + x[0] * (r - x[0] ** 2 - x[1] ** 2) + x[0],
            synapse * x[0] * s + x[1] * (r - x[0] ** 2 - x[1] ** 2) + x[1],
        ]

    nengo.Connection(x, x[:2], synapse=synapse, function=oscillator)

    stim_speed = nengo.Node(0)
    speed = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim_speed, speed)
    nengo.Connection(speed, x[2])

    p_x = nengo.Probe(x, synapse=0.03)
    p_speed = nengo.Probe(stim_speed)


if __name__ == "__main__":
    # Drive the speed input through a four-segment program and save a
    # figure showing how the oscillator's frequency tracks it.
    from pathlib import Path

    def speed_program(t):
        if t < 3.0:
            return 0.3          # slow forward
        if t < 7.0:
            return 0.6          # spin up
        if t < 11.0:
            return 0.0          # halt
        return -0.6             # reverse

    stim_speed.output = speed_program

    with nengo.Simulator(model) as sim:
        sim.run(14.0)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        traj = sim.data[p_x]
        speeds = sim.data[p_speed]
        t = sim.trange()

        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.3)
        ax_speed = fig.add_subplot(gs[0, 0])
        ax_state = fig.add_subplot(gs[1, 0])
        ax_phase = fig.add_subplot(gs[:, 1])

        ax_speed.plot(t, speeds, color="k", lw=1.0)
        ax_speed.set_ylabel("speed input")
        ax_speed.set_title("Speed program — +0.3 → +0.6 → 0 → −0.6")
        ax_speed.set_xlim(0, t[-1])

        ax_state.plot(t, traj[:, 0], lw=0.6, label="$x_0$")
        ax_state.plot(t, traj[:, 1], lw=0.6, label="$x_1$")
        ax_state.set_xlabel("time (s)")
        ax_state.set_ylabel("oscillator state")
        ax_state.set_title("Oscillator state vs. time")
        ax_state.legend(loc="upper right", fontsize=8)
        ax_state.set_xlim(0, t[-1])

        # Phase portrait, segments coloured by speed regime.
        bounds = [0.0, 3.0, 7.0, 11.0, t[-1]]
        labels = ["slow forward (s=+0.3)", "spin up (s=+0.6)", "halt", "reverse (s=−0.6)"]
        colors = ["#7fbfff", "#1f77b4", "#888888", "#d62728"]
        for lo, hi, lab, col in zip(bounds[:-1], bounds[1:], labels, colors):
            m = (t >= lo) & (t < hi)
            ax_phase.plot(traj[m, 0], traj[m, 1], color=col, lw=0.8, label=lab)
        ax_phase.set_xlabel("$x_0$")
        ax_phase.set_ylabel("$x_1$")
        ax_phase.set_title("Phase portrait")
        ax_phase.set_xlim(-1.3, 1.3)
        ax_phase.set_ylim(-1.3, 1.3)
        ax_phase.set_aspect("equal")
        ax_phase.legend(loc="lower center", fontsize=7)

        fig.savefig(fig_dir / "speed_control.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_dir / 'speed_control.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")
