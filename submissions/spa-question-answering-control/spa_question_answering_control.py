# Question Answering with Control using SPA

# Now we will build this model again, using the spa (semantic pointer
# architecture) package built into Nengo 2.0.

# Press the play button to run the simulation.
# The top graph is the input to the visual subnet. When this input is a
# STATEMENT, there is no response shown in the motor graph and the input is
# stored in memory (shown in the memory graph).  To see bound pairs (e.g.
# RED*CIRCLE) in memory, you need to right-click on the memory graph and
# select 'show pairs'.  This can be cluttered but is more informative.
# When the input to 'visual' is a QUESTION, the motor graph shows the
# appropriate answer. For instance, when the input to visual is
# QUESTION+BLUE (showin in the visual graphs), the output from motor is SQUARE.
# Note this simulation is fairly sensitive to the dimensionality, you may
# need multiple runs or a higher dimension to get the expected results.

# Setup the environment
import nengo
import nengo.spa as spa
import numpy as np
from nengo.spa import Vocabulary

dim = 32  # The dimensionality of the vectors
rng = np.random.RandomState(11)
vocab = Vocabulary(dimensions=dim, rng=rng, max_similarity=0.1)

# Adding semantic pointers to the vocabulary
CIRCLE = vocab.parse("CIRCLE")
BLUE = vocab.parse("BLUE")
RED = vocab.parse("RED")
SQUARE = vocab.parse("SQUARE")
ZERO = vocab.add("ZERO", [0] * dim)

# Create the spa.SPA network to which we can add SPA objects
model = spa.SPA(label="Question Answering with Control", vocabs=[vocab])
with model:
    model.visual = spa.State(dim)
    model.motor = spa.State(dim)
    model.memory = spa.State(dim, feedback=1, feedback_synapse=0.1)

    actions = spa.Actions(
        "dot(visual, STATEMENT) --> memory=visual",
        "dot(visual, QUESTION) --> motor = memory * ~visual",
    )

    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)

    # Function for providing visual input
    def visual_input(t):
        if 0.1 < t < 0.3:
            return "STATEMENT+RED*CIRCLE"
        elif 0.35 < t < 0.5:
            return "STATEMENT+BLUE*SQUARE"
        elif 0.55 < t < 0.7:
            return "QUESTION+BLUE"
        elif 0.75 < t < 0.9:
            return "QUESTION+CIRCLE"
        return "ZERO"

    # Inputs
    model.input = spa.Input(visual=visual_input)

    # Probes for the headline figure (harmless for NengoGUI).
    p_visual = nengo.Probe(model.visual.output, synapse=0.03)
    p_memory = nengo.Probe(model.memory.output, synapse=0.03)
    p_motor = nengo.Probe(model.motor.output, synapse=0.03)


if __name__ == "__main__":
    from pathlib import Path

    sim_T = 1.0
    with nengo.Simulator(model) as sim:
        sim.run(sim_T)

    # Atomic vocab vectors (one-hots) and the two bound pairs that get stored.
    atomic = ["RED", "BLUE", "CIRCLE", "SQUARE", "STATEMENT", "QUESTION"]
    bound = ["RED*CIRCLE", "BLUE*SQUARE"]

    def vocab_vec(key):
        return vocab.parse(key).v

    V_atomic = np.stack([vocab_vec(k) for k in atomic])
    V_bound = np.stack([vocab_vec(k) for k in bound])
    t = sim.trange()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        sims_v = sim.data[p_visual] @ V_atomic.T
        for i, k in enumerate(atomic):
            axs[0].plot(t, sims_v[:, i], label=k, lw=0.9)
        axs[0].set_ylabel("similarity")
        axs[0].set_title("visual input")
        axs[0].legend(loc="upper right", fontsize=7, ncol=3)

        sims_m = sim.data[p_memory] @ V_bound.T
        for i, k in enumerate(bound):
            axs[1].plot(t, sims_m[:, i], label=k, lw=0.9)
        axs[1].set_ylabel("similarity")
        axs[1].set_title("memory — bound-pair contents")
        axs[1].legend(loc="upper right", fontsize=7)

        sims_mo = sim.data[p_motor] @ V_atomic[:4].T  # only colour & shape
        for i, k in enumerate(atomic[:4]):
            axs[2].plot(t, sims_mo[:, i], label=k, lw=0.9)
        axs[2].set_ylabel("similarity")
        axs[2].set_title("motor — answer to the most recent question")
        axs[2].legend(loc="upper right", fontsize=7, ncol=4)
        axs[2].set_xlabel("time (s)")
        for ax in axs:
            ax.set_xlim(0, sim_T)

        fig.tight_layout()
        fig.savefig(fig_dir / "qa_control_dynamics.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'qa_control_dynamics.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")
