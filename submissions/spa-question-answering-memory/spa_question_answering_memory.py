# Question Answering with Memory using SPA

# Now we will build this model again, using the spa (semantic pointer
# architecture) package built into Nengo 2.0.

# Press the play button to run the simulation.
# Graphs show the colour, shape and cue inputs. The last graph
# shows that the output is most similar to the semantic pointer which was
# initially bound to the given cue. For example, when SQUARE is
# provided as a cue, the output is most similar to BLUE.

import nengo
import nengo.spa as spa
import numpy as np
from nengo.spa import Vocabulary

dim = 32  # The dimensionality of the vectors
rng = np.random.RandomState(4)
vocab = Vocabulary(dimensions=dim, rng=rng, max_similarity=0.1)

# Adding semantic pointers to the vocabulary
CIRCLE = vocab.parse("CIRCLE")
BLUE = vocab.parse("BLUE")
RED = vocab.parse("RED")
SQUARE = vocab.parse("SQUARE")
ZERO = vocab.add("ZERO", [0] * dim)

# Create the spa.SPA network to which we can add SPA objects
model = spa.SPA(label="Question Answering with Memory", vocabs=[vocab])
with model:
    model.A = spa.State(dim)
    model.B = spa.State(dim)
    model.C = spa.State(dim)
    model.D = spa.State(dim)
    model.E = spa.State(dim)
    model.memory = spa.State(dim, feedback=1)

    actions = spa.Actions("D = A * B", "memory = D", "E = memory * ~C")

    model.cortical = spa.Cortical(actions)

    # Function for providing color input
    def color_input(t):
        if t < 0.25:
            return "RED"
        elif t < 0.5:
            return "BLUE"
        return "ZERO"

    # Function for providing shape input
    def shape_input(t):
        if t < 0.25:
            return "CIRCLE"
        elif t < 0.5:
            return "SQUARE"
        return "ZERO"

    # Function for providing the cue
    def cue_input(t):
        if t < 0.5:
            return "ZERO"
        sequence = ["ZERO", "CIRCLE", "RED", "ZERO", "SQUARE", "BLUE"]
        idx = int(((t - 0.5) // (1.0 / len(sequence))) % len(sequence))
        return sequence[idx]

    # Inputs
    model.input = spa.Input(A=color_input, B=shape_input, C=cue_input)

    # Probes for the headline figure (harmless for NengoGUI).
    p_A = nengo.Probe(model.A.output, synapse=0.03)
    p_B = nengo.Probe(model.B.output, synapse=0.03)
    p_C = nengo.Probe(model.C.output, synapse=0.03)
    p_memory = nengo.Probe(model.memory.output, synapse=0.03)
    p_E = nengo.Probe(model.E.output, synapse=0.03)


if __name__ == "__main__":
    from pathlib import Path

    sim_T = 2.0
    with nengo.Simulator(model) as sim:
        sim.run(sim_T)

    atomic = ["RED", "BLUE", "CIRCLE", "SQUARE"]
    bound = ["RED*CIRCLE", "BLUE*SQUARE"]

    V_atomic = np.stack([vocab.parse(k).v for k in atomic])
    V_bound = np.stack([vocab.parse(k).v for k in bound])
    t = sim.trange()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axs = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
        panels = [
            (p_A, atomic, V_atomic, "A — colour input"),
            (p_B, atomic, V_atomic, "B — shape input"),
            (p_C, atomic, V_atomic, "C — cue"),
            (p_memory, bound, V_bound, "memory — bound-pair contents"),
            (p_E, atomic, V_atomic, "E — recovered answer"),
        ]
        for ax, (probe, keys, V, title) in zip(axs, panels):
            sims = sim.data[probe] @ V.T
            for i, k in enumerate(keys):
                ax.plot(t, sims[:, i], label=k, lw=0.9)
            ax.set_ylabel("similarity")
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=7, ncol=len(keys))
            ax.set_xlim(0, sim_T)
        axs[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(fig_dir / "qa_memory_dynamics.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'qa_memory_dynamics.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")
