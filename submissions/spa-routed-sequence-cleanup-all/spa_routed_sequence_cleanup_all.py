# Routed Sequencing with Cleanup all Memory

# The previous model on Routed Sequencing with cleanup memory used a scalar
# ensemble to project only state 'A'. In this model, you will project all the
# states onto an ensemble of neurons as the state cycles through a five element
# sequence.

# This model has parameters as described in the book. It extends the routed
# sequencing model by creating a scalar ensemble 'cleanup' and then projecting
# all the states on to this ensemble using a transformation matrix 'pd_new',
# specified by the vectors in the vocabulary.

# Press the play button to run the simulation.
# The graph in the middle shows the semantic pointer representation of the
# values stored in state and the graph on the bottom-right shows the response of
# the cleanup population. and the graph on the top-right shows the utility
# (similarity) of the current basal ganglia input (i.e., state) with the
# possible vocabulary vectors.

# Since the cleanup operation is similar to a dot product between the state and
# the defined vocabulary vectors, the value of the cleanup population in a
# particular dimension rises only when the value of state (top-right graph)
# corresponds to that particular dimension.

import nengo

# Setup the environment
import numpy as np
from nengo import spa  # import spa related packages
from nengo.spa import Vocabulary

# Number of dimensions for the Semantic Pointers
dim = 16

# Change the seed of this RNG to change the vocabulary
rng = np.random.RandomState(4)
vocab = Vocabulary(dimensions=dim, rng=rng, max_similarity=0.1)

# Make a model object with the SPA network
model = spa.SPA(label="Routed_Sequence with cleanupAll", vocabs=[vocab])

with model:
    # Specify the modules to be used
    model.state = spa.State(dimensions=dim, feedback=1, feedback_synapse=0.01)
    model.vision = spa.State(dimensions=dim)
    # Specify the action mapping
    actions = spa.Actions(
        "dot(vision, START) --> state = vision",
        "dot(state, A) --> state = B",
        "dot(state, B) --> state = C",
        "dot(state, C) --> state = D",
        "dot(state, D) --> state = E",
        "dot(state, E) --> state = A",
    )

    # Creating the BG and thalamus components that confirm to the specified rules
    model.bg = spa.BasalGanglia(actions=actions)
    model.thal = spa.Thalamus(model.bg)

    # Get vocabulary items in order of creation
    vsize = len((model.get_output_vocab("state").keys))
    vocab_items = []
    for index in range(vsize):
        vocab_items.append(model.get_output_vocab("state").keys[index])

    # Creating the transformation matrix (pd) and cleanup SPA State (cleanup)
    pd = []
    for item in vocab_items:
        pd.append([model.get_output_vocab("state")[item].v.tolist()])

    model.cleanup = nengo.Ensemble(n_neurons=300, dimensions=vsize)

    # Function that provides the model with an initial input semantic pointer.
    def start(t):
        if t < 0.4:
            return "0.8*START+D"
        return "0"

    # Input
    model.input = spa.Input(vision=start)

    # Projecting the state on to the cleanup ensemble using a transformation matrix 'pd'
    # Note that the first item in the vocabulary (`START`) is ignored.
    for i in range(1, vsize):
        nengo.Connection(model.state.output, model.cleanup[i], transform=pd[i])

    # Probes for the headline figure (harmless for NengoGUI).
    p_state = nengo.Probe(model.state.output, synapse=0.03)
    p_cleanup = nengo.Probe(model.cleanup, synapse=0.03)


if __name__ == "__main__":
    from pathlib import Path

    sim_T = 1.5
    with nengo.Simulator(model) as sim:
        sim.run(sim_T)

    # Plot similarity of state to the sequence elements A..E.
    seq = ["A", "B", "C", "D", "E"]
    V = np.stack([vocab[k].v for k in seq])
    t = sim.trange()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(__file__).resolve().parent / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

        state_sims = sim.data[p_state] @ V.T
        for i, k in enumerate(seq):
            axs[0].plot(t, state_sims[:, i], label=k, lw=0.9)
        axs[0].set_ylabel("similarity")
        axs[0].set_title("state — decoded similarity to A..E")
        axs[0].legend(loc="upper right", fontsize=7, ncol=5)
        axs[0].set_xlim(0, sim_T)

        # Cleanup ensemble: each dimension corresponds to one vocab element.
        cleanup = sim.data[p_cleanup]
        # Drop the START dim at index 0; the connections in the script also skip it.
        for i, k in enumerate(vocab_items):
            if k == "START":
                continue
            axs[1].plot(t, cleanup[:, i], label=k, lw=0.9)
        axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("cleanup activity")
        axs[1].set_title("cleanup ensemble — one dimension per vocab element")
        axs[1].legend(loc="upper right", fontsize=7, ncol=5)
        axs[1].set_xlim(0, sim_T)

        fig.tight_layout()
        fig.savefig(fig_dir / "sequence_dynamics.png", dpi=110)
        plt.close(fig)
        print(f"Saved {fig_dir / 'sequence_dynamics.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot.")
