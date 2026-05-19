"""
Runnable example showing how to use this submission.

Tier-1 CI runs this script and requires it to complete without error.
Keep total runtime under ~30 seconds.
"""

import nengo
import numpy as np

from my_submission import MyNetwork


def build_model():
    with nengo.Network(seed=0) as model:
        sub = MyNetwork(n_neurons=100)
        stim = nengo.Node(lambda t: np.sin(2 * np.pi * t))
        nengo.Connection(stim, sub.input)
        model.probe = nengo.Probe(sub.output, synapse=0.01)
    return model


def main():
    model = build_model()
    with nengo.Simulator(model) as sim:
        sim.run(0.5)
    print(f"Ran for {sim.trange()[-1]:.3f}s, final output: {sim.data[model.probe][-1]}")


if __name__ == "__main__":
    main()
