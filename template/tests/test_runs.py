"""
Minimum-viable test for Tier-1 CI: imports, builds, runs 100ms.

Add more tests below for behavior verification — e.g. asserting expected
output shapes or values for specific inputs.
"""

import nengo

from my_submission import MyNetwork


def test_imports():
    assert MyNetwork is not None


def test_builds_and_runs():
    with nengo.Network() as model:
        sub = MyNetwork(n_neurons=50)
        probe = nengo.Probe(sub.output)

    with nengo.Simulator(model) as sim:
        sim.run(0.1)

    # Sanity: probe got 100 timesteps of data.
    assert sim.data[probe].shape[0] == 100
