"""Tier-1 sanity tests for the basal-ganglia submission."""

import nengo
import numpy as np

from basal_ganglia import BasalGanglia


def test_imports():
    assert BasalGanglia is not None


def test_builds():
    with nengo.Network() as model:
        bg = BasalGanglia(dimensions=3, n_neurons_per_ensemble=50)
        assert bg.input is not None
        assert bg.output is not None


def test_runs_100ms():
    with nengo.Network() as model:
        bg = BasalGanglia(dimensions=3, n_neurons_per_ensemble=50)
        stim = nengo.Node([0.6, 0.3, 0.1])
        nengo.Connection(stim, bg.input)
        probe = nengo.Probe(bg.output, synapse=0.01)

    with nengo.Simulator(model) as sim:
        sim.run(0.1)

    # 100 timesteps at dt=0.001s, 3 output dimensions.
    assert sim.data[probe].shape == (100, 3)


def test_selects_highest_utility():
    """After settling, action 0 should be the least-inhibited output."""
    with nengo.Network(seed=0) as model:
        bg = BasalGanglia(dimensions=3, n_neurons_per_ensemble=100)
        stim = nengo.Node([0.8, 0.4, 0.2])
        nengo.Connection(stim, bg.input)
        probe = nengo.Probe(bg.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.3)

    # Skip the transient; check the second half.
    final = sim.data[probe][-100:].mean(axis=0)
    assert int(np.argmax(final)) == 0, f"expected action 0 selected, got {final}"


def test_rejects_zero_dimensions():
    import pytest
    with pytest.raises(ValueError):
        BasalGanglia(dimensions=0)
