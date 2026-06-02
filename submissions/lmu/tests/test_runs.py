"""Tier-1 sanity tests for the LMU submission."""

import nengo
import numpy as np
import pytest

from lmu import LMU


def test_imports():
    assert LMU is not None


def test_builds():
    with nengo.Network():
        lmu = LMU(theta=0.5, order=6)
        assert lmu.input is not None
        assert lmu.state is not None
        assert lmu.A.shape == (6, 6)
        assert lmu.B.shape == (6, 1)
        assert lmu.theta == 0.5
        assert lmu.order == 6


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        LMU(theta=0.0, order=4)
    with pytest.raises(ValueError):
        LMU(theta=-0.1, order=4)
    with pytest.raises(ValueError):
        LMU(theta=1.0, order=0)


def test_runs_100ms():
    with nengo.Network() as model:
        lmu = LMU(theta=0.5, order=6)
        stim = nengo.Node(lambda t: np.sin(2 * np.pi * t))
        nengo.Connection(stim, lmu.input, synapse=None)
        p = nengo.Probe(lmu.state)

    with nengo.Simulator(model) as sim:
        sim.run(0.1)

    # 100 timesteps at dt=0.001s, 6-dim state.
    assert sim.data[p].shape == (100, 6)
