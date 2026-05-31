"""Tier-1 sanity tests for the ssp-path-integrator submission."""

import numpy as np
import nengo

import sspslam
from ssp_path_integrator import PathIntegrator


def _small_ssp_space(seed=0):
    return sspslam.HexagonalSSPSpace(
        domain_dim=2,
        n_scales=1,
        n_rotates=1,
        domain_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        length_scale=0.2,
        seed=seed,
    )


def test_imports():
    assert PathIntegrator is not None


def test_builds():
    ssp_space = _small_ssp_space()
    with nengo.Network():
        pi = PathIntegrator(ssp_space, n_neurons=50)
        assert pi.velocity_input is not None
        assert pi.input is not None
        assert pi.output is not None
        assert pi.oscillators is not None


def test_runs_100ms():
    ssp_space = _small_ssp_space()
    d = ssp_space.ssp_dim
    with nengo.Network() as model:
        pi = PathIntegrator(ssp_space, n_neurons=50)
        vel = nengo.Node([0.5, 0.0])
        nengo.Connection(vel, pi.velocity_input, synapse=None)
        probe = nengo.Probe(pi.output, synapse=0.05)

    with nengo.Simulator(model) as sim:
        sim.run(0.1)

    # 100 timesteps at dt=0.001s.
    assert sim.data[probe].shape == (100, d)
