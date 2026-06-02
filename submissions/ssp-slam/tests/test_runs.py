"""Tier-1 sanity tests for the ssp-slam submission."""

import nengo
import numpy as np
import sspslam

from ssp_slam import SSPSlam


def _small_setup():
    """Tiny SSP + landmark spaces for fast build/run."""
    ssp_space = sspslam.HexagonalSSPSpace(
        domain_dim=2, n_scales=1, n_rotates=1,
        domain_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        length_scale=0.3, seed=0,
    )
    d = ssp_space.ssp_dim
    landmark_sps = nengo.dists.UniformHypersphere(surface=True).sample(
        2, d, rng=np.random.RandomState(0))
    lm_space = sspslam.SPSpace(2, d, seed=0, vectors=landmark_sps)
    return ssp_space, lm_space


def test_imports():
    assert SSPSlam is not None


def test_builds():
    ssp_space, lm_space = _small_setup()
    with nengo.Network():
        slam = SSPSlam(
            ssp_space=ssp_space, lm_space=lm_space, n_landmarks=2,
            pi_n_neurons=50, mem_n_neurons=4 * ssp_space.ssp_dim,
            circconv_n_neurons=30,
        )
        assert slam.velocity_input is not None
        assert slam.landmark_vec_ssp is not None
        assert slam.landmark_id_input is not None
        assert slam.no_landmark_in_view is not None
        assert slam.pathintegrator is not None
        assert slam.position_estimate is not None
        assert slam.assomemory is not None


def test_runs_100ms():
    ssp_space, lm_space = _small_setup()
    d = ssp_space.ssp_dim
    with nengo.Network() as model:
        slam = SSPSlam(
            ssp_space=ssp_space, lm_space=lm_space, n_landmarks=2,
            pi_n_neurons=50, mem_n_neurons=4 * d, circconv_n_neurons=30,
        )
        vel = nengo.Node([0.1, 0.0])
        nengo.Connection(vel, slam.velocity_input, synapse=None)
        # Drive landmark inputs with zeros + "no landmark in view" gate on
        # so the memory writes are suppressed (a minimal sanity run).
        nengo.Connection(nengo.Node([0] * d), slam.landmark_vec_ssp, synapse=None)
        nengo.Connection(nengo.Node([0] * d), slam.landmark_id_input, synapse=None)
        nengo.Connection(nengo.Node([10]), slam.no_landmark_in_view, synapse=None)
        probe = nengo.Probe(slam.pathintegrator.output, synapse=0.05)

    with nengo.Simulator(model) as sim:
        sim.run(0.1)
    assert sim.data[probe].shape == (100, d)
