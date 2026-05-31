"""Tier-1 sanity tests for the probability-encoding submission."""

import nengo
import numpy as np
from scipy.stats import beta as beta_dist

from ssp_bayes_opt import sspspace

from probability_encoding import ProbabilityEncoder


def _samples_and_space(ssp_dim=64, n_samples=100):
    np.random.seed(0)
    rng = np.random.default_rng(0)
    samples = beta_dist.rvs(2, 5, size=n_samples, random_state=rng).reshape(-1, 1)
    ssp_space = sspspace.RandomSSPSpace(ssp_dim=ssp_dim, domain_dim=1)
    return samples, ssp_space


def test_imports():
    assert ProbabilityEncoder is not None


def test_builds_single_neuron():
    samples, ssp_space = _samples_and_space()
    enc = ProbabilityEncoder(
        ssp_space=ssp_space,
        training_samples=samples,
        domain_bounds=(0.0, 1.0),
    )
    assert enc.ensemble.n_neurons == 1
    assert enc.mu.shape == (ssp_space.ssp_dim,)
    assert isinstance(enc.xi, float)


def test_builds_population():
    samples, ssp_space = _samples_and_space()
    query_xs = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    enc = ProbabilityEncoder(
        ssp_space=ssp_space,
        training_samples=samples,
        domain_bounds=(0.0, 1.0),
        query_points=query_xs,
    )
    assert enc.ensemble.n_neurons == 20


def test_rejects_overspecified_encoders():
    import pytest
    samples, ssp_space = _samples_and_space()
    with pytest.raises(ValueError):
        ProbabilityEncoder(
            ssp_space=ssp_space,
            training_samples=samples,
            domain_bounds=(0.0, 1.0),
            query_points=np.array([[0.5]]),
            encoders=np.ones((1, ssp_space.ssp_dim)),
        )


def test_runs_100ms():
    samples, ssp_space = _samples_and_space()
    with nengo.Network() as model:
        enc = ProbabilityEncoder(
            ssp_space=ssp_space,
            training_samples=samples,
            domain_bounds=(0.0, 1.0),
        )
        stim = nengo.Node(lambda t: enc.mu)
        nengo.Connection(stim, enc.input, synapse=None)
        probe = nengo.Probe(enc.ensemble.neurons)

    with nengo.Simulator(model) as sim:
        sim.run(0.1)
    assert sim.data[probe].shape == (100, 1)
