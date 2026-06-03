"""
NengoGUI demo: a spiking population whose firing rates trace the
density of a 1-D Gaussian mixture model.

This wraps cell 22 of Furlong's `Probability Encodings — Spiking
Implementation` notebook in the v0.2.0 `ProbabilityEncoder` API. A
2-component GMM is sampled offline; `ProbabilityEncoder` is constructed
in *population* mode (one neuron per query point on the input domain),
so that when the ensemble is driven by the trained mean SSP `mu`, each
neuron's firing rate approximates the GMM density at the query point
its encoder represents.

To open in NengoGUI:

    pip install nengo-gui
    PYTHONPATH=src nengo examples/nengogui/gmm_population.py

The interesting probe is `pop.ensemble.neurons` — averaging the spike
trains over a short window reproduces the bimodal density. The constant
input `mu` and the (also constant) `density estimate` decoded value are
exposed as `nengo.Node`s for easy plotting.
"""

import nengo
import numpy as np
from ssp_bayes_opt import sspspace

from probability_encoding import ProbabilityEncoder


SEED = 0
SSP_DIM = 128
N_SAMPLES = 500
N_QUERY = 60
DOMAIN = (-6.0, 6.0)


def gmm_samples(n, rng):
    """Two-component 1-D GMM: 60% N(-2, 1.0), 40% N(+2, 1.2)."""
    coin = rng.random(n) < 0.6
    return np.where(
        coin, rng.normal(-2.0, 1.0, n), rng.normal(+2.0, 1.2, n)
    ).reshape(-1, 1)


# Training data and the SSP space. Computed at module load time so that
# `model` (below) can be picked up by NengoGUI at the top level.
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
samples = gmm_samples(N_SAMPLES, rng)
ssp_space = sspspace.RandomSSPSpace(ssp_dim=SSP_DIM, domain_dim=1)
query_xs = np.linspace(*DOMAIN, N_QUERY).reshape(-1, 1)

model = nengo.Network(seed=SEED)
with model:
    pop = ProbabilityEncoder(
        ssp_space=ssp_space,
        training_samples=samples,
        domain_bounds=DOMAIN,
        query_points=query_xs,
        max_rate=50.0,
    )
    mu_input = nengo.Node(lambda t: pop.mu, label="mu")
    nengo.Connection(mu_input, pop.input, synapse=None)
