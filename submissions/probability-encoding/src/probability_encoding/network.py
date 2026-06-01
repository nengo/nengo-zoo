"""
Furlong probability-encoding network.

A spiking-neural population that estimates a probability density from
samples using Spatial Semantic Pointers and the Function Inverse
Estimator (FIE) normalization.

Background
----------
The notebook this is curated from
(`Probability Encodings - Spiking Implementation.ipynb`,
ctn-waterloo/summerschool2025) demonstrates two variants on the same
spiking population:

  * Single-neuron — the neuron's encoder is set to `mu` (the mean SSP of
    the training samples). Sweeping a query SSP over the domain produces
    a spike train whose firing rate traces the PDF.
  * Population — one neuron per query point, with encoders set to the
    SSPs of those queries. Driving the population with the constant `mu`
    vector gives a per-neuron firing rate that approximates p(query[i]).

Both variants share the same ensemble construction; only the `encoders`
matrix differs. This wrapper exposes that one knob via the `encoders`
argument.
"""

from __future__ import annotations

import nengo
import numpy as np

from . import fie_util


class ProbabilityEncoder(nengo.Network):
    """Spiking probability-density estimator with SSP encoding.

    Computes `mu` (mean SSP of `training_samples`) and `xi` (FIE
    normalization on `domain_bounds`), and builds a spiking ensemble
    whose firing rate at the SSP-encoded query x_q approximates p(x_q):

        rate(x_q) ~= max_rate * max(0, mu . phi(x_q) - xi)

    Parameters
    ----------
    ssp_space : sspspace.SSPSpace
        From ctn-waterloo/ssp-bayesopt. The lengthscale is updated in
        place to the fit bandwidth before encoding (see Notes).
    training_samples : array-like, shape (n_samples, 1)
        Samples drawn from the target 1-D distribution.
    domain_bounds : (float, float)
        (low, high) bounds for the FIE normalization integral.
    query_points : array-like, shape (n_query, domain_dim) or None, optional
        If given, builds the population mode: one neuron per query point,
        with encoders set to `ssp_space.encode(query_points)` after the
        lengthscale has been fit. Drive the network with `mu` to read out
        p(query_points) from the per-neuron firing rates.
    encoders : np.ndarray, shape (n_neurons, ssp_dim) or None, optional
        Lower-level alternative to `query_points` — pass pre-computed
        encoder rows directly. Mutually exclusive with `query_points`.
        If neither is given, defaults to single-neuron mode with
        encoder = mu (suitable for swept-input demos).
    max_rate : float, optional
        Per-neuron gain. Default 50.0.
    bandwidth : 'ecf', 'silverman', or float, optional
        Lengthscale selection method (or a hard-set value). Default
        'ecf' — Furlong's empirical characteristic-function estimator
        (initialised from Silverman).
    neuron_type : nengo.NeuronType, optional
        Defaults to `SpikingRectifiedLinear()`.
    label, seed
        Forwarded to `nengo.Network`.

    Attributes
    ----------
    mu : np.ndarray, shape (ssp_dim,)
        Mean SSP of training samples (divided by bandwidth).
    xi : float
        FIE normalization constant.
    bandwidth : float
        The lengthscale used (and written back to `ssp_space`).
    input : nengo.Node
        Connect upstream stimulation here. Shape (ssp_dim,).
    ensemble : nengo.Ensemble
        The spiking population.

    Notes
    -----
    `ssp_space.update_lengthscale(...)` is called in place. Construct a
    fresh `ssp_space` if you need different lengthscales for different
    distributions in the same script.
    """

    def __init__(
        self,
        ssp_space,
        training_samples,
        domain_bounds,
        query_points=None,
        encoders=None,
        max_rate: float = 50.0,
        bandwidth="ecf",
        neuron_type=None,
        label: str | None = None,
        seed: int | None = None,
    ):
        super().__init__(label=label or "ProbabilityEncoder", seed=seed)

        if query_points is not None and encoders is not None:
            raise ValueError("Specify at most one of `query_points` or `encoders`.")

        training_samples = np.asarray(training_samples).reshape(-1, 1)
        low, high = domain_bounds

        # 1. Bandwidth selection.
        if bandwidth == "silverman":
            ls = float(np.atleast_1d(fie_util.bandwidth_silverman(training_samples))[0])
        elif bandwidth == "ecf":
            init_h = fie_util.bandwidth_silverman(training_samples)
            ls = fie_util.bandwidth_ecf(training_samples, init_h=init_h) * np.pi
            ls = float(np.atleast_1d(ls)[0])
        else:
            ls = float(bandwidth)
        if ls <= 0:
            raise ValueError(f"Computed bandwidth must be positive, got {ls}.")
        self.bandwidth = ls

        # 2. Apply lengthscale to the SSP space (in place).
        ssp_space.update_lengthscale(ls)
        self.ssp_space = ssp_space

        # 3. Compute mu and xi.
        phis = ssp_space.encode(training_samples)
        self.mu = np.mean(phis, axis=0) / ls
        xi_val, _ = fie_util.fit_dist(training_samples, low, high, ls=ls)
        self.xi = float(np.atleast_1d(xi_val)[0])

        # 4. Pick encoders. Note: `query_points` is encoded here, AFTER
        # update_lengthscale, so the SSPs match the trained mu/xi.
        if query_points is not None:
            query_points = np.asarray(query_points).reshape(-1, 1)
            E = ssp_space.encode(query_points)
        elif encoders is not None:
            E = np.atleast_2d(encoders)
            if E.shape[1] != ssp_space.ssp_dim:
                raise ValueError(
                    f"`encoders` columns ({E.shape[1]}) must match "
                    f"ssp_space.ssp_dim ({ssp_space.ssp_dim})."
                )
        else:
            E = np.atleast_2d(self.mu)
        n_neurons = E.shape[0]

        if neuron_type is None:
            neuron_type = nengo.SpikingRectifiedLinear()

        # 5. Build the network.
        with self:
            self.input = nengo.Node(size_in=ssp_space.ssp_dim, label="input")
            self.ensemble = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=ssp_space.ssp_dim,
                encoders=E,
                gain=max_rate * np.ones(n_neurons),
                bias=-self.xi * max_rate * np.ones(n_neurons),
                neuron_type=neuron_type,
                normalize_encoders=False,
                label="ensemble",
            )
            nengo.Connection(self.input, self.ensemble, synapse=None)
