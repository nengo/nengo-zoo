"""
Dumont et al. (2023) SSP-based path integration.

A curated NengoZoo wrapper around `sspslam.networks.PathIntegration` — the
velocity-controlled-oscillator (VCO) network that integrates a velocity
signal into a Spatial Semantic Pointer (SSP) estimate of position.

Reference
---------
Dumont, N. S.-Y., Furlong, P. M., Orchard, J., & Eliasmith, C. (2023).
Exploiting semantic information in a spiking neural SLAM system.
Frontiers in Neuroscience, 17.
"""

from __future__ import annotations

import nengo

from sspslam.networks import PathIntegration


class PathIntegrator(nengo.Network):
    """SSP-based path-integration subnetwork.

    Takes a `domain_dim`-length velocity signal and continuously updates an
    SSP representation of position. The SSP encoding is determined by the
    `ssp_space` argument (typically a `HexagonalSSPSpace` from sspslam).

    Parameters
    ----------
    ssp_space : sspslam.sspspace.SSPSpace
        Defines the SSP representation: dimensionality, domain
        dimensionality, length scale, and phase matrix. Construct with
        `sspslam.HexagonalSSPSpace(...)` or `sspslam.RandomSSPSpace(...)`.
    n_neurons : int
        Neurons per VCO population. ~400–800 typically gives good accuracy.
    recurrent_tau : float, optional
        Synapse on the recurrent VCO connections. Default 0.05.
    scaling_factor : float, optional
        Scaling applied to the velocity signal. Use when the velocity input
        has been normalized (see `examples/example_usage.py`). Default 1.0.
    stable : bool, optional
        Use non-linear attractor oscillator dynamics (True) or a plain
        simple harmonic oscillator (False). Default True.
    max_radius : float, optional
        Target radius of the attractor (only relevant when `stable=True`).
        Default 1.0.
    with_gcs : bool, optional
        If True, the output is a grid-cell-encoded ensemble; if False, a
        passthrough node. Default False.
    n_gcs : int, optional
        Neurons in the grid-cell output ensemble when `with_gcs=True`.
        Default 1000.
    solver_weights : bool, optional
        Solve full weight matrices for recurrent connections instead of
        decoders. Default False.
    label : str, optional
        Network label.
    seed : int, optional
        Network seed.
    **kwargs
        Forwarded to the underlying VCO `nengo.networks.EnsembleArray`.

    Attributes
    ----------
    velocity_input : nengo.Node
        Velocity signal input. Shape: (domain_dim,).
    input : nengo.Node
        SSP input — used to initialize or correct the integrator. Shape:
        (ssp_dim,).
    output : nengo.Node or nengo.Ensemble
        SSP estimate of current position. Shape: (ssp_dim,).
    oscillators : nengo.networks.EnsembleArray
        The VCO populations (one ensemble per Fourier component).
    """

    def __init__(
        self,
        ssp_space,
        n_neurons: int,
        recurrent_tau: float = 0.05,
        scaling_factor: float = 1.0,
        stable: bool = True,
        max_radius: float = 1.0,
        with_gcs: bool = False,
        n_gcs: int = 1000,
        solver_weights: bool = False,
        label: str | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(label=label or "PathIntegrator", seed=seed)
        with self:
            self.pi = PathIntegration(
                ssp_space,
                n_neurons,
                recurrent_tau=recurrent_tau,
                scaling_factor=scaling_factor,
                stable=stable,
                max_radius=max_radius,
                with_gcs=with_gcs,
                n_gcs=n_gcs,
                solver_weights=solver_weights,
                **kwargs,
            )
            self.velocity_input = self.pi.velocity_input
            self.input = self.pi.input
            self.output = self.pi.output
            self.oscillators = self.pi.oscillators
