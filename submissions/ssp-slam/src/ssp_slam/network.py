"""
Dumont et al. (2023) SSP-based SLAM network.

A curated NengoZoo wrapper around `sspslam.networks.SLAMNetwork` — the
spiking neural SLAM system that integrates velocity into a Spatial
Semantic Pointer estimate of self-position, corrects the estimate on
landmark sightings, and learns a semantic associative memory mapping
landmark identities to SSP locations.

The wrapper exposes the upstream network's named ports as clearly-named
attributes and forwards a curated subset of constructor parameters so the
class is drop-in for downstream models without depending on the full
17-parameter upstream signature.

Reference
---------
Dumont, N. S.-Y., Furlong, P. M., Orchard, J., & Eliasmith, C. (2023).
Exploiting semantic information in a spiking neural SLAM system.
Frontiers in Neuroscience, 17.
"""

from __future__ import annotations

import nengo

from sspslam.networks import SLAMNetwork


class SSPSlam(nengo.Network):
    """SSP-based spiking SLAM subnetwork.

    Drives a velocity-controlled-oscillator path integrator, snaps the
    SSP estimate of self-position on landmark sightings, and learns a
    Voja+PES associative memory that pairs landmark identities (semantic
    pointers) with SSP locations.

    Parameters
    ----------
    ssp_space : sspslam.HexagonalSSPSpace
        SSP encoding of the agent's spatial domain (typically 2-D).
    lm_space : sspslam.SPSpace
        Semantic-pointer space of landmark identities.
    n_landmarks : int
        Number of landmark identities (items + walls).
    view_rad : float, optional
        Detection radius for landmarks (used only by the input plumbing
        you wire up; the network itself takes pre-computed visibility
        signals on its inputs). Default 0.3.
    pi_n_neurons : int, optional
        Neurons per VCO population in the internal path integrator.
        Default 250.
    mem_n_neurons : int, optional
        Neurons in the associative memory. If `None`, defaults to
        `10 * ssp_space.ssp_dim`.
    circconv_n_neurons : int, optional
        Neurons per circular-convolution ensemble. Default 100.
    vel_scaling_factor : float, optional
        Velocity-input normalisation. Default 1.0.
    tau_pi : float, optional
        Synapse on path-integrator recurrent connections. Default 0.05.
    update_thres : float, optional
        Position-update threshold for memory writes. Default 0.2.
    shift_rate : float, optional
        Position-correction shift rate. Default 0.1.
    voja_learning_rate : float, optional
        Voja learning rate (encoder updates in the memory). Default 5e-4.
    pes_learning_rate : float, optional
        PES learning rate (decoder updates in the memory). Default 5e-3.
    clean_up_method : str, optional
        Cleanup strategy (see sspslam). Default 'grid'.
    seed : int, optional
        Forwarded to the underlying SLAMNetwork.
    label : str, optional
        Network label.
    **kwargs
        Forwarded to the underlying `SLAMNetwork`.

    Attributes
    ----------
    velocity_input : nengo.Node
        Velocity signal input (shape `(domain_dim,)`).
    landmark_vec_ssp : nengo.Node
        SSP-encoded vector from agent to currently-visible landmark.
    landmark_id_input : nengo.Node
        Semantic-pointer identity of the currently-visible landmark
        (zero vector when no landmark is in view).
    no_landmark_in_view : nengo.Node
        Inhibitory gate: 10 when no landmark is in view, 0 when one is
        (used to disable memory writes when nothing is observed).
    pathintegrator : nengo.Network
        Internal `PathIntegration` subnetwork. `.input` accepts an SSP
        initialisation/correction; `.output` is the PI's SSP estimate of
        self-position.
    position_estimate : nengo.Network
        Circular-convolution subnetwork producing a SLAM-corrected SSP
        estimate of self-position (from memory + current sighting).
    assomemory : nengo.Network
        Voja+PES associative memory linking landmark IDs to SSPs.
    """

    def __init__(
        self,
        ssp_space,
        lm_space,
        n_landmarks: int,
        view_rad: float = 0.3,
        pi_n_neurons: int = 250,
        mem_n_neurons: int | None = None,
        circconv_n_neurons: int = 100,
        vel_scaling_factor: float = 1.0,
        tau_pi: float = 0.05,
        update_thres: float = 0.2,
        shift_rate: float = 0.1,
        voja_learning_rate: float = 5e-4,
        pes_learning_rate: float = 5e-3,
        clean_up_method: str = "grid",
        seed: int = 0,
        label: str | None = None,
        **kwargs,
    ):
        super().__init__(label=label or "SSPSlam", seed=seed)
        d = ssp_space.ssp_dim
        if mem_n_neurons is None:
            mem_n_neurons = 10 * d

        with self:
            self.slam = SLAMNetwork(
                ssp_space, lm_space, view_rad, n_landmarks,
                pi_n_neurons, mem_n_neurons, circconv_n_neurons,
                tau_pi=tau_pi,
                update_thres=update_thres,
                vel_scaling_factor=vel_scaling_factor,
                shift_rate=shift_rate,
                voja_learning_rate=voja_learning_rate,
                pes_learning_rate=pes_learning_rate,
                clean_up_method=clean_up_method,
                seed=seed,
                **kwargs,
            )
            self.velocity_input = self.slam.velocity_input
            self.landmark_vec_ssp = self.slam.landmark_vec_ssp
            self.landmark_id_input = self.slam.landmark_id_input
            self.no_landmark_in_view = self.slam.no_landmark_in_view
            self.pathintegrator = self.slam.pathintegrator
            self.position_estimate = self.slam.position_estimate
            self.assomemory = self.slam.assomemory
