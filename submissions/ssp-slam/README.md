# ssp-slam

A spiking neural SLAM subnetwork: continuously integrate velocity into a Spatial Semantic Pointer (SSP) estimate of self-position, correct the estimate on landmark sightings, and learn a semantic associative memory mapping landmark identities to SSP locations.

## Description

This is a curated NengoZoo wrapper around the SLAM network from Dumont et al. (2023). It composes three pieces:

1. **Path integrator** — a velocity-controlled-oscillator network maintaining a continuous SSP estimate of self-position. (Same algorithm as the [`ssp-path-integrator`](../ssp-path-integrator/) submission, used internally here.)
2. **Position correction** — on each landmark sighting, the network shifts its self-position SSP toward `landmark_position − vector_to_landmark`, snapping the estimate back to ground truth and arresting integrator drift.
3. **Associative memory** — a Voja+PES learning loop pairs landmark identities (semantic pointers) with the corresponding SSP locations. After training, semantic queries like "blue triangle" or "all blues" return SSP similarity maps over the domain.

The wrapper exposes the upstream `SLAMNetwork`'s ports as clean attributes (`velocity_input`, `landmark_vec_ssp`, `landmark_id_input`, `no_landmark_in_view`, plus `pathintegrator`, `position_estimate`, and `assomemory` for downstream probing). It's intended as a drop-in component for navigation models that want both a corrected self-position estimate and a queryable semantic map.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`sspslam` is pulled from Nicole Dumont's GitHub repository (not on PyPI). The `nengo-loihi` line is a workaround — `sspslam` hard-imports it through its `utils.network_diagram` even when you don't use Loihi.

## Usage

```python
import nengo
import numpy as np
import sspslam
from ssp_slam import SSPSlam

ssp_space = sspslam.HexagonalSSPSpace(
    domain_dim=2, n_scales=2, n_rotates=3,
    domain_bounds=np.array([[-1, 1], [-1, 1]]),
    length_scale=0.3, seed=0,
)
d = ssp_space.ssp_dim

# Build your landmark SP space (n_landmarks, d).
landmark_sps = np.random.randn(4, d); landmark_sps /= np.linalg.norm(landmark_sps, axis=1, keepdims=True)
lm_space = sspslam.SPSpace(4, d, seed=0, vectors=landmark_sps)

with nengo.Network() as model:
    slam = SSPSlam(
        ssp_space=ssp_space, lm_space=lm_space, n_landmarks=4,
        view_rad=0.3,
        pi_n_neurons=500,
        mem_n_neurons=10 * d,
        circconv_n_neurons=100,
        vel_scaling_factor=vel_scaling_factor,  # see example
    )

    nengo.Connection(velocity_node, slam.velocity_input, synapse=None)
    nengo.Connection(init_ssp_node, slam.pathintegrator.input, synapse=None)
    nengo.Connection(landmark_vec_node, slam.landmark_vec_ssp, synapse=None)
    nengo.Connection(landmark_id_node, slam.landmark_id_input, synapse=None)
    nengo.Connection(no_view_gate_node, slam.no_landmark_in_view, synapse=None)

    self_ssp = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
```

See [`examples/example_usage.py`](examples/example_usage.py) for the full toy-environment demo (path generation, environment with objects + a wall, semantic memory queries).

## How it works

The path-integrator half is the same as in [`ssp-path-integrator`](../ssp-path-integrator/): velocity is projected onto each of the SSP's Fourier components, and a recurrent oscillator (with attractor dynamics) tracks each component's phase over time.

On top of that, two additional pathways:

- **Landmark binding.** When the agent is within `view_rad` of a landmark, an upstream input function provides (i) the landmark's semantic-pointer ID and (ii) the SSP-encoded vector from agent to landmark. Circularly convolving the inverse of the current self-SSP with the vector-to-landmark SSP gives the SSP-encoded landmark location; the corrected self-SSP is then this location bound with the inverse vector-to-landmark.
- **Semantic memory.** A Voja+PES learning loop trains an associative memory from landmark IDs (semantic pointers) onto SSP locations. After enough sightings, querying the memory with a semantic term like `bind(triangle, sum(color_sps))` returns an SSP whose similarity map over the domain peaks at every triangular landmark — a single neural circuit performs semantic-conditioned spatial recall.

For the math, see Dumont et al. (2023).

## Citation

```bibtex
@article{dumont2023exploiting,
  author  = {Dumont, P. Michaela Nicole and Furlong, P. Michael and Orchard, Jeff and Eliasmith, Chris},
  title   = {Exploiting semantic information in a spiking neural {SLAM} system},
  journal = {Frontiers in Neuroscience},
  volume  = {17},
  year    = {2023},
  doi     = {10.3389/fnins.2023.1190515}
}
```

## License

GPLv2 for this wrapper (see `LICENSE`). The underlying `sspslam` package is distributed by Nicole Dumont under its own license (Apache-2.0 per the upstream README).
