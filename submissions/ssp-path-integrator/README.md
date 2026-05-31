# ssp-path-integrator

A spiking neural path-integration network: given a continuous velocity signal, continuously updates a Spatial Semantic Pointer (SSP) estimate of position.

## Description

This is a curated NengoZoo wrapper around the path-integration subnetwork from Dumont et al. (2023). It implements continuous path integration using velocity-controlled oscillators (VCOs) with attractor dynamics: each oscillator tracks one Fourier component of the SSP encoding of position, and the velocity input modulates the rotation rate of each component. The combined output is a continuously updated SSP that can be decoded back to (x, y) coordinates.

It's the localization half of the larger SSP-SLAM system in Dumont's [Semantic-Spiking-Neural-SLAM-2023](https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023) repository — useful on its own as a biologically-grounded path integrator, or as a drop-in component for navigation / SLAM-style models that want SSP representations.

## Installation

We recommend a fresh virtual environment to avoid Nengo/NumPy version conflicts:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `sspslam` package is pulled directly from Nicole Dumont's GitHub repository (it's not on PyPI yet).

## Usage

```python
import nengo
import numpy as np
import sspslam
from ssp_path_integrator import PathIntegrator

# 1. Build an SSP space.
ssp_space = sspslam.HexagonalSSPSpace(
    domain_dim=2, n_scales=2, n_rotates=3,
    domain_bounds=np.array([[-1, 1], [-1, 1]]),
    length_scale=0.2, seed=0,
)
d = ssp_space.ssp_dim

# 2. Wire it up.
with nengo.Network() as net:
    velocity = nengo.Node(lambda t: [0.5, 0.0])
    init_ssp = nengo.Node(
        lambda t: ssp_space.encode(np.array([[0.0, 0.0]])).flatten()
                  if t < 0.05 else np.zeros(d)
    )

    pi = PathIntegrator(ssp_space, n_neurons=500)
    nengo.Connection(velocity, pi.velocity_input, synapse=None)
    nengo.Connection(init_ssp, pi.input, synapse=None)

    out = nengo.Probe(pi.output, synapse=0.05)

with nengo.Simulator(net) as sim:
    sim.run(0.5)

# Decode the SSP back to (x, y):
xy = ssp_space.decode(sim.data[out], "from-set", "grid", 50)
```

See [`examples/example_usage.py`](examples/example_usage.py) for a complete demo with a circular trajectory and a path-comparison plot.

## How it works

The SSP encoding maps a continuous position `x ∈ R^N` into a `d`-dimensional vector via the inverse FFT of complex exponentials whose phases are determined by a `phase_matrix Φ ∈ R^(d × N)`. In the Fourier domain, the SSP factorizes into independent components, each of which behaves as a 2D oscillator whose angular frequency is proportional to `Φ_k · v` (the dot product of the velocity with the `k`-th phase vector).

PathIntegrator implements each such oscillator as a 3-dimensional Nengo ensemble: two dimensions for the real and imaginary parts of the Fourier component, one dimension that receives the velocity-projected frequency input. With `stable=True`, each oscillator additionally has attractor dynamics pulling the (real, imag) state onto the unit circle, which substantially improves accuracy over a plain harmonic oscillator. A final linear transform reads the SSP back out of the oscillator array.

See Dumont et al. (2023) for the math and the broader SLAM context.

## Citation

```bibtex
@article{dumont2023exploiting,
  author  = {Dumont, P. Michaela Nicole and Furlong, P. Michael and Orchard, Jeff and Eliasmith, Chris},
  title   = {Exploiting semantic information in a spiking neural {SLAM} system},
  journal = {Frontiers in Neuroscience},
  volume  = {17},
  year    = {2023}
}
```

## License

GPLv2 for this wrapper (see `LICENSE`). The underlying `sspslam` package is distributed by Nicole Dumont under its own license (Apache-2.0 per the upstream README).
