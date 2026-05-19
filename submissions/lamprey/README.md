# lamprey

A spiking model of lamprey swimming locomotion. A 3D central pattern generator (CPG) drives 10D muscle tensions through a basis-function decoding, reproducing the traveling-wave undulation pattern of real lampreys.

## Description

This is the lamprey control module from the Neural Engineering Framework (Eliasmith & Anderson, 2003), implemented in modern Nengo. Originally written by Michael Furlong and Chris Eliasmith at the 2019 Nengo Summer School and migrated to the NengoZoo from `nengo/nengo-examples`.

The model has three pieces:

1. **CPG ensemble** — 500 spiking neurons representing a 3D state, with recurrent dynamics shaped as a damped harmonic oscillator at ~3 Hz. A brief 1-second "kick" stimulus pushes the state off the origin; the dynamics then sustain a stable limit-cycle oscillation indefinitely.
2. **Tension decoding** — the 3D CPG state is mapped to 10 muscle-tension values along the lamprey's body via a basis-function decoding (`T(x)`). Adjacent body segments are phase-shifted, producing the traveling wave.
3. **Body** — a `Lamprey` node consumes the 10 tensions, computes the resulting body shape, and (optionally) renders an SVG visualization for NengoGUI.

## Installation

We recommend a fresh virtual environment to avoid Nengo/NumPy version conflicts with your global Python install:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
from lamprey import build_model

model = build_model(seed=1)
# `model` is a nengo.Network with attached probes on the CPG and tensions.
```

See `examples/run.py` for a complete simulation that produces the headline figures (tension waveforms + body snapshots).

For the original NengoGUI-compatible script (with embedded SVG visualization), see `examples/nengogui/lamprey_model.py`.

## How it works

The CPG ensemble `a` has recurrent dynamics

```
da/dt ≈ (1/τ) · ( τ · M_d · a + 0.05 · a )
```

where `M_d` is a 3×3 matrix encoding a damped oscillator at frequency `freq_hz` with damping `damp0`. With the `1.05·a` term overcoming the damping slightly, the system has an asymptotically stable limit cycle once kicked off the origin.

The tension decoding uses a Galerkin-projection-style basis with `phi(z, m)` Gaussians at positions m/10 along the body and `Phi(z)` = `[1, sin(2πz), cos(2πz), sin(4πz)]` as the temporal basis. The `Gamma` matrix and its pseudo-inverse give the change of basis, and `Z_mat` evaluates the spatial basis at 10 segment positions to produce the final tensions.

## Citation

```bibtex
@book{eliasmith2003neural,
  author    = {Eliasmith, Chris and Anderson, Charles H.},
  title     = {Neural Engineering: Computation, Representation, and Dynamics in Neurobiological Systems},
  publisher = {MIT Press},
  year      = {2003}
}
```

Original implementation: Michael Furlong & Chris Eliasmith, Nengo Summer School 2019.

## License

GPLv2 (see `LICENSE`). Matches Nengo's license.
