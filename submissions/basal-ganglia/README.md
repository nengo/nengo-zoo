# basal-ganglia

A spiking model of the basal ganglia for action selection. Feed in a utility vector, read out a one-hot selection signal.

## Description

This is the Stewart / Choo / Eliasmith (2010) basal ganglia: a biologically-grounded action-selection circuit implementing the direct and indirect pathways through striatum (StrD1, StrD2), subthalamic nucleus (STN), and globus pallidus (GPi/SNr, GPe).

The model takes a `dimensions`-length input vector — interpreted as the "utility" of each candidate action — and produces an output vector that is approximately zero for the selected (highest-utility) action and strongly negative for the others. Downstream networks (typically a thalamus + cortex) can use this to gate action execution.

It's the canonical action-selection primitive used throughout the Spaun line of models and most CNRG cognitive architectures.

## Installation

We recommend a fresh virtual environment to avoid Nengo/NumPy version conflicts with your global Python install:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
import nengo
from basal_ganglia import BasalGanglia

with nengo.Network() as model:
    bg = BasalGanglia(dimensions=3, n_neurons_per_ensemble=100)

    utility = nengo.Node([0.7, 0.5, 0.3])     # action utilities
    nengo.Connection(utility, bg.input)

    output_probe = nengo.Probe(bg.output, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(0.5)
```

See `examples/example_usage.py` for a 3-action demo where utilities change over time and the BG cleanly switches its selection.

## How it works

Per Stewart, Choo & Eliasmith (2010):

- **StrD1** (direct pathway striatum) and **StrD2** (indirect) each receive the utility input through a soft-threshold nonlinearity (the cortico-striatal projection).
- **StrD1 → GPi** is inhibitory: high-utility actions silence their corresponding GPi neuron.
- **StrD2 → GPe → GPi** is doubly inhibitory: the indirect pathway disinhibits competing actions.
- **STN → GPi** is excitatory (diffuse): provides background drive against which the direct-pathway inhibition is detected.
- **GPi/SNr** is the output: zero (or near-zero) firing for the selected action, strong negative output for the rest.

Constants and weights follow the values in Stewart 2010, Table 1.

This implementation wraps `nengo.networks.BasalGanglia` from core Nengo, which is the same model — the wrapper exists to provide curated metadata, a tested example, citation info, and a stable Zoo-level API.

## Citation

```bibtex
@inproceedings{stewart2010basalganglia,
  author    = {Stewart, Terrence C. and Choo, Xuan and Eliasmith, Chris},
  title     = {Dynamic behaviour of a spiking model of action selection in the basal ganglia},
  booktitle = {Proceedings of the 10th International Conference on Cognitive Modeling},
  year      = {2010}
}
```

## License

GPLv2 (see `LICENSE`). Matches Nengo's license.
