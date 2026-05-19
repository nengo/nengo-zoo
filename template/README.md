# {{ submission-name }}

<!-- One-line tagline describing what this submission is and who it's for. -->

## Description

<!-- 1–3 paragraphs. What does it do? What's the intuition? What's it useful for? -->

## Installation

We recommend a fresh virtual environment to avoid Nengo/NumPy version conflicts with whatever else is on your system:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
import nengo
from my_submission import MyNetwork  # ← replace with your import

with nengo.Network() as net:
    sub = MyNetwork(n_neurons=100)
    # ...
```

See `examples/example_usage.py` for a complete, runnable example.

## How it works

<!-- Short technical explanation. Math, architecture, key equations, design choices. -->

## Citation

<!-- If this is paper-linked, include a citation block.
     Otherwise: "If you use this submission, please cite the NengoZoo entry: ..."
-->

```bibtex
@article{example2025,
  author  = {…},
  title   = {…},
  journal = {…},
  year    = {2025},
  doi     = {10.1234/example}
}
```

## License

GPLv2 (see `LICENSE`).
