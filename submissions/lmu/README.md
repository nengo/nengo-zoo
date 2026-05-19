# lmu

A Legendre Memory Unit in Nengo, demonstrated learning a fixed delay.

## Description

Legendre Memory Units (LMUs), described in [Voelker, Kajić, and Eliasmith (NeurIPS 2019)](https://papers.nips.cc/paper_files/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html), are a recurrent neural-network architecture. Broadly: an LMU optimally represents a sliding window of a continuous-time signal using a basis of Legendre polynomials, giving the network access to a compressed but information-dense history of its input.

This submission wires an LMU layer into a Nengo network and trains a downstream spiking ensemble (with PES learning) to compute a fixed time delay of a white-noise input. Learning is on for the first 80 % of the run and off for the last 20 % so generalization can be assessed.

This is a **GUI-first** submission: open `lmu.py` in NengoGUI to watch the network learn the delay live. The full notebook walkthrough is in `lmu.ipynb` and the long-form training-and-plotting script is in `examples/run_training.py`.

## Run it

In NengoGUI:

```bash
pip install -r requirements.txt nengo-gui
nengo lmu.py
```

Long-form training run (~100 simulated seconds; takes a couple of minutes wall-clock and produces plots):

```bash
python examples/run_training.py
```

Tutorial walkthrough — open `lmu.ipynb` in Jupyter for a step-by-step explanation of the math (A/B matrix derivation, the IdealDelay synapse, the PES training loop).

## Citation

```bibtex
@inproceedings{voelker2019lmu,
  author    = {Voelker, Aaron R. and Kajić, Ivana and Eliasmith, Chris},
  title     = {Legendre Memory Units: Continuous-time representation in recurrent neural networks},
  booktitle = {Advances in Neural Information Processing Systems 32 (NeurIPS 2019)},
  year      = {2019}
}
```

## License

GPLv2 (see `LICENSE`).
