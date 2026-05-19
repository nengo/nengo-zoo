# mnist-convnet

A convolutional spiking network trained on MNIST with **NengoDL**, deployed on **Loihi** neuromorphic hardware.

## Description

This submission demonstrates the canonical Nengo "train in TensorFlow, deploy on Loihi" workflow:

1. Define a convolutional architecture using NengoDL's TensorFlow integration.
2. Train it on MNIST with backprop (TensorFlow handles the gradient flow).
3. Convert the resulting weights into a spiking Nengo network.
4. Run inference on Loihi (or a Loihi simulator if you don't have hardware).

The full walkthrough — architecture, training loop, weight analysis, and Loihi deployment — lives in [`mnist-convnet.ipynb`](mnist-convnet.ipynb). Open it in Jupyter.

## Run it

This submission requires a fairly heavy stack. We strongly recommend a fresh venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install nengo nengo-dl tensorflow nengo-loihi matplotlib requests
jupyter notebook mnist-convnet.ipynb
```

For inference-only (no training, faster), the notebook has a `do_training = False` toggle that loads pre-trained weights from the Nengo examples bucket.

## A note on CI

This submission is **`ci_runnable: false`** — the Zoo's CI only verifies metadata and structure, not runtime, because:

- Training requires TensorFlow + GPU (or a long CPU pass) to be meaningful.
- The Loihi deployment step requires either Intel's `nengo_loihi` simulator (~slow) or actual Loihi hardware (no GitHub Actions runners).
- The combined install (`nengo_dl + tensorflow + nengo_loihi`) is multi-hundred-MB and slow to provision per CI run.

The `nengo_dl` and `nengo_loihi` backends in `metadata.yaml` are **author-claimed**, not CI-verified. Future zoo infrastructure could add an optional heavy-CI lane that runs against a Docker image with the full stack pre-installed; that's tracked as a design question.

## Citation

```bibtex
@article{hunsberger2016training,
  author  = {Hunsberger, Eric and Eliasmith, Chris},
  title   = {Training spiking deep networks for neuromorphic hardware},
  journal = {arXiv preprint arXiv:1611.05141},
  year    = {2016}
}
```

## License

GPLv2 (see `LICENSE`).
