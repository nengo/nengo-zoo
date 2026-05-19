"""Tier-1 sanity tests for the LMU submission.

These tests deliberately do NOT simulate the full sim_t=100 training run
(that's `examples/run_training.py`'s job, several minutes wall-clock).
We just verify the network builds and runs 100ms cleanly.
"""

import importlib.util
from pathlib import Path

import nengo
import numpy as np


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location(
        "lmu_script", SUBMISSION_ROOT / "lmu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_cleanly():
    mod = load_script()
    assert hasattr(mod, "model")
    assert isinstance(mod.model, nengo.Network)


def test_lmu_matrices_shape():
    mod = load_script()
    assert mod.A.shape == (mod.order, mod.order)
    assert mod.B.shape == (mod.order, 1)


def test_ideal_delay_synapse_round_trip():
    """IdealDelay should reproduce its input shifted by `delay` seconds."""
    mod = load_script()
    syn = mod.IdealDelay(delay=0.05)  # 50ms delay
    step = syn.make_step((1,), (1,), dt=0.001, rng=None, state={})
    # First 50 steps should output 0 (the buffered zeros).
    for _ in range(50):
        out = step(0, np.array([1.0]))
    # 51st step should start emitting the buffered input.
    out = step(0.051, np.array([1.0]))
    assert out[0] == 1.0


def test_runs_100ms():
    mod = load_script()
    with nengo.Simulator(mod.model) as sim:
        sim.run(0.1)
    assert sim.time > 0.09
