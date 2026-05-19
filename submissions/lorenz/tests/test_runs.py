"""Tier-1 sanity tests for the lorenz submission (GUI-first model)."""

import importlib.util
from pathlib import Path

import nengo
import numpy as np


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def load_script():
    """Import lorenz.py from the submission root, returning its module."""
    script = SUBMISSION_ROOT / "lorenz.py"
    spec = importlib.util.spec_from_file_location("lorenz_script", str(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_cleanly():
    mod = load_script()
    assert hasattr(mod, "model"), "lorenz.py must expose a top-level `model`"
    assert isinstance(mod.model, nengo.Network)


def test_top_level_x_exposed_for_cfg():
    """The .cfg references the bare name `x` — must resolve in script namespace."""
    mod = load_script()
    assert hasattr(mod, "x"), "lorenz.py.cfg references `x`; it must be top-level"


def test_runs_100ms():
    mod = load_script()
    with nengo.Simulator(mod.model) as sim:
        sim.run(0.1)
    # 100ms of probed activity from at least one signal in the network.
    assert sim.time > 0.09


def test_attractor_explores_nontrivial_volume():
    """After settling, the trajectory should sweep through non-trivial state."""
    mod = load_script()
    with mod.model:
        p = nengo.Probe(mod.x, synapse=0.01)
    with nengo.Simulator(mod.model) as sim:
        sim.run(2.0)
    traj = sim.data[p][-1000:]
    spans = traj.max(axis=0) - traj.min(axis=0)
    assert spans.min() > 1.0, f"trajectory looks degenerate; spans={spans}"
