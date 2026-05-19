"""Tier-1 sanity tests for communication-channel."""

import importlib.util
from pathlib import Path

import nengo


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location(
        "comm_channel", SUBMISSION_ROOT / "communication_channel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_cleanly():
    mod = load_script()
    assert isinstance(mod.model, nengo.Network)


def test_top_level_objects_exposed():
    mod = load_script()
    for name in ("model", "a", "b", "stim"):
        assert hasattr(mod, name)


def test_runs_100ms():
    mod = load_script()
    with nengo.Simulator(mod.model) as sim:
        sim.run(0.1)
    assert sim.time > 0.09
