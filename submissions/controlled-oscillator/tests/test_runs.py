"""Tier-1 sanity tests for the controlled-oscillator submission."""

import importlib.util
from pathlib import Path

import nengo


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location(
        "controlled_oscillator_script",
        SUBMISSION_ROOT / "controlled_oscillator.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_cleanly():
    mod = load_script()
    assert hasattr(mod, "model"), "must expose a top-level `model`"
    assert isinstance(mod.model, nengo.Network)


def test_cfg_referenced_names_exposed():
    """controlled_oscillator.py.cfg references `x`, `stim_speed`, `speed`."""
    mod = load_script()
    for name in ("model", "x", "stim_speed", "speed"):
        assert hasattr(mod, name), f"missing top-level `{name}`"


def test_runs_100ms():
    mod = load_script()
    with nengo.Simulator(mod.model) as sim:
        sim.run(0.1)
    assert sim.time > 0.09
