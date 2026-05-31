"""Tier-1 sanity tests for the spa-routed-sequence-cleanup-all submission (GUI-first model)."""

import importlib.util
from pathlib import Path

import nengo


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location(
        "spa_routed_sequence_cleanup_all_script",
        SUBMISSION_ROOT / "spa_routed_sequence_cleanup_all.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_cleanly():
    mod = load_script()
    assert hasattr(mod, "model"), "must expose a top-level `model`"
    assert isinstance(mod.model, nengo.Network)


def test_spa_modules_present():
    """The .cfg references state, vision, bg, thal, cleanup, and input."""
    mod = load_script()
    for attr in ("state", "vision", "bg", "thal", "cleanup", "input"):
        assert hasattr(mod.model, attr), f"model.{attr} missing"


def test_runs_100ms():
    mod = load_script()
    with nengo.Simulator(mod.model) as sim:
        sim.run(0.1)
    assert sim.time > 0.09
