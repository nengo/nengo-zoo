"""Tier-1 sanity tests for the 2d-decision-integrator submission (GUI-first model)."""

import importlib.util
from pathlib import Path

import nengo


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location(
        "decision_integrator_script",
        SUBMISSION_ROOT / "2d_decision_integrator.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_cleanly():
    mod = load_script()
    assert hasattr(mod, "model"), "must expose a top-level `model`"
    assert isinstance(mod.model, nengo.Network)


def test_cfg_referenced_names_exposed():
    """2d_decision_integrator.py.cfg references LIP, MT, ens_inp, ens_out, input1, input2, model."""
    mod = load_script()
    for name in ("model", "LIP", "MT", "ens_inp", "ens_out", "input1", "input2"):
        assert hasattr(mod, name), f"missing top-level `{name}`"


def test_runs_100ms():
    mod = load_script()
    with nengo.Simulator(mod.model) as sim:
        sim.run(0.1)
    assert sim.time > 0.09
