"""Tier-1 sanity tests for the lamprey submission."""

import nengo
import numpy as np

from lamprey import Lamprey, T, build_decoding_matrices, build_model


def test_imports():
    assert Lamprey is not None
    assert T is not None
    assert build_model is not None


def test_decoding_matrices_shape():
    gamma_inv, z_mat = build_decoding_matrices()
    assert gamma_inv.shape == (10, 3)
    assert z_mat.shape == (10, 10)


def test_T_maps_3d_to_10d():
    out = T(np.array([1.0, 0.0, 0.0]))
    assert out.shape == (10,)


def test_model_builds():
    model, handles = build_model(seed=0, n_neurons_cpg=100)
    assert isinstance(model, nengo.Network)
    for key in ("cpg", "tensions", "body", "p_cpg", "p_tensions"):
        assert key in handles


def test_runs_200ms():
    model, handles = build_model(seed=0, n_neurons_cpg=100)
    with nengo.Simulator(model) as sim:
        sim.run(0.2)
    assert sim.data[handles["p_cpg"]].shape == (200, 3)
    assert sim.data[handles["p_tensions"]].shape == (200, 10)


def test_cpg_oscillates_after_settling():
    """After the kick, the CPG should be in a non-trivial limit cycle.

    The default parameters give a small-amplitude oscillation (peak |state|
    ~ 0.1, per-axis std ~ 0.04 over the trailing half-second). A dead-at-
    origin system would produce std on the order of the neural noise floor
    (~0.001-0.01), so 0.02 is a safe lower bound.
    """
    model, handles = build_model(seed=1)
    with nengo.Simulator(model) as sim:
        sim.run(2.0)
    cpg_late = sim.data[handles["p_cpg"]][-500:]
    assert cpg_late.std(axis=0).max() > 0.02
