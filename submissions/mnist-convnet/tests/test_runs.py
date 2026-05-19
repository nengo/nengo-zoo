"""
Minimal tests for the mnist-convnet submission.

This submission is `ci_runnable: false` — the heavy deps (nengo_dl,
tensorflow, nengo_loihi) aren't installed in default CI, so we can't
actually build or run the network. We test only that metadata is
parseable and the notebook is present.
"""

from pathlib import Path

import yaml


SUBMISSION_ROOT = Path(__file__).resolve().parent.parent


def test_metadata_parses():
    meta = yaml.safe_load((SUBMISSION_ROOT / "metadata.yaml").read_text())
    assert meta["name"] == "mnist-convnet"
    assert meta.get("ci_runnable") is False, (
        "this submission is intentionally ci_runnable=false; if you remove "
        "that flag, also add proper build/run tests"
    )


def test_notebook_present():
    nb = SUBMISSION_ROOT / "mnist-convnet.ipynb"
    assert nb.exists(), "the notebook is the canonical artifact for this submission"


def test_notebook_parses_as_json():
    import json
    nb_path = SUBMISSION_ROOT / "mnist-convnet.ipynb"
    nb = json.loads(nb_path.read_text())
    assert "cells" in nb
    assert any(c["cell_type"] == "code" for c in nb["cells"])
