"""Tier-1 sanity tests for the realtime-audio submission.

This submission's entry_point opens a live microphone stream at module
load time (`mic_input.get_mic()` is invoked from `realtime_audio.py`'s
top level), which is not available on a headless CI runner — hence
`ci_runnable: false` in metadata. These tests therefore verify only the
parts that do *not* require an open audio device:

  * `mic_input` imports cleanly and exposes the expected API.
  * `MicrophoneInput.get_window` can be exercised on a hand-built
    instance using a synthetic ring buffer, without ever opening the
    stream.
"""

import sys
from pathlib import Path

import numpy as np

SUBMISSION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUBMISSION_ROOT))

import mic_input  # noqa: E402


def test_imports():
    assert mic_input.MicrophoneInput is not None
    assert mic_input.get_mic is not None


def test_microphone_input_exposes_node_factory():
    # Bypass __init__ — opening a real stream is what CI does not have.
    mic = mic_input.MicrophoneInput.__new__(mic_input.MicrophoneInput)
    mic.samplerate = 16000
    mic.gain = 1.0
    mic._buffer = np.linspace(-1.0, 1.0, 16000)
    mic._write_idx = 0

    import threading
    mic._lock = threading.Lock()

    # The factory should return a callable suitable for nengo.Node(...).
    fn = mic.make_node_fn(dims=32, window_ms=10)
    out = fn(0.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (32,)
