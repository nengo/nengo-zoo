"""Live microphone audio injected into a population of neurons.

Run with: nengo model.py
"""

import nengo

from mic_input import get_mic

AUDIO_DIMS = 32  # dimensions of the waveform window / ensemble
WINDOW_MS = 10  # how much recent audio each window spans
GAIN = 10.0  # scales raw mic samples toward the ensemble's +/-1 radius

mic = get_mic()
mic.gain = GAIN

model = nengo.Network(label="Realtime audio")
with model:
    audio_in = nengo.Node(
        mic.make_node_fn(AUDIO_DIMS, WINDOW_MS),
        size_out=AUDIO_DIMS,
        label="microphone",
    )
    audio_ens = nengo.Ensemble(
        n_neurons=20 * AUDIO_DIMS,
        dimensions=AUDIO_DIMS,
        label="audio neurons",
    )
    # No synapse so the waveform is not low-pass filtered on the way in;
    # the ensemble itself does the encoding.
    nengo.Connection(audio_in, audio_ens, synapse=None)
