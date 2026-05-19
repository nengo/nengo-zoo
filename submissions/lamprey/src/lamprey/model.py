"""
Builds the Nengo network for the lamprey locomotion model.

Architecture (Eliasmith & Anderson 2003; implemented by Furlong & Eliasmith 2019):

  stim(t)  --kick-->   CPG (3D, 500 spiking neurons, damped oscillator)
                       |
                       T(x) basis-function decoding
                       |
                       v
                   tensions (10D)  --->  Lamprey body
"""

from __future__ import annotations

import nengo
import numpy as np

from .body import Lamprey
from .encoding import T


def build_model(
    *,
    seed: int = 1,
    n_neurons_cpg: int = 500,
    freq_hz: float = 3.0,
    damping: float = 0.1,
    tau: float = 0.2,
    kick_until: float = 1.0,
    record_body_history: bool = True,
):
    """Build the lamprey Nengo network.

    Returns
    -------
    (model, handles) : tuple
        ``model`` is a ``nengo.Network`` ready to simulate.
        ``handles`` is a dict with keys:
          * 'cpg'      — the 3D CPG ensemble
          * 'tensions' — the 10D tension ensemble (Direct neurons)
          * 'body'     — the ``Lamprey`` Python object
          * 'p_cpg', 'p_tensions' — probes for plotting
    """
    body = Lamprey(record_history=record_body_history)

    with nengo.Network("lamprey", seed=seed) as model:
        gauss_dist = nengo.dists.Gaussian(mean=0, std=0.1)
        white_noise = nengo.processes.WhiteNoise(dist=gauss_dist)

        freq = freq_hz * 2 * np.pi

        # CPG: 3D damped harmonic oscillator with slight overcompensation,
        # producing a stable limit cycle once kicked off the origin.
        M_d = np.array([
            [-damping,  freq, -damping],
            [-freq,     0,     0      ],
            [-damping, -freq, -damping],
        ])
        # Kick matrix: projects the 3D constant kick into the oscillator basis.
        M_i = np.array([
            [ 0.5, 0, -0.5],
            [ 0,   1,  0  ],
            [-0.5, 0,  0.5],
        ])

        cpg = nengo.Ensemble(
            n_neurons=n_neurons_cpg,
            dimensions=3,
            radius=1,
            noise=white_noise,
            label="cpg",
        )

        def feedback_func(x):
            return tau * np.dot(M_d, x) + 1.05 * x

        nengo.Connection(cpg, cpg, function=feedback_func, synapse=tau)

        def stim_func(t):
            return np.array([1.0, 1.0, 1.0]) if t < kick_until else np.array([0.0, 0.0, 0.0])

        u = nengo.Node(stim_func, label="kick")

        def kick_func(x):
            return tau * np.dot(M_i, x)

        nengo.Connection(u, cpg, function=kick_func, synapse=tau)

        # Tension decoding (3D CPG → 10D body tensions) via the analytic basis.
        tensions = nengo.Ensemble(
            n_neurons=100, dimensions=10, neuron_type=nengo.Direct(), label="tensions"
        )
        nengo.Connection(cpg, tensions, function=T)

        # Body node: consumes tensions, optionally renders SVG.
        body_node = nengo.Node(body, size_in=10, label="body")
        nengo.Connection(tensions, body_node)

        # Probes for downstream plotting.
        p_cpg = nengo.Probe(cpg, synapse=0.01)
        p_tensions = nengo.Probe(tensions, synapse=0.01)

    handles = {
        "cpg": cpg,
        "kick": u,
        "tensions": tensions,
        "body": body,           # the Python Lamprey object (state, history)
        "body_node": body_node, # the nengo.Node wrapping it (for .cfg refs)
        "p_cpg": p_cpg,
        "p_tensions": p_tensions,
    }
    return model, handles
