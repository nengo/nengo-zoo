"""
Communication channel: a value in ensemble `a` is transmitted to ensemble `b`.
The simplest useful Nengo network.
"""

import nengo

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: t % 2 - 1)   # slow ramp in [-1, 1]

    a = nengo.Ensemble(n_neurons=50, dimensions=1)
    b = nengo.Ensemble(n_neurons=50, dimensions=1)

    nengo.Connection(stim, a)
    nengo.Connection(a, b)
