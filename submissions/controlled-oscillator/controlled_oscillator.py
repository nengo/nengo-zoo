# Controlled Oscillator
#
# The same 2D oscillator dynamics as a plain oscillator, but with a
# third recurrent dimension `x[2]` that gates the angular speed `s`.
# A separate `speed` ensemble feeds that dimension via a slider, so the
# oscillation can be sped up, slowed down, or stopped by another neural
# population.
#
# Equations:
#   dx0/dt = -x1 * s + x0 * (r - x0**2 - x1**2)
#   dx1/dt =  x0 * s + x1 * (r - x0**2 - x1**2)
# with r = 1 and s = 10 * x[2].

import nengo

model = nengo.Network()
with model:

    x = nengo.Ensemble(n_neurons=400, dimensions=3)

    synapse = 0.1

    def oscillator(x):
        r = 1
        s = 10 * x[2]
        return [
            synapse * -x[1] * s + x[0] * (r - x[0] ** 2 - x[1] ** 2) + x[0],
            synapse * x[0] * s + x[1] * (r - x[0] ** 2 - x[1] ** 2) + x[1],
        ]

    nengo.Connection(x, x[:2], synapse=synapse, function=oscillator)

    stim_speed = nengo.Node(0)
    speed = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim_speed, speed)
    nengo.Connection(speed, x[2])
