"""
The submission's main module.

Rename this file (and its parent package) to match your submission's name.
For a `network` type, this typically contains a nengo.Network subclass.
"""

import nengo


class MyNetwork(nengo.Network):
    """One-line description of the network.

    Parameters
    ----------
    n_neurons : int
        Number of neurons per ensemble.
    **kwargs
        Forwarded to `nengo.Network`.
    """

    def __init__(self, n_neurons: int = 100, **kwargs):
        super().__init__(**kwargs)
        with self:
            self.ens = nengo.Ensemble(n_neurons, dimensions=1)
            self.input = self.ens
            self.output = self.ens
