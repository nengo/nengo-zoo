"""
Stewart, Choo & Eliasmith (2010) basal ganglia.

A spiking model of action selection through the direct/indirect pathways
of the basal ganglia. This is a curated Zoo-level wrapper around the same
implementation that ships in `nengo.networks.BasalGanglia`.

Reference
---------
Stewart, T. C., Choo, X., & Eliasmith, C. (2010). Dynamic behaviour of a
spiking model of action selection in the basal ganglia. In Proceedings of
the 10th International Conference on Cognitive Modeling.
"""

from __future__ import annotations

import nengo


class BasalGanglia(nengo.Network):
    """Spiking basal ganglia for action selection.

    Takes a `dimensions`-length utility vector as input. Produces a
    `dimensions`-length output that is near zero for the selected
    (highest-utility) action and strongly negative for the others.

    Parameters
    ----------
    dimensions : int
        Number of competing actions.
    n_neurons_per_ensemble : int, optional
        Neurons per nucleus ensemble. Defaults to 100.
    output_weight : float, optional
        Scaling applied to the GPi → output projection. Defaults to -3
        (matches Stewart 2010).
    input_bias : float, optional
        Constant bias added to the input. Defaults to 0.
    label : str, optional
        Network label (forwarded to nengo.Network).
    seed : int, optional
        Seed (forwarded to nengo.Network).

    Attributes
    ----------
    input : nengo.Node
        Connect upstream networks here. Shape: (dimensions,).
    output : nengo.Node
        Read the selection signal here. Shape: (dimensions,).
    bg : nengo.networks.BasalGanglia
        The underlying core-Nengo BG subnetwork. Exposed for advanced
        inspection (e.g. probing individual nuclei).
    """

    def __init__(
        self,
        dimensions: int,
        n_neurons_per_ensemble: int = 100,
        output_weight: float = -3.0,
        input_bias: float = 0.0,
        label: str | None = None,
        seed: int | None = None,
    ):
        super().__init__(label=label or "BasalGanglia", seed=seed)
        if dimensions < 1:
            raise ValueError("`dimensions` must be >= 1.")

        with self:
            self.bg = nengo.networks.BasalGanglia(
                dimensions=dimensions,
                n_neurons_per_ensemble=n_neurons_per_ensemble,
                output_weight=output_weight,
                input_bias=input_bias,
            )
            self.input = self.bg.input
            self.output = self.bg.output

    def add_input(self, source, transform):
        """Convenience: add a transformed input projection into the BG.

        Equivalent to `nengo.Connection(source, self.input, transform=transform)`.
        """
        return nengo.Connection(source, self.input, transform=transform)
