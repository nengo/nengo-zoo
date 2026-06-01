"""nengo_a2c — Nengo-based Actor-Critic networks from Bartlett et al. (2022).

Public API:
    ActorCritic, ActorCriticLDN, LDN              # nengo_a2c.networks
    TD0, TDt                                       # nengo_a2c.rules
    NormalRep, OneHotRepCP, OneHotRepRB           # nengo_a2c.representations
    sparsity_to_x_intercept, softmax              # nengo_a2c.utils
"""

from .networks import ActorCritic, ActorCriticLDN, LDN
from .representations import NormalRep, OneHotRepCP, OneHotRepRB
from .rules import TD0, TDt
from .utils import softmax, sparsity_to_x_intercept

__all__ = [
    "ActorCritic",
    "ActorCriticLDN",
    "LDN",
    "TD0",
    "TDt",
    "NormalRep",
    "OneHotRepCP",
    "OneHotRepRB",
    "softmax",
    "sparsity_to_x_intercept",
]
