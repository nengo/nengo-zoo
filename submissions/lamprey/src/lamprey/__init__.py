"""Lamprey locomotion model for the NengoZoo.

Eliasmith & Anderson (2003), NEF book — implementation by
Michael Furlong & Chris Eliasmith, Nengo Summer School 2019.
"""

from .body import Lamprey
from .encoding import T, build_decoding_matrices
from .model import build_model

__all__ = ["Lamprey", "T", "build_decoding_matrices", "build_model"]
__version__ = "0.1.0"
