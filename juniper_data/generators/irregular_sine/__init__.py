"""Irregular-Δt (non-uniform sampling) sine synthetic time-series regression generator."""

from juniper_data.generators.irregular_sine.generator import VERSION, IrregularSineGenerator, get_schema
from juniper_data.generators.irregular_sine.params import IrregularSineParams

__all__ = [
    "IrregularSineGenerator",
    "IrregularSineParams",
    "VERSION",
    "get_schema",
]
