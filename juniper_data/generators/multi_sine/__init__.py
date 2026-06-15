"""Multi-sine synthetic time-series regression dataset generator."""

from juniper_data.generators.multi_sine.generator import VERSION, MultiSineGenerator, get_schema
from juniper_data.generators.multi_sine.params import MultiSineParams

__all__ = [
    "MultiSineGenerator",
    "MultiSineParams",
    "VERSION",
    "get_schema",
]
