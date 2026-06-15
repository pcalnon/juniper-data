"""AR(p) synthetic time-series regression dataset generator."""

from juniper_data.generators.ar_p.generator import VERSION, ArPGenerator, get_schema
from juniper_data.generators.ar_p.params import ArPParams

__all__ = [
    "ArPGenerator",
    "ArPParams",
    "VERSION",
    "get_schema",
]
