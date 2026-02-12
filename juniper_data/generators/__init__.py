"""Data generators module for Juniper Data."""

from .spiral import SpiralGenerator, SpiralParams
from .spiral import get_schema as get_spiral_schema

# Backwards-compatible alias: existing code may still import `get_schema`
# from this package. Prefer `get_spiral_schema` for new code.
get_schema = get_spiral_schema

__all__ = [
    "SpiralGenerator",
    "SpiralParams",
    "get_spiral_schema",
]
