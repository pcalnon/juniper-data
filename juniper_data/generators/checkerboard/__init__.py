"""Checkerboard classification dataset generator."""

from juniper_data.generators.checkerboard.generator import (
    VERSION,
    CheckerboardGenerator,
    get_schema,
)
from juniper_data.generators.checkerboard.params import CheckerboardParams

__all__ = [
    "CheckerboardGenerator",
    "CheckerboardParams",
    "VERSION",
    "get_schema",
]
