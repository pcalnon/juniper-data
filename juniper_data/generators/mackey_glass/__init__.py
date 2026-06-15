"""Mackey-Glass synthetic time-series regression dataset generator."""

from juniper_data.generators.mackey_glass.generator import VERSION, MackeyGlassGenerator, get_schema
from juniper_data.generators.mackey_glass.params import MackeyGlassParams

__all__ = [
    "MackeyGlassGenerator",
    "MackeyGlassParams",
    "VERSION",
    "get_schema",
]
