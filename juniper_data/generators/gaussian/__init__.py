"""Gaussian blobs classification dataset generator."""

from juniper_data.generators.gaussian.generator import VERSION, GaussianGenerator, get_schema
from juniper_data.generators.gaussian.params import GaussianParams

__all__ = [
    "GaussianGenerator",
    "GaussianParams",
    "VERSION",
    "get_schema",
]
