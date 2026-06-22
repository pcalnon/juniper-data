"""Delay-product (bilinear capacity) irregular-Δt synthetic time-series regression generator."""

from juniper_data.generators.delay_product.generator import VERSION, DelayProductGenerator, get_schema
from juniper_data.generators.delay_product.params import DelayProductParams

__all__ = [
    "DelayProductGenerator",
    "DelayProductParams",
    "VERSION",
    "get_schema",
]
