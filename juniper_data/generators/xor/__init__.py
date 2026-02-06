"""XOR classification dataset generator."""

from juniper_data.generators.xor.generator import VERSION, XorGenerator, get_schema
from juniper_data.generators.xor.params import XorParams

__all__ = [
    "XorGenerator",
    "XorParams",
    "VERSION",
    "get_schema",
]
