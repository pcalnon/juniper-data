"""Two-moons (interleaving half-moons) classification dataset generator.

Introduced to resolve XREPO-01b / DC-02: the ``juniper-data-client``
constant ``GENERATOR_MOON`` previously referenced a server generator
that did not exist. This module implements the server-side generator
so ``client.create_dataset("moon", ...)`` succeeds.
"""

from juniper_data.generators.moon.generator import VERSION, MoonGenerator, get_schema
from juniper_data.generators.moon.params import MoonParams

__all__ = [
    "MoonGenerator",
    "MoonParams",
    "VERSION",
    "get_schema",
]
