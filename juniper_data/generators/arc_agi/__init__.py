"""ARC-AGI (Abstraction and Reasoning Corpus) dataset generator."""

from juniper_data.generators.arc_agi.generator import VERSION, ArcAgiGenerator, get_schema
from juniper_data.generators.arc_agi.params import ArcAgiParams

__all__ = [
    "ArcAgiGenerator",
    "ArcAgiParams",
    "VERSION",
    "get_schema",
]
