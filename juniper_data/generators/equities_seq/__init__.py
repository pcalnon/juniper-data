"""Windowed (3-D sequence) equities time-series dataset generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     __init__.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from juniper_data.generators.equities_seq.generator import VERSION, EquitiesSeqGenerator, get_schema
from juniper_data.generators.equities_seq.params import EquitiesSeqParams

__all__ = [
    "EquitiesSeqGenerator",
    "EquitiesSeqParams",
    "VERSION",
    "get_schema",
]
