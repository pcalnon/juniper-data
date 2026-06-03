"""Equities (S&P 500) time-series dataset generator."""

# Project:       Juniper
# Sub-Project:   JuniperData
# Application:   juniper_data
# File Name:     __init__.py
# Author:        Paul Calnon
# Version:       0.6.0
# License:       MIT License

from juniper_data.generators.equities.generator import VERSION, EquitiesGenerator, get_schema
from juniper_data.generators.equities.params import EquitiesParams

__all__ = [
    "EquitiesGenerator",
    "EquitiesParams",
    "VERSION",
    "get_schema",
]
