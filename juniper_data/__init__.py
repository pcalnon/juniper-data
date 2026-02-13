"""
Juniper Data - Dataset generation and management service for the Juniper ecosystem.
"""

import os

__version__ = "0.4.0"
__author__ = "Paul Calnon"


def get_arc_agi_api_url() -> str | None:
    """
    Return the current value of the ARC_AGI_API environment variable as a URL/endpoint string.

    Reading the environment at call time avoids import-time side effects
    and makes it easier to adjust configuration in tests.
    """
    return os.getenv("ARC_AGI_API")


def get_arc_agi_api() -> str | None:
    """
    Deprecated alias for :func:`get_arc_agi_api_url`.

    This function returns the same value as :func:`get_arc_agi_api_url` and will be
    removed in a future release. Use :func:`get_arc_agi_api_url` instead.
    """
    return get_arc_agi_api_url()
