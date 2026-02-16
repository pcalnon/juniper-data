"""
Juniper Data - Dataset generation and management service for the Juniper ecosystem.
"""

import os
import arc_agi

from dotenv import load_dotenv


__version__ = "0.4.0"
__author__ = "Paul Calnon"


def get_arc_api_key() -> str | None:
    """
    Return the current value of the ARC_API_KEY environment variable as a string.
    """
    load_dotenv()
    return os.getenv("ARC_API_KEY")

def get_arc_agi_api_url() -> str | None:
    """
    Return the current value of the ARC_AGI_API as a URL/endpoint string.

    Reading the environment at call time avoids import-time side effects
    and makes it easier to adjust configuration in tests.
    """
    load_dotenv()
    return os.getenv("ARC_AGI_API")

def get_arc_agi_arcade() -> arc_agi.Arcade | None:
    """
    Return the current value of the ARC_AGI_API environment variable as a URL/endpoint string.

    Reading the environment at call time avoids import-time side effects
    and makes it easier to adjust configuration in tests.
    """
    # Automatically uses ARC_API_KEY from environment:  arc = arc_agi.Arcade(), Or pass the API key explicitly
    return arc_agi.Arcade(arc_api_key=get_arc_api_key())

# Deprecated
def get_arc_agi_api() -> str | None:
    """
    Deprecated alias for :func:`get_arc_agi_api_url`.

    This function returns the same value as :func:`get_arc_agi_api_url` and will be
    removed in a future release. Use :func:`get_arc_agi_api_url` instead.
    """
    return get_arc_agi_api_url()
