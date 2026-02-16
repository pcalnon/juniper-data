"""
Juniper Data - Dataset generation and management service for the Juniper ecosystem.
"""

import os

import arc_agi
from dotenv import load_dotenv

__version__ = "0.4.0"
__author__ = "Paul Calnon"


def get_arc_agi_env() -> bool:
    """
    Loads the Environment Variables if not already Loaded.
    Returns:
        bool:
            True: if env IS already loaded
            Result of load_dotenv() method (True/False) when env is NOT already loaded
    """
    return True if os.getenv("ARC_AGI_ENV") else bool(load_dotenv())


def reload_arc_agi_env() -> bool:
    """
    Reloads all of the Environment Variables from local OS env whether already loaded or not.

    Returns:
        bool: _description_
            Result of load_dotenv() method (True/False).
    """
    return bool(load_dotenv())


def get_arc_api_key() -> str | None:
    """
    Return the current value of the ARC_API_KEY environment variable as a string.
    """
    return os.getenv("ARC_API_KEY") or None


def get_arc_agi_api_url() -> str | None:
    """
    Return the current value of the ARC_AGI_API as a URL/endpoint string.

    Reading the environment at call time avoids import-time side effects
    and makes it easier to adjust configuration in tests.
    """
    return os.getenv("ARC_AGI_API") or None


def get_arc_agi_arcade() -> arc_agi.Arcade | None:
    """
    Return the current value of the ARC_AGI_API environment variable as a URL/endpoint string.

    Reading the environment at call time avoids import-time side effects
    and makes it easier to adjust configuration in tests.
    """
    # Automatically uses ARC_API_KEY from the environment by default, or you can pass the API key explicitly.
    return arc_agi.Arcade(arc_api_key=get_arc_api_key()) or None


# Deprecated
def get_arc_agi_api() -> str | None:
    """
    Deprecated alias for :func:`get_arc_agi_api_url`.

    This function returns the same value as :func:`get_arc_agi_api_url` and will be
    removed in a future release. Use :func:`get_arc_agi_api_url` instead.
    """
    return get_arc_agi_api_url()
