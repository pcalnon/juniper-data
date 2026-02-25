"""
Juniper Data - Dataset generation and management service for the Juniper ecosystem.
"""

import os

from dotenv import load_dotenv

try:
    import arc_agi

    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False
    arc_agi = None  # type: ignore[assignment]

__version__ = "0.4.2"
__author__ = "Paul Calnon"


def get_arc_agi_env() -> bool:
    """
    Ensure ARC_AGI_ENV is available by loading environment variables if needed.

    This function attempts to load environment variables from a `.env` file and then
    returns whether the `ARC_AGI_ENV` environment variable is set.

    Returns:
        bool: True if ARC_AGI_ENV is set after loading, otherwise False.
    """
    # Attempt to load variables from a .env file, but base the result solely on
    # whether ARC_AGI_ENV is present afterwards to provide consistent semantics.
    load_dotenv()
    return bool(os.getenv("ARC_AGI_ENV"))


def reload_arc_agi_env() -> bool:
    """
    Reloads all of the Environment Variables from local OS env whether already loaded or not.

    Returns:
        bool: True if environment variables were loaded from a .env file, False otherwise.
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


def get_arc_agi_arcade() -> "arc_agi.Arcade | None":
    """
    Create and return an :class:`arc_agi.Arcade` instance configured from environment variables.

    The API key is read from the environment via :func:`get_arc_api_key`, avoiding import-time
    side effects and making it easier to adjust configuration in tests.

    Raises:
        ImportError: If the ``arc-agi`` package is not installed.
    """
    if not ARC_AGI_AVAILABLE:
        raise ImportError("arc-agi package not installed. Install with: pip install 'juniper-data[arc-agi]'")
    # Automatically uses ARC_API_KEY from environment:  arc = arc_agi.Arcade(), Or pass the API key explicitly
    return arc_agi.Arcade(arc_api_key=get_arc_api_key()) or None


# Deprecated
def get_arc_agi_api() -> str | None:
    """
    Deprecated alias for :func:`get_arc_agi_api_url`.

    This function returns the same value as :func:`get_arc_agi_api_url` and will be
    removed in a future release. Use :func:`get_arc_agi_api_url` instead.
    """
    return get_arc_agi_api_url()
