"""API module for Juniper Data service.

``create_app`` is exposed LAZILY (PEP 562) rather than imported eagerly. That import is what
made ``juniper_data.generators.csv_import`` impossible to import on its own
(juniper-data#316):

    csv_import/__init__
      -> csv_import.generator          imports juniper_data.api.settings
      -> juniper_data.api/__init__     eagerly imported .app
      -> api.app                       imports .routes.datasets / .routes.generators
      -> api.routes.generators         imports juniper_data.generators.csv_import  <-- still
                                       initialising, so VERSION did not exist yet
      -> ImportError: cannot import name 'VERSION'

Importing any submodule initialises its parent package first, so
``from juniper_data.api.settings import get_settings`` -- a perfectly ordinary leaf import --
dragged the entire FastAPI app and every route into the cycle.

Deferring ``create_app`` breaks it for the whole class, not just for ``csv_import``: nothing
that merely wants ``Settings`` pulls the routes in any more. ``Settings`` / ``get_settings``
stay eager because they are leaves -- ``api.settings`` imports only ``api.constants`` and
``core.secrets``.

``from juniper_data.api import create_app`` continues to work unchanged. No caller uses it
today (every consumer imports ``juniper_data.api.app`` directly), but it is part of the
package's public surface and removing it would be a gratuitous break.
"""

from typing import TYPE_CHECKING, Any

from .settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - import-time only, for type checkers
    from .app import create_app

__all__ = [
    "create_app",
    "Settings",
    "get_settings",
]


def __getattr__(name: str) -> Any:
    """Resolve ``create_app`` on first access (PEP 562)."""
    if name == "create_app":
        from .app import create_app as _create_app

        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep ``dir()`` and tab-completion honest about the lazy attribute."""
    return sorted(set(globals()) | set(__all__))
