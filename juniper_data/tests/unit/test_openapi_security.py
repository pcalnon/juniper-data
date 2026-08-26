"""The OpenAPI document is served behind the API key, and declares the scheme (APD-DATA-005 / APD-DATA-024).

These two entries were one defect wearing two faces, and neither could be fixed alone.

``APD-DATA-005``: ``api_key_header`` was instantiated in ``api/security.py`` and referenced
nowhere, so the generated document declared no ``securitySchemes`` at all -- a consumer
generating an SDK from it produced a client that never sends ``X-API-Key``.

``APD-DATA-024``: ``openapi_url`` was ``None`` whenever any key was configured, so a secured
deployment served **no document at all** -- which is exactly why the missing scheme was
unobservable where it mattered. Fixing ``-005`` alone would have added a scheme to a document
nobody could fetch.

**The trap this module exists to pin.** ``EXEMPT_PATHS`` already listed ``/docs``,
``/openapi.json`` and ``/redoc``, and ``SecurityMiddleware._is_exempt()`` is a bare
``path in EXEMPT_PATHS`` evaluated *regardless of whether a key was supplied*. So the obvious
reading of "serve the document behind the key" -- re-enable ``openapi_url`` and stop -- does
not serve it behind the key at all. It serves it to **everyone**, while looking exactly like
the intended fix. The decisive arm below is therefore not "is the document served?" but
"is it served *only* to a caller holding the key?".

The interactive explorers stay off under auth on purpose: Swagger UI and ReDoc are browser
pages that fetch ``/openapi.json`` by XHR with no ``X-API-Key`` header, so mounting them
behind the key would serve a page that can only 401.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from juniper_data.api import constants as api_constants
from juniper_data.api.app import create_app
from juniper_data.api.settings import Settings

API_KEY = "test-openapi-key"
DOCUMENTATION_PATHS = ("/openapi.json", "/docs", "/redoc")


def _secured_app(tmp_path: Path):
    """An app configured exactly as a secured deployment is."""
    return create_app(settings=Settings(storage_path=str(tmp_path), api_keys=[API_KEY]))


def _open_app(tmp_path: Path):
    """An app with no keys configured -- the local-development posture."""
    return create_app(settings=Settings(storage_path=str(tmp_path), api_keys=None))


@pytest.mark.unit
class TestDocumentationIsNotExempt:
    """The documentation surface must not be waved through by ``_is_exempt``."""

    @pytest.mark.parametrize("path", DOCUMENTATION_PATHS)
    def test_documentation_path_is_not_exempt(self, path: str) -> None:
        # A membership assertion, not a behavioural one, because `_is_exempt` ignores
        # the key: while any of these sits in the set, the path is open to everyone
        # the moment it is mounted, no matter what the auth configuration says.
        assert path not in api_constants.EXEMPT_PATHS, f"{path} is exempt from auth -- serving it would expose it to unauthenticated callers"


@pytest.mark.unit
class TestSecuredDeploymentServesTheDocumentBehindTheKey:
    """APD-DATA-024, and the decisive arm for the whole change."""

    def test_document_is_served_when_keys_are_configured(self, tmp_path: Path) -> None:
        with TestClient(_secured_app(tmp_path)) as client:
            assert client.get("/openapi.json", headers={"X-API-Key": API_KEY}).status_code == 200

    def test_document_requires_the_key(self, tmp_path: Path) -> None:
        """The arm that separates "behind the key" from "open to everyone"."""
        with TestClient(_secured_app(tmp_path)) as client:
            assert client.get("/openapi.json").status_code == 401
            assert client.get("/openapi.json", headers={"X-API-Key": "wrong"}).status_code == 401

    def test_explorers_are_not_mounted_under_auth(self, tmp_path: Path) -> None:
        """Deliberate: a browser page cannot send the header, so it could only 401."""
        app = _secured_app(tmp_path)
        assert app.docs_url is None
        assert app.redoc_url is None

    def test_explorers_are_mounted_without_keys(self, tmp_path: Path) -> None:
        """The local-development posture is unchanged by this fix."""
        app = _open_app(tmp_path)
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"


@pytest.mark.unit
class TestDocumentDeclaresTheSecurityScheme:
    """APD-DATA-005 -- and that the declaration tells the truth about which routes need a key."""

    def test_security_scheme_is_declared(self, tmp_path: Path) -> None:
        schemes = _secured_app(tmp_path).openapi()["components"]["securitySchemes"]
        assert schemes, "no securitySchemes -- a generated SDK would never send the key"
        declared = [s for s in schemes.values() if s.get("type") == "apiKey"]
        assert declared, f"no apiKey scheme among {list(schemes)}"
        assert declared[0]["in"] == "header"
        assert declared[0]["name"] == api_constants.HEADER_X_API_KEY

    def test_protected_routes_carry_a_security_requirement(self, tmp_path: Path) -> None:
        document = _secured_app(tmp_path).openapi()
        protected = [path for path in document["paths"] if not path.startswith(f"{api_constants.API_PREFIX}/health")]
        assert protected, "no protected paths -- the pin would be vacuous"
        missing = [path for path, item in document["paths"].items() if path in protected and not any(operation.get("security") for operation in item.values() if isinstance(operation, dict))]
        assert not missing, f"documented without a security requirement: {sorted(missing)}"

    def test_exempt_health_routes_are_not_documented_as_requiring_a_key(self, tmp_path: Path) -> None:
        """The document must not overstate the policy either: health is genuinely exempt."""
        document = _secured_app(tmp_path).openapi()
        health = {path: item for path, item in document["paths"].items() if path.startswith(f"{api_constants.API_PREFIX}/health")}
        assert health, "no health paths -- the pin would be vacuous"
        overstated = [path for path, item in health.items() if any(operation.get("security") for operation in item.values() if isinstance(operation, dict))]
        assert not overstated, f"exempt routes documented as requiring a key: {sorted(overstated)}"
