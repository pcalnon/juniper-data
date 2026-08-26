"""Unit tests for SecurityMiddleware and RequestBodyLimitMiddleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from juniper_data.api.constants import DEFAULT_FAILED_AUTH_MAX_FAILURES
from juniper_data.api.middleware import EXEMPT_PATHS, RequestBodyLimitMiddleware, SecurityMiddleware
from juniper_data.api.security import APIKeyAuth, FailedAuthThrottle, RateLimiter, build_failed_auth_throttle


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with security middleware."""

    def _create(api_keys=None, rate_limit_enabled=False, rpm=60, throttle=None):
        app = FastAPI()
        auth = APIKeyAuth(api_keys)
        limiter = RateLimiter(requests_per_minute=rpm, enabled=rate_limit_enabled)
        # ``throttle=None`` deliberately exercises the production default (an enabled
        # FailedAuthThrottle at the library budget), so the pre-existing arms below prove the
        # default is transparent to well-behaved traffic.
        app.add_middleware(SecurityMiddleware, api_key_auth=auth, rate_limiter=limiter, failed_auth_throttle=throttle)

        @app.get("/v1/health")
        async def health():
            return {"status": "ok"}

        @app.get("/v1/datasets")
        async def datasets():
            return {"data": []}

        return app

    return _create


@pytest.fixture
def app_with_body_limit():
    """Create a FastAPI app with only the body-limit middleware installed."""

    def _create(max_bytes=100):
        app = FastAPI()
        app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_bytes)

        @app.post("/v1/echo")
        async def echo():
            return {"ok": True}

        @app.put("/v1/echo")
        async def echo_put():
            return {"ok": True, "method": "PUT"}

        @app.patch("/v1/echo")
        async def echo_patch():
            return {"ok": True, "method": "PATCH"}

        @app.post("/v1/echo-body")
        async def echo_body(payload: dict):
            return {"received": payload}

        return app

    return _create


async def _drive_asgi(app, *, headers, chunks):
    """Drive ``app`` at the ASGI layer and return ``(status, body)``.

    TestClient/httpx always recomputes ``Content-Length`` from the payload it is
    given, so the two bypass shapes this middleware exists to stop -- a chunked
    body carrying no ``Content-Length`` at all, and one that under-declares --
    are not expressible through the test client. They have to be driven here.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/echo",
        "raw_path": b"/v1/echo",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }
    pending = list(chunks)
    sent: list[dict] = []

    async def receive():
        if pending:
            chunk = pending.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], payload


@pytest.mark.unit
class TestRequestBodyLimitMiddleware:
    """CR-024: ``Content-Length`` is an early-reject hint, never the sole check.

    The bypasses below are the reason the middleware always stream-reads mutating
    methods instead of trusting the declared length.
    """

    def test_rejects_oversized_post(self, app_with_body_limit):
        client = TestClient(app_with_body_limit(max_bytes=10))
        response = client.post("/v1/echo", content=b"x" * 50)
        assert response.status_code == 413
        assert response.json()["detail"] == "Request body too large"

    def test_allows_body_within_limit(self, app_with_body_limit):
        client = TestClient(app_with_body_limit(max_bytes=100))
        response = client.post("/v1/echo", content=b"x" * 5)
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_rejects_oversized_put_and_patch(self, app_with_body_limit):
        """PUT/PATCH share the POST cap -- all three are stream-read."""
        client = TestClient(app_with_body_limit(max_bytes=10))
        assert client.put("/v1/echo", content=b"x" * 50).status_code == 413
        assert client.patch("/v1/echo", content=b"x" * 50).status_code == 413

    def test_malformed_content_length_is_400_not_500(self, app_with_body_limit):
        """APD-DATA-036: an unguarded ``int()`` here escapes as a 500.

        The ValueError would propagate outside ``ExceptionMiddleware``, so the
        app's own ValueError handler never sees it -- a malformed client header
        reported as a server fault.
        """
        client = TestClient(app_with_body_limit(max_bytes=100))
        response = client.post("/v1/echo", content=b"x", headers={"content-length": "not-a-number"})
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid Content-Length header"

    def test_body_is_still_readable_downstream(self, app_with_body_limit):
        """The stream-read must cache the body, or every POST route breaks.

        Guards the regression the fix itself could introduce: the middleware now
        consumes ``request.stream()``, so without the ``request._body`` cache the
        handler would see an empty payload.
        """
        client = TestClient(app_with_body_limit(max_bytes=1000))
        response = client.post("/v1/echo-body", json={"hello": "world"})
        assert response.status_code == 200
        assert response.json() == {"received": {"hello": "world"}}

    async def test_chunked_body_without_content_length_is_capped(self, app_with_body_limit):
        """APD-DATA-002: the named bypass -- no ``Content-Length`` at all.

        Two 8-byte chunks against a 10-byte cap, so the abort must happen on the
        second chunk: the cap is cumulative and enforced mid-stream, not after
        the whole body has been buffered.
        """
        status_code, payload = await _drive_asgi(
            app_with_body_limit(max_bytes=10),
            headers=[(b"content-type", b"application/octet-stream")],
            chunks=[b"x" * 8, b"x" * 8],
        )
        assert status_code == 413
        assert b"Request body too large" in payload

    async def test_underdeclared_content_length_is_capped(self, app_with_body_limit):
        """A small declared length must not buy passage for a large real body."""
        status_code, payload = await _drive_asgi(
            app_with_body_limit(max_bytes=10),
            headers=[(b"content-length", b"5"), (b"content-type", b"application/octet-stream")],
            chunks=[b"x" * 50],
        )
        assert status_code == 413
        assert b"Request body too large" in payload


@pytest.mark.unit
class TestSecurityMiddleware:
    def test_exempt_path_bypasses_security(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_auth_required_returns_401(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/datasets")
        assert response.status_code == 401

    def test_invalid_key_returns_401(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/datasets", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_valid_key_passes(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)
        response = client.get("/v1/datasets", headers={"X-API-Key": "secret"})
        assert response.status_code == 200

    def test_rate_limit_exceeded_returns_429(self, app_with_middleware):
        app = app_with_middleware(rate_limit_enabled=True, rpm=2)
        client = TestClient(app)
        for _ in range(2):
            client.get("/v1/datasets")
        response = client.get("/v1/datasets")
        assert response.status_code == 429

    def test_rate_limit_headers_included(self, app_with_middleware):
        app = app_with_middleware(rate_limit_enabled=True, rpm=10)
        client = TestClient(app)
        response = client.get("/v1/datasets")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_failed_auth_attempts_are_throttled(self, app_with_middleware):
        """APD-DATA-001: the 401 path must consume budget.

        This is the arm that catches a half-port. Wiring only the pre-auth ``check()`` and
        omitting ``record_failure()`` yields a throttle that never accumulates -- a silent
        no-op -- and every request below would stay 401 forever instead of turning 429.
        """
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(max_failures=3, window_seconds=60))
        client = TestClient(app)

        for _ in range(3):
            assert client.get("/v1/datasets", headers={"X-API-Key": "wrong"}).status_code == 401

        response = client.get("/v1/datasets", headers={"X-API-Key": "wrong"})
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1

    def test_valid_credentials_never_consume_the_throttle_budget(self, app_with_middleware):
        """Well-behaved traffic sees no behaviour change, which is why the default is enabled."""
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(max_failures=2, window_seconds=60))
        client = TestClient(app)

        for _ in range(25):
            assert client.get("/v1/datasets", headers={"X-API-Key": "secret"}).status_code == 200

    def test_throttle_is_enabled_by_default(self, app_with_middleware):
        """No throttle passed: the production ``add_middleware`` call site must still be covered.

        juniper-data's own ``app.py`` constructs SecurityMiddleware without a throttle argument,
        so a default of ``None`` that meant "disabled" would leave the running service exactly as
        unprotected as before the fix.
        """
        app = app_with_middleware(api_keys=["secret"])
        client = TestClient(app)

        for _ in range(10):
            assert client.get("/v1/datasets", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/datasets", headers={"X-API-Key": "wrong"}).status_code == 429

    def test_throttle_can_be_opted_out(self, app_with_middleware):
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(enabled=False))
        client = TestClient(app)

        for _ in range(25):
            assert client.get("/v1/datasets", headers={"X-API-Key": "wrong"}).status_code == 401

    def test_quota_429_is_not_counted_as_an_authentication_failure(self, app_with_middleware):
        """Only a 401 feeds the throttle.

        A 429 from the identity-keyed limiter is a quota outcome, not a credential guess.
        Counting it would let an authenticated caller throttle *itself* out of the auth path
        merely by exceeding its own quota.
        """
        throttle = build_failed_auth_throttle(max_failures=2, window_seconds=60)
        app = app_with_middleware(api_keys=["secret"], rate_limit_enabled=True, rpm=1, throttle=throttle)
        client = TestClient(app)

        assert client.get("/v1/datasets", headers={"X-API-Key": "secret"}).status_code == 200
        for _ in range(5):
            assert client.get("/v1/datasets", headers={"X-API-Key": "secret"}).status_code == 429

        # None of those quota 429s were recorded, so the failure budget is still intact.
        assert throttle.check("testclient")[0] is False

    def test_exempt_paths_bypass_the_throttle(self, app_with_middleware):
        """Health checks stay reachable even from an IP that is currently throttled."""
        app = app_with_middleware(api_keys=["secret"], throttle=build_failed_auth_throttle(max_failures=1, window_seconds=60))
        client = TestClient(app)

        assert client.get("/v1/datasets", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/datasets", headers={"X-API-Key": "wrong"}).status_code == 429
        assert client.get("/v1/health").status_code == 200

    def test_is_exempt_checks_known_paths(self):
        assert "/v1/health" in EXEMPT_PATHS
        # APD-DATA-024: the documentation surface is deliberately NOT exempt.
        # `_is_exempt()` ignores whether a key is configured, so leaving these
        # listed meant that re-enabling `openapi_url` would serve the document to
        # everyone while looking like it sat behind the key.
        assert "/openapi.json" not in EXEMPT_PATHS
        assert "/docs" not in EXEMPT_PATHS
        assert "/redoc" not in EXEMPT_PATHS
        # Prometheus scrape endpoint must be exempt from API-key auth;
        # SEC-16 MetricsAuthMiddleware (IP allowlist) still gates it.
        # Both the bare path and the trailing-slash form are listed because
        # the prometheus_client ASGI sub-app mount triggers a 307 redirect
        # from /metrics to /metrics/ — without the trailing-slash entry,
        # the redirect target re-enters SecurityMiddleware and returns 401.
        assert "/metrics" in EXEMPT_PATHS
        assert "/metrics/" in EXEMPT_PATHS
        assert "/v1/datasets" not in EXEMPT_PATHS


@pytest.mark.unit
class TestFailedAuthThrottle:
    """Unit behaviour of the throttle itself (APD-DATA-001).

    Mirrors the corpus in ``juniper-service-core/tests/test_middleware.py`` so the fork and the
    shared package cannot drift apart silently again.
    """

    def test_check_does_not_consume_budget(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        for _ in range(10):
            assert throttle.check("1.2.3.4") == (False, 0)
        throttle.record_failure("1.2.3.4")
        blocked, retry_after = throttle.check("1.2.3.4")
        assert blocked is True
        assert retry_after >= 1

    def test_is_keyed_per_source_ip(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4")[0] is True
        assert throttle.check("5.6.7.8")[0] is False

    def test_window_rolls_over(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=0)  # every check starts a new window
        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4")[0] is False

    def test_disabled_never_blocks(self):
        throttle = FailedAuthThrottle(max_failures=1, enabled=False)
        for _ in range(10):
            throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4") == (False, 0)

    def test_reset_clears_state(self):
        throttle = FailedAuthThrottle(max_failures=1, window_seconds=60)
        throttle.record_failure("1.2.3.4")
        assert throttle.check("1.2.3.4")[0] is True
        throttle.reset()
        assert throttle.check("1.2.3.4")[0] is False

    def test_prunes_expired_entries(self):
        """Bounded memory: a dict keyed by attacker-supplied source IPs is itself a DoS vector."""
        throttle = FailedAuthThrottle(max_failures=100, window_seconds=0)
        for i in range(FailedAuthThrottle._CLEANUP_INTERVAL + 10):
            throttle.record_failure(f"10.0.0.{i % 255}")
        assert len(throttle._failures) <= FailedAuthThrottle._MAX_ENTRIES

    def test_build_factory_defaults_match_the_documented_budget(self):
        throttle = build_failed_auth_throttle()
        assert throttle.enabled is True
        assert throttle.max_failures == DEFAULT_FAILED_AUTH_MAX_FAILURES
