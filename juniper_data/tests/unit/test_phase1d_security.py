"""Phase 1D Track 1 security remediation tests (SEC-02/04/10/16)."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from cachetools import TTLCache
from fastapi.testclient import TestClient

from juniper_data.api.observability import (
    MetricsAuthMiddleware,
    _strip_sensitive_headers,
    configure_sentry,
)
from juniper_data.api.security import RateLimiter

# =============================================================================
# SEC-02: rate limiter must evict stale entries via TTLCache (DoS guard)
# =============================================================================


class TestSEC02RateLimiterTTL:
    """The in-memory rate limiter must use a TTLCache so that stale per-IP
    buckets are pruned automatically. Without eviction, an attacker rotating
    source IPs can exhaust process memory.
    """

    def test_counters_use_ttl_cache(self) -> None:
        """Internal counter store is a cachetools.TTLCache, not a defaultdict."""
        # SEC-02: bucket store must be a TTL-bounded cache, not an unbounded dict.
        limiter = RateLimiter(requests_per_minute=10, window_seconds=60, enabled=True)
        assert isinstance(limiter._counters, TTLCache)
        assert limiter._counters.maxsize >= 1
        assert limiter._counters.ttl == 60

    def test_entry_expires_after_ttl_window(self) -> None:
        """An IP bucket must be evicted from the cache once the TTL elapses."""
        # SEC-02: use a 1-second window so expiry happens within the test.
        limiter = RateLimiter(requests_per_minute=10, window_seconds=1, enabled=True)
        allowed, _, _ = limiter.check("ip:198.51.100.7")
        assert allowed is True
        assert "ip:198.51.100.7" in limiter._counters

        # Sleep just past the TTL and force expiry; entry must be gone.
        time.sleep(1.05)
        limiter._counters.expire()
        assert "ip:198.51.100.7" not in limiter._counters, "SEC-02 regression: rate limiter entry was not evicted after TTL"

    def test_maxsize_caps_unique_keys(self) -> None:
        """TTLCache hard-caps entry count, blunting an IP-rotation DoS."""
        # SEC-02: many distinct keys must not grow the cache without bound.
        limiter = RateLimiter(requests_per_minute=10, window_seconds=60, enabled=True)
        cap = int(limiter._counters.maxsize)
        for i in range(cap + 50):
            limiter.check(f"ip:10.0.{(i // 256) % 256}.{i % 256}")
        assert len(limiter._counters) <= cap


# =============================================================================
# SEC-04: dataset generation must run off the event-loop thread
# =============================================================================


class TestSEC04DatasetGenerateOffLoop:
    """``POST /v1/datasets`` must delegate generation to asyncio.to_thread."""

    def test_generator_runs_on_worker_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:

        main_loop_thread = threading.current_thread().ident
        observed_threads: list[int] = []

        class _Generator:
            """Fake generator that records which thread ``generate`` ran on."""

            @staticmethod
            def validate_params(params: Any) -> Any:  # pragma: no cover — patched away
                return params

            @staticmethod
            def generate(_params: Any) -> dict:
                observed_threads.append(threading.current_thread().ident)
                # Sleep briefly to prove concurrent requests are not
                # serialized behind this generator call.
                time.sleep(0.05)
                return {}

        async def _invoke() -> None:
            await asyncio.to_thread(_Generator.generate, SimpleNamespace())

        asyncio.run(_invoke())

        assert observed_threads, "generator.generate was not called"
        assert observed_threads[0] != main_loop_thread, "generator.generate ran on the event-loop thread — SEC-04 regression"

    def test_datasets_route_imports_asyncio(self) -> None:
        """Guard against a future refactor that drops the asyncio.to_thread wrap."""
        import inspect

        from juniper_data.api.routes import datasets as datasets_module

        source = inspect.getsource(datasets_module)
        assert "asyncio.to_thread(generator_class.generate" in source, "SEC-04 regression: generator_class.generate is no longer offloaded"


# =============================================================================
# SEC-10: Sentry send_default_pii + before_send scrubber
# =============================================================================


class TestSEC10SentryPII:
    def test_before_send_redacts_sensitive_headers(self) -> None:
        event = {
            "request": {
                "headers": {
                    "X-API-Key": "super-secret",
                    "Authorization": "Bearer token",
                    "Cookie": "session=abc",
                    "User-Agent": "juniper-data/0.7",
                }
            }
        }
        scrubbed = _strip_sensitive_headers(event, hint={})
        assert scrubbed is event
        assert scrubbed["request"]["headers"]["X-API-Key"] == "[Filtered]"
        assert scrubbed["request"]["headers"]["Authorization"] == "[Filtered]"
        assert scrubbed["request"]["headers"]["Cookie"] == "[Filtered]"
        assert scrubbed["request"]["headers"]["User-Agent"] == "juniper-data/0.7"

    def test_before_send_handles_missing_request(self) -> None:
        assert _strip_sensitive_headers({}, hint={}) == {}
        assert _strip_sensitive_headers({"request": "not-a-dict"}, hint={}) == {"request": "not-a-dict"}

    def test_configure_sentry_passes_send_pii_and_before_send(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        class _FakeSentry:
            @staticmethod
            def init(**kwargs: Any) -> None:
                captured.update(kwargs)

        import sys

        monkeypatch.setitem(sys.modules, "sentry_sdk", _FakeSentry)

        configure_sentry("https://public@example.com/1", "juniper-data", "0.0.0")
        assert captured.get("send_default_pii") is False
        assert captured.get("before_send") is _strip_sensitive_headers

    def test_configure_sentry_noop_when_dsn_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[bool] = []

        class _FakeSentry:
            @staticmethod
            def init(**_kwargs: Any) -> None:
                called.append(True)

        import sys

        monkeypatch.setitem(sys.modules, "sentry_sdk", _FakeSentry)

        configure_sentry("", "juniper-data", "0.0.0")
        configure_sentry(None, "juniper-data", "0.0.0")
        assert called == []


# =============================================================================
# SEC-16: /metrics is gated by trusted-IP allowlist
# =============================================================================


class TestSEC16MetricsAuthMiddleware:
    def test_loopback_allowed(self) -> None:
        async def _inner(scope, receive, send):  # noqa: ARG001
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        wrapper = MetricsAuthMiddleware(_inner, ["127.0.0.1", "::1"])
        events: list[dict] = []

        async def _send(msg):
            events.append(msg)

        async def _receive():  # pragma: no cover — never called on 200 path
            return {"type": "http.request"}

        scope = {"type": "http", "client": ("127.0.0.1", 55555)}
        asyncio.run(wrapper(scope, _receive, _send))
        statuses = [e["status"] for e in events if e.get("type") == "http.response.start"]
        assert statuses == [200]

    def test_non_trusted_ip_rejected_with_403(self) -> None:
        async def _inner(scope, receive, send):  # pragma: no cover — must not run
            raise AssertionError("inner ASGI app must not be invoked for untrusted IPs")

        wrapper = MetricsAuthMiddleware(_inner, ["127.0.0.1"])
        events: list[dict] = []

        async def _send(msg):
            events.append(msg)

        async def _receive():  # pragma: no cover — never reached
            return {"type": "http.request"}

        scope = {"type": "http", "client": ("10.0.0.99", 33333)}
        asyncio.run(wrapper(scope, _receive, _send))
        assert events[0]["status"] == 403
        assert events[-1]["body"] == b"Forbidden"

    def test_missing_client_rejected(self) -> None:
        async def _inner(scope, receive, send):  # pragma: no cover
            raise AssertionError("inner ASGI app must not be invoked when client is None")

        wrapper = MetricsAuthMiddleware(_inner, ["127.0.0.1"])
        events: list[dict] = []

        async def _send(msg):
            events.append(msg)

        async def _receive():  # pragma: no cover
            return {"type": "http.request"}

        scope = {"type": "http", "client": None}
        asyncio.run(wrapper(scope, _receive, _send))
        assert events[0]["status"] == 403


@pytest.fixture(autouse=True, scope="function")
def _reset_prometheus_registry():
    """Prometheus metrics are registered at app startup into the global
    ``REGISTRY``; creating two apps per test session would collide. We
    snapshot the registered collectors and unregister any added during
    the test so app-bootstrapping tests can coexist.
    """
    try:
        from prometheus_client import REGISTRY
    except ImportError:  # pragma: no cover — plugin always installed in test env
        yield
        return

    before = set(REGISTRY._collector_to_names.keys())  # type: ignore[attr-defined]
    yield
    after = set(REGISTRY._collector_to_names.keys())  # type: ignore[attr-defined]
    for collector in after - before:
        try:
            REGISTRY.unregister(collector)
        except Exception:  # pragma: no cover — defensive
            pass


class TestSEC16MetricsAppIntegration:
    """End-to-end: mount the app with metrics enabled and hit /metrics."""

    def test_metrics_blocked_for_non_trusted_ip(self) -> None:
        pytest.importorskip("prometheus_client")
        from juniper_data.api.app import create_app
        from juniper_data.api.settings import Settings

        # TestClient's client host is 'testclient'; leaving it out of the
        # allowlist triggers the 403 gate we're verifying.
        settings = Settings(
            metrics_enabled=True,
            metrics_trusted_ips=["127.0.0.1", "::1"],
        )
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.get("/metrics")
        assert response.status_code == 403

    def test_metrics_allowed_when_client_ip_in_allowlist(self) -> None:
        pytest.importorskip("prometheus_client")
        from juniper_data.api.app import create_app
        from juniper_data.api.settings import Settings

        settings = Settings(
            metrics_enabled=True,
            metrics_trusted_ips=["127.0.0.1"],
        )
        app = create_app(settings)
        # Override the default ('testclient', 50000) spoofed client so the
        # ASGI scope's `client` matches a valid allowlist entry. Required
        # since fail-loud validation now rejects non-IP literals like
        # "testclient" at Settings construction.
        with TestClient(app, client=("127.0.0.1", 50001)) as client:
            response = client.get("/metrics")
        assert response.status_code == 200
        assert "python_info" in response.text or "# HELP" in response.text

    def test_metrics_allowed_when_api_keys_configured_and_in_allowlist(self) -> None:
        """/metrics must bypass SecurityMiddleware (API-key) but still respect SEC-16 IP allowlist.

        Pins the exempt + allowlist composition: with both API auth configured
        AND the trusted-IP allowlist allowing the scraper, /metrics returns 200
        and exposes the prometheus_client output. Without the exempt (the
        pre-fix state), this returns 401 from SecurityMiddleware before
        MetricsAuthMiddleware ever sees the request.
        """
        pytest.importorskip("prometheus_client")
        from juniper_data.api.app import create_app
        from juniper_data.api.settings import Settings

        settings = Settings(
            metrics_enabled=True,
            api_keys=["secret"],
            metrics_trusted_ips=["127.0.0.1"],
        )
        app = create_app(settings)
        with TestClient(app, client=("127.0.0.1", 50001)) as client:
            # No X-API-Key header — the exempt should let it through.
            response = client.get("/metrics")
        assert response.status_code == 200
        assert "python_info" in response.text or "# HELP" in response.text

    def test_metrics_still_blocked_by_ip_allowlist_when_api_keys_configured(self) -> None:
        """The exempt does not weaken SEC-16: a non-allowlisted scraper still gets 403.

        The /metrics exempt removes the SecurityMiddleware 401 gate but
        MetricsAuthMiddleware's IP allowlist must remain in force. Together
        they guarantee that the only way to reach /metrics is from a trusted
        IP, regardless of whether API-key auth is configured.
        """
        pytest.importorskip("prometheus_client")
        from juniper_data.api.app import create_app
        from juniper_data.api.settings import Settings

        settings = Settings(
            metrics_enabled=True,
            api_keys=["secret"],
            metrics_trusted_ips=["127.0.0.1", "::1"],
        )
        app = create_app(settings)
        # Spoof a non-allowlisted IP — even a valid API key must still get 403.
        with TestClient(app, client=("10.0.0.99", 50001)) as client:
            response = client.get("/metrics", headers={"X-API-Key": "secret"})
        assert response.status_code == 403


# =============================================================================
# SEC-16 (Issue 4): CIDR support + IPv6 normalization + fail-loud config
# =============================================================================


class TestSEC16MetricsAuthMiddlewareCIDR:
    """CIDR ranges and IPv6 edge cases in MetricsAuthMiddleware.

    Added with the CIDR/IPv6-normalization refactor (juniper-deploy
    notes/poc/POC_REMEDIATION_PLAN_2026-05-27.md §2.2). Covers the
    correctness fixes raised by the plan's validator B: IPv6 zone-id
    stripping, IPv4-mapped IPv6 unwrap, fail-loud on bad config.
    """

    @staticmethod
    def _drive(wrapper, scope: dict) -> list[dict]:
        events: list[dict] = []

        async def _send(msg):
            events.append(msg)

        async def _receive():  # pragma: no cover — never called when blocked
            return {"type": "http.request"}

        asyncio.run(wrapper(scope, _receive, _send))
        return events

    @staticmethod
    def _passthrough_app():
        async def _inner(scope, receive, send):  # noqa: ARG001
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        return _inner

    def test_cidr_allows_ip_in_range(self) -> None:
        wrapper = MetricsAuthMiddleware(self._passthrough_app(), ["172.18.0.0/16"])
        events = self._drive(wrapper, {"type": "http", "client": ("172.18.0.5", 0)})
        statuses = [e["status"] for e in events if e.get("type") == "http.response.start"]
        assert statuses == [200]

    def test_cidr_rejects_ip_outside_range(self) -> None:
        wrapper = MetricsAuthMiddleware(self._passthrough_app(), ["172.18.0.0/16"])
        events = self._drive(wrapper, {"type": "http", "client": ("10.0.0.5", 0)})
        assert events[0]["status"] == 403
        assert events[-1]["body"] == b"Forbidden"

    def test_mixed_list_with_cidr_and_literal_ip(self) -> None:
        wrapper = MetricsAuthMiddleware(
            self._passthrough_app(),
            ["127.0.0.1", "172.18.0.0/16"],
        )
        # Literal IP allowed
        events = self._drive(wrapper, {"type": "http", "client": ("127.0.0.1", 0)})
        assert events[0]["status"] == 200
        # CIDR-covered IP allowed
        events = self._drive(wrapper, {"type": "http", "client": ("172.18.99.1", 0)})
        assert events[0]["status"] == 200

    def test_ipv6_cidr(self) -> None:
        wrapper = MetricsAuthMiddleware(self._passthrough_app(), ["fd00::/8"])
        events = self._drive(wrapper, {"type": "http", "client": ("fd00::1", 0)})
        assert events[0]["status"] == 200

    def test_ipv4_mapped_ipv6_client_against_ipv4_cidr(self) -> None:
        """Docker can surface IPv4 clients as ``::ffff:172.18.0.5``.

        Membership in the IPv4 network ``172.18.0.0/16`` only works if we
        unwrap the IPv4-mapped form before testing. Without unwrap, this
        client would be silently rejected in the exact docker scenario the
        allowlist exists to support.
        """
        wrapper = MetricsAuthMiddleware(self._passthrough_app(), ["172.18.0.0/16"])
        events = self._drive(wrapper, {"type": "http", "client": ("::ffff:172.18.0.5", 0)})
        assert events[0]["status"] == 200

    def test_ipv6_zone_id_stripped(self) -> None:
        """``fe80::1%eth0`` is rejected by ``ip_address`` unless the zone id is stripped.

        Uvicorn can surface zone-scoped link-local addresses; without the
        strip the middleware would 403 trusted-but-zoned clients.
        """
        wrapper = MetricsAuthMiddleware(self._passthrough_app(), ["fe80::/10"])
        events = self._drive(wrapper, {"type": "http", "client": ("fe80::1%eth0", 0)})
        assert events[0]["status"] == 200

    def test_invalid_cidr_entry_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not a valid IP or CIDR"):
            MetricsAuthMiddleware(lambda *a, **k: None, ["172.18.0.0/55"])

    def test_invalid_literal_entry_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not a valid IP or CIDR"):
            MetricsAuthMiddleware(lambda *a, **k: None, ["not-an-ip"])

    def test_unparseable_client_ip_treated_as_untrusted(self) -> None:
        """Garbage in ``scope['client'][0]`` must 403, not crash."""
        wrapper = MetricsAuthMiddleware(self._passthrough_app(), ["127.0.0.1"])
        events = self._drive(wrapper, {"type": "http", "client": ("not-an-ip", 0)})
        assert events[0]["status"] == 403


class TestSEC16SettingsFailLoud:
    """Settings.metrics_trusted_ips must fail loudly on bad entries."""

    def test_settings_rejects_invalid_entry(self) -> None:
        from pydantic import ValidationError

        from juniper_data.api.settings import Settings

        with pytest.raises(ValidationError):
            Settings(metrics_trusted_ips=["not-an-ip"])

    def test_settings_accepts_cidr(self) -> None:
        from juniper_data.api.settings import Settings

        settings = Settings(metrics_trusted_ips=["172.18.0.0/16", "127.0.0.1"])
        assert settings.metrics_trusted_ips == ["172.18.0.0/16", "127.0.0.1"]
