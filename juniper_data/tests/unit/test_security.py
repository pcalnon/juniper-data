"""Unit tests for API security: authentication and rate limiting."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from juniper_data.api.security import (
    RATE_LIMITER_MAX_ENTRIES,
    APIKeyAuth,
    RateLimiter,
)


class TestAPIKeyAuth:
    """Tests for APIKeyAuth class."""

    def test_disabled_when_no_keys(self) -> None:
        """Auth should be disabled when no keys are configured."""
        auth = APIKeyAuth(None)
        assert not auth.enabled

        auth = APIKeyAuth([])
        assert not auth.enabled

    def test_enabled_when_keys_configured(self) -> None:
        """Auth should be enabled when keys are configured."""
        auth = APIKeyAuth(["key1", "key2"])
        assert auth.enabled

    def test_validate_returns_true_when_disabled(self) -> None:
        """Validate should return True when auth is disabled."""
        auth = APIKeyAuth(None)
        assert auth.validate(None)
        assert auth.validate("any-key")

    def test_validate_valid_key(self) -> None:
        """Validate should return True for valid key."""
        auth = APIKeyAuth(["valid-key"])
        assert auth.validate("valid-key")

    def test_validate_invalid_key(self) -> None:
        """Validate should return False for invalid key."""
        auth = APIKeyAuth(["valid-key"])
        assert not auth.validate("invalid-key")
        assert not auth.validate(None)

    @pytest.mark.asyncio
    async def test_call_returns_none_when_disabled(self) -> None:
        """Dependency should return None when auth is disabled."""
        auth = APIKeyAuth(None)
        request = MagicMock()
        request.headers.get.return_value = None

        result = await auth(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_call_raises_401_when_missing_key(self) -> None:
        """Dependency should raise 401 when key is missing."""
        auth = APIKeyAuth(["valid-key"])
        request = MagicMock()
        request.headers.get.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401
        assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_raises_401_when_invalid_key(self) -> None:
        """Dependency should raise 401 when key is invalid."""
        auth = APIKeyAuth(["valid-key"])
        request = MagicMock()
        request.headers.get.return_value = "invalid-key"

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_returns_key_when_valid(self) -> None:
        """Dependency should return the key when valid."""
        auth = APIKeyAuth(["valid-key"])
        request = MagicMock()
        request.headers.get.return_value = "valid-key"

        result = await auth(request)
        assert result == "valid-key"

    def test_validate_multiple_keys(self) -> None:
        """SEC-01/JD-SEC-02: all configured keys should validate."""
        auth = APIKeyAuth(["alpha-key", "beta-key", "gamma-key"])
        assert auth.validate("alpha-key")
        assert auth.validate("beta-key")
        assert auth.validate("gamma-key")
        assert not auth.validate("delta-key")

    def test_validate_rejects_prefix_match(self) -> None:
        """SEC-01/JD-SEC-02: prefix of a valid key must not validate."""
        auth = APIKeyAuth(["valid-key-1234"])
        assert not auth.validate("valid-key")
        assert not auth.validate("valid-key-123")

    def test_validate_uses_constant_time_comparison(self) -> None:
        """SEC-01/JD-SEC-02: validate must delegate to hmac.compare_digest.

        This confirms the implementation no longer relies on Python's
        short-circuiting equality; breaking this assertion means a future
        refactor has reintroduced the timing side-channel.
        """
        from juniper_data.api import security as security_module

        auth = APIKeyAuth(["valid-key"])
        calls: list[tuple[str, str]] = []
        original = security_module.hmac.compare_digest

        def spy(a: str, b: str) -> bool:
            calls.append((a, b))
            return original(a, b)

        security_module.hmac.compare_digest = spy  # type: ignore[assignment]
        try:
            auth.validate("probe-key")
        finally:
            security_module.hmac.compare_digest = original  # type: ignore[assignment]

        assert calls, "validate() must call hmac.compare_digest"
        assert all(candidate == "valid-key" for _probe, candidate in calls)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_disabled_allows_all(self) -> None:
        """Disabled limiter should allow all requests."""
        limiter = RateLimiter(requests_per_minute=5, enabled=False)

        for _ in range(100):
            allowed, remaining, _ = limiter.check("key")
            assert allowed

    def test_allows_within_limit(self) -> None:
        """Limiter should allow requests within limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for i in range(5):
            allowed, remaining, _ = limiter.check("key")
            assert allowed
            assert remaining == 5 - i - 1

    def test_blocks_over_limit(self) -> None:
        """Limiter should block requests over limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for _ in range(5):
            limiter.check("key")

        allowed, remaining, reset_in = limiter.check("key")
        assert not allowed
        assert remaining == 0
        assert reset_in > 0

    def test_different_keys_tracked_separately(self) -> None:
        """Different keys should have separate limits."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for _ in range(5):
            limiter.check("key1")

        allowed1, _, _ = limiter.check("key1")
        allowed2, _, _ = limiter.check("key2")

        assert not allowed1
        assert allowed2

    def test_window_reset(self) -> None:
        """Window should reset after time expires."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=1, enabled=True)

        for _ in range(5):
            limiter.check("key")

        allowed, _, _ = limiter.check("key")
        assert not allowed

        time.sleep(1.1)

        allowed, remaining, _ = limiter.check("key")
        assert allowed
        assert remaining == 4

    def test_reset_clears_counters(self) -> None:
        """Reset should clear all counters."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)

        for _ in range(5):
            limiter.check("key")

        allowed, _, _ = limiter.check("key")
        assert not allowed

        limiter.reset()

        allowed, _, _ = limiter.check("key")
        assert allowed

    @pytest.mark.asyncio
    async def test_call_allows_when_within_limit(self) -> None:
        """Dependency should allow requests within limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        for _ in range(5):
            await limiter(request, api_key=None)

    @pytest.mark.asyncio
    async def test_call_raises_429_when_over_limit(self) -> None:
        """Dependency should raise 429 when over limit."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        for _ in range(5):
            await limiter(request, api_key=None)

        with pytest.raises(HTTPException) as exc_info:
            await limiter(request, api_key=None)
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_uses_api_key_for_limiting(self) -> None:
        """Dependency should use API key for rate limiting when provided."""
        limiter = RateLimiter(requests_per_minute=5, enabled=True)
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.state = MagicMock()

        for _ in range(5):
            await limiter(request, api_key="key1")

        with pytest.raises(HTTPException):
            await limiter(request, api_key="key1")

        await limiter(request, api_key="key2")

    def test_rate_limiter_window_property(self) -> None:
        """Window property should return configured window seconds."""
        limiter = RateLimiter(requests_per_minute=10, window_seconds=30)
        assert limiter.window == 30

    def test_get_key_with_no_client(self) -> None:
        """_get_key should return 'ip:unknown' when request has no client."""
        limiter = RateLimiter()
        request = MagicMock()
        request.client = None
        key = limiter._get_key(request, None)
        assert key == "ip:unknown"

    @pytest.mark.asyncio
    async def test_call_noop_when_disabled(self) -> None:
        """Dependency should do nothing when disabled."""
        limiter = RateLimiter(requests_per_minute=5, enabled=False)
        request = MagicMock()
        request.client.host = "127.0.0.1"

        for _ in range(100):
            await limiter(request, api_key=None)

    def test_counter_evicts_after_ttl(self) -> None:
        """JD-SEC-03: expired entries must be evicted by TTLCache."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=1, enabled=True)
        limiter.check("ephemeral-key")
        assert len(limiter._counters) == 1
        time.sleep(1.1)
        # A second check on a different key forces TTLCache to expire the
        # stale entry on access.
        limiter.check("fresh-key")
        assert "ephemeral-key" not in limiter._counters

    def test_counter_bounded_by_max_entries(self) -> None:
        """JD-SEC-03: counter dict must never exceed the configured ceiling."""
        limiter = RateLimiter(requests_per_minute=5, window_seconds=60, enabled=True)
        # Generate many more unique keys than the cache capacity allows.
        for i in range(RATE_LIMITER_MAX_ENTRIES + 250):
            limiter.check(f"ip:{i}")
        assert len(limiter._counters) <= RATE_LIMITER_MAX_ENTRIES

    def test_capacity_warning_emitted_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """JD-SEC-03: near-capacity should produce a single warning per crossing."""
        import logging as _logging

        limiter = RateLimiter(requests_per_minute=5, window_seconds=60, enabled=True)
        threshold_count = int(RATE_LIMITER_MAX_ENTRIES * 0.82)
        with caplog.at_level(_logging.WARNING, logger="juniper_data.api.security"):
            for i in range(threshold_count):
                limiter.check(f"ip:{i}")
            # Trigger another check to give the warning a second opportunity to
            # fire — it must not double-log.
            limiter.check("extra-ip")
        warnings = [rec for rec in caplog.records if "Rate limiter cache" in rec.message]
        assert len(warnings) == 1


class TestSecurityModuleFunctions:
    """Tests for module-level security functions."""

    def test_get_api_key_auth_returns_instance(self) -> None:
        """get_api_key_auth should return an APIKeyAuth instance."""
        from juniper_data.api.security import get_api_key_auth, reset_security_state

        reset_security_state()
        auth = get_api_key_auth()
        assert isinstance(auth, APIKeyAuth)

    def test_get_api_key_auth_returns_same_instance(self) -> None:
        """get_api_key_auth should return same instance on second call."""
        from juniper_data.api.security import get_api_key_auth, reset_security_state

        reset_security_state()
        auth1 = get_api_key_auth()
        auth2 = get_api_key_auth()
        assert auth1 is auth2

    def test_get_rate_limiter_returns_instance(self) -> None:
        """get_rate_limiter should return a RateLimiter instance."""
        from juniper_data.api.security import get_rate_limiter, reset_security_state

        reset_security_state()
        limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_get_rate_limiter_returns_same_instance(self) -> None:
        """get_rate_limiter should return same instance on second call."""
        from juniper_data.api.security import get_rate_limiter, reset_security_state

        reset_security_state()
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_reset_security_state(self) -> None:
        """reset_security_state should clear cached instances."""
        from juniper_data.api.security import get_api_key_auth, get_rate_limiter, reset_security_state

        reset_security_state()
        auth1 = get_api_key_auth()
        limiter1 = get_rate_limiter()
        reset_security_state()
        auth2 = get_api_key_auth()
        limiter2 = get_rate_limiter()
        assert auth1 is not auth2
        assert limiter1 is not limiter2
