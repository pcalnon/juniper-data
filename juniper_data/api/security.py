"""API security: authentication and rate limiting middleware."""

import hmac
import logging
import time
from threading import Lock

from cachetools import TTLCache
from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

from juniper_data.api.constants import (
    DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    HEADER_RETRY_AFTER,
    HEADER_X_API_KEY,
    HEADER_X_RATELIMIT_LIMIT,
    HEADER_X_RATELIMIT_REMAINING,
    HEADER_X_RATELIMIT_RESET,
)

from .settings import get_settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name=HEADER_X_API_KEY, auto_error=False)

# Rate limiter memory cap. Each unique key consumes an entry until its TTL
# expires; the cap prevents unbounded memory growth under IP-rotation attacks
# (see JD-SEC-03/SEC-02).
RATE_LIMITER_MAX_ENTRIES = 10_000
# Emit a warning once the cache crosses this fraction of capacity so that an
# operator can intervene before eviction starts dropping legitimate entries.
RATE_LIMITER_CAPACITY_WARNING_THRESHOLD = 0.8


class APIKeyAuth:
    """API key authentication handler.

    Validates requests against configured API keys. When no API keys are
    configured, authentication is disabled (open access mode for development).
    """

    def __init__(self, api_keys: list[str] | None = None) -> None:
        """Initialize with optional list of valid API keys.

        Args:
            api_keys: List of valid API keys. If None or empty, auth is disabled.
        """
        # Keys are held in a list rather than a set because validate() compares
        # against each key with hmac.compare_digest to eliminate the timing
        # side-channel that a `value in set` membership test would leak
        # (SEC-01/JD-SEC-02).
        self._api_keys: list[str] = list(dict.fromkeys(api_keys)) if api_keys else []
        self._enabled = len(self._api_keys) > 0

    @property
    def enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self._enabled

    def validate(self, api_key: str | None) -> bool:
        """Validate an API key.

        Args:
            api_key: The API key to validate.

        Returns:
            True if auth is disabled or key is valid, False otherwise.
        """
        if not self._enabled:
            return True
        if api_key is None:
            return False
        # Constant-time comparison against every configured key. `any()` would
        # short-circuit on the first match, but hmac.compare_digest itself runs
        # in time proportional to the input length regardless of where a
        # mismatching byte appears, so iterating the full key list preserves
        # the constant-time property per key while still accepting on a match.
        matched = False
        for candidate in self._api_keys:
            if hmac.compare_digest(api_key, candidate):
                matched = True
        return matched

    async def __call__(self, request: Request) -> str | None:
        """FastAPI dependency for API key validation.

        Args:
            request: The incoming request.

        Returns:
            The validated API key, or None if auth is disabled.

        Raises:
            HTTPException: 401 if auth is enabled and key is invalid/missing.
        """
        api_key = request.headers.get(HEADER_X_API_KEY)

        if not self._enabled:
            return None

        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide X-API-Key header.",
            )

        if not self.validate(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

        return api_key


class RateLimiter:
    """In-memory fixed-window rate limiter.

    Tracks request counts per key within fixed time windows. Thread-safe
    implementation suitable for single-process deployments.
    """

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        enabled: bool = True,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per window.
            window_seconds: Window duration in seconds.
            enabled: Whether rate limiting is enabled.
        """
        self._limit = requests_per_minute
        self._window = window_seconds
        self._enabled = enabled
        # TTLCache provides automatic entry eviction on access once the
        # per-entry TTL elapses, and a hard `maxsize` ceiling so that an
        # attacker rotating source IPs cannot grow the counter dict without
        # bound (JD-SEC-03/SEC-02).
        self._counters: TTLCache[str, tuple[int, float]] = TTLCache(
            maxsize=RATE_LIMITER_MAX_ENTRIES,
            ttl=window_seconds,
        )
        self._capacity_warning_emitted = False
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self._enabled

    @property
    def limit(self) -> int:
        """Get the rate limit."""
        return self._limit

    @property
    def window(self) -> int:
        """Get the window duration in seconds."""
        return self._window

    def _get_key(self, request: Request, api_key: str | None) -> str:
        """Generate a rate limit key for the request.

        Uses API key if available, otherwise falls back to client IP.

        Args:
            request: The incoming request.
            api_key: The authenticated API key, if any.

        Returns:
            A string key for rate limiting.
        """
        if api_key:
            return f"key:{api_key}"
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def check(self, key: str) -> tuple[bool, int, int]:
        """Check if a request is allowed under rate limit.

        Args:
            key: The rate limit key.

        Returns:
            Tuple of (allowed, remaining, reset_seconds).
        """
        if not self._enabled:
            return (True, self._limit, self._window)

        now = time.time()

        with self._lock:
            # TTLCache purges expired entries lazily on access; call expire()
            # explicitly so capacity reporting reflects live entries.
            self._counters.expire()
            self._warn_on_capacity_locked()

            entry = self._counters.get(key)
            if entry is None:
                self._counters[key] = (1, now)
                return (True, self._limit - 1, self._window)

            count, window_start = entry
            if now - window_start >= self._window:
                self._counters[key] = (1, now)
                return (True, self._limit - 1, self._window)

            if count >= self._limit:
                reset_in = int(self._window - (now - window_start))
                return (False, 0, reset_in)

            self._counters[key] = (count + 1, window_start)
            return (True, self._limit - count - 1, int(self._window - (now - window_start)))

    def _warn_on_capacity_locked(self) -> None:
        """Log a one-shot warning when the rate-limiter cache nears capacity.

        Must be called with ``self._lock`` held. The warning is emitted once
        per limiter instance per high-water crossing so logs are not flooded
        while under sustained pressure.
        """
        threshold = int(RATE_LIMITER_MAX_ENTRIES * RATE_LIMITER_CAPACITY_WARNING_THRESHOLD)
        if len(self._counters) >= threshold:
            if not self._capacity_warning_emitted:
                logger.warning(
                    "Rate limiter cache at %.0f%% capacity (%d/%d); eviction may drop legitimate entries soon",
                    len(self._counters) / RATE_LIMITER_MAX_ENTRIES * 100,
                    len(self._counters),
                    RATE_LIMITER_MAX_ENTRIES,
                )
                self._capacity_warning_emitted = True
        else:
            self._capacity_warning_emitted = False

    async def __call__(self, request: Request, api_key: str | None = None) -> None:
        """FastAPI dependency for rate limit checking.

        Args:
            request: The incoming request.
            api_key: The authenticated API key, if any.

        Raises:
            HTTPException: 429 if rate limit exceeded.
        """
        if not self._enabled:
            return

        key = self._get_key(request, api_key)
        allowed, remaining, reset_in = self.check(key)

        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = reset_in

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {reset_in} seconds.",
                headers={
                    HEADER_X_RATELIMIT_LIMIT: str(self._limit),
                    HEADER_X_RATELIMIT_REMAINING: "0",
                    HEADER_X_RATELIMIT_RESET: str(reset_in),
                    HEADER_RETRY_AFTER: str(reset_in),
                },
            )

    def reset(self) -> None:
        """Reset all rate limit counters. Useful for testing."""
        with self._lock:
            self._counters.clear()


_api_key_auth: APIKeyAuth | None = None
_rate_limiter: RateLimiter | None = None


def get_api_key_auth() -> APIKeyAuth:
    """Get the global API key auth handler, creating if needed."""
    global _api_key_auth
    if _api_key_auth is None:
        settings = get_settings()
        api_keys = getattr(settings, "api_keys", None)
        _api_key_auth = APIKeyAuth(api_keys)
    return _api_key_auth


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter, creating if needed."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        enabled = getattr(settings, "rate_limit_enabled", False)
        requests_per_minute = getattr(settings, "rate_limit_requests_per_minute", DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE)
        _rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            enabled=enabled,
        )
    return _rate_limiter


def reset_security_state() -> None:
    """Reset global security state. Useful for testing."""
    global _api_key_auth, _rate_limiter
    _api_key_auth = None
    _rate_limiter = None
