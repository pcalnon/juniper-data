"""FastAPI middleware for security and request processing."""

import logging

from fastapi import HTTPException, Request, Response
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from juniper_data.api.constants import (
    DEFAULT_CONTENT_SECURITY_POLICY,
    HEADER_CONTENT_SECURITY_POLICY,
    HEADER_PERMISSIONS_POLICY,
    HEADER_REFERRER_POLICY,
    HEADER_RETRY_AFTER,
    HEADER_STRICT_TRANSPORT_SECURITY,
    HEADER_X_CONTENT_TYPE_OPTIONS,
    HEADER_X_FORWARDED_PROTO,
    HEADER_X_FRAME_OPTIONS,
    HEADER_X_RATELIMIT_LIMIT,
    HEADER_X_RATELIMIT_REMAINING,
    HEADER_X_RATELIMIT_RESET,
    HSTS_MAX_AGE_VALUE,
    PERMISSIONS_POLICY_RESTRICTED,
    PROXY_PROTOCOL_HTTPS,
    REFERRER_POLICY_STRICT_ORIGIN,
    X_CONTENT_TYPE_OPTIONS_NOSNIFF,
    X_FRAME_OPTIONS_DENY,
)
from juniper_data.api.constants import (
    EXEMPT_PATHS as _EXEMPT_PATHS_CONST,
)
from juniper_data.api.constants import (
    MAX_REQUEST_BODY_BYTES as _MAX_REQUEST_BODY_BYTES_CONST,
)

from .security import APIKeyAuth, FailedAuthThrottle, RateLimiter, build_failed_auth_throttle

logger = logging.getLogger(__name__)

# Module-level aliases preserved for tests that may import these names directly.
# The canonical source of truth is :mod:`juniper_data.api.constants`.
EXEMPT_PATHS = _EXEMPT_PATHS_CONST
_DEFAULT_CSP = DEFAULT_CONTENT_SECURITY_POLICY
_MAX_REQUEST_BODY_BYTES = _MAX_REQUEST_BODY_BYTES_CONST


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses.

    Injects standard security headers (X-Content-Type-Options, X-Frame-Options,
    Referrer-Policy, Permissions-Policy, CSP, and conditional HSTS) into every
    HTTP response.
    """

    def __init__(self, app: ASGIApp, content_security_policy: str = _DEFAULT_CSP) -> None:
        super().__init__(app)
        self._csp = content_security_policy

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        response.headers[HEADER_X_CONTENT_TYPE_OPTIONS] = X_CONTENT_TYPE_OPTIONS_NOSNIFF
        response.headers[HEADER_X_FRAME_OPTIONS] = X_FRAME_OPTIONS_DENY
        response.headers[HEADER_REFERRER_POLICY] = REFERRER_POLICY_STRICT_ORIGIN
        response.headers[HEADER_PERMISSIONS_POLICY] = PERMISSIONS_POLICY_RESTRICTED
        response.headers[HEADER_CONTENT_SECURITY_POLICY] = self._csp

        # Only add HSTS when the request arrived over TLS (via reverse proxy)
        if request.headers.get(HEADER_X_FORWARDED_PROTO) == PROXY_PROTOCOL_HTTPS:
            response.headers[HEADER_STRICT_TRANSPORT_SECURITY] = HSTS_MAX_AGE_VALUE

        return response


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds a configurable limit.

    ``Content-Length`` is an early-reject hint only and is not trusted as the
    sole size check (CR-024): a malicious client can under-declare or omit the
    header and send an unbounded chunked stream. For POST/PUT/PATCH requests we
    always stream-read the body with a cumulative byte cap, aborting with HTTP
    413 as soon as the cap is exceeded. This prevents the classic
    chunked-encoding memory-exhaustion bypass in which ``await request.body()``
    would allocate the entire body before any size check runs.

    The fully-read body is cached on ``request._body`` so downstream FastAPI
    route handlers can consume it via ``request.body()`` / ``request.json()``
    / pydantic body parsing without triggering a second read.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = _MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Fast-path early reject on declared Content-Length. Still untrusted
        # as a floor, so the stream-read below enforces the real limit.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                # A malformed header is a client error. Without this guard the
                # ValueError escapes BaseHTTPMiddleware.dispatch -- outside
                # ExceptionMiddleware, so the app's own ValueError handler never
                # sees it -- and surfaces as a 500.
                return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": "Invalid Content-Length header"})
            if declared_length > self._max_bytes:
                return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": "Request body too large"})
        if request.method in ("POST", "PUT", "PATCH"):
            # CR-024: always stream-read mutating methods -- Content-Length is an
            # early-reject hint only. An under-declared CL with a larger real body,
            # or a chunked stream with no CL at all, must still hit the cumulative
            # cap; skipping the stream when CL is present-and-small is the classic
            # bypass.
            chunks: list[bytes] = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > self._max_bytes:
                    return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": "Request body too large"})
                chunks.append(chunk)
            # Cache body for downstream handlers. Starlette's
            # ``BaseHTTPMiddleware._CachedRequest.wrapped_receive`` short-
            # circuits to a synthetic ``http.request`` message constructed
            # from ``self._body`` when that attribute is set, so subsequent
            # ``await request.body()`` / ``request.json()`` / Pydantic body
            # parsing in downstream handlers all see the cached payload.
            request._body = b"".join(chunks)
        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and rate limiting.

    Applies authentication and rate limiting to all requests except
    explicitly exempt paths (health checks, docs).

    The identity-keyed :class:`~juniper_data.api.security.RateLimiter` runs *after*
    authentication and is therefore never reached when auth raises, so before APD-DATA-001 the
    401 path consumed no budget at all and credential guessing was unthrottled. The fix is not
    to reorder -- that trades a real protection for a worse one, collapsing every authenticated
    caller behind one NAT into a single ``ip:`` bucket -- but to add a coarse, IP-keyed
    :class:`~juniper_data.api.security.FailedAuthThrottle` *ahead* of authentication. It only
    consumes budget on a failed attempt, so a caller with a valid key is never counted and
    well-behaved traffic is unaffected.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key_auth: APIKeyAuth,
        rate_limiter: RateLimiter,
        failed_auth_throttle: FailedAuthThrottle | None = None,
    ) -> None:
        """Initialize the security middleware.

        Args:
            app: The ASGI application.
            api_key_auth: API key authentication handler.
            rate_limiter: Rate limiter instance.
            failed_auth_throttle: Pre-authentication, IP-keyed throttle for failed attempts.
                Defaults to an enabled throttle at the library default budget. Pass
                ``build_failed_auth_throttle(enabled=False)`` to opt out. Defaulting to enabled
                is safe because budget is consumed only on a *failed* attempt, so a caller with
                valid credentials never sees a behaviour change.
        """
        super().__init__(app)
        self._api_key_auth = api_key_auth
        self._rate_limiter = rate_limiter
        self._failed_auth_throttle = failed_auth_throttle if failed_auth_throttle is not None else build_failed_auth_throttle()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request through security checks.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The response from the application.
        """
        path = request.url.path

        if self._is_exempt(path):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        # Pre-authentication throttle. Checked first so an IP already over its failed-attempt
        # budget is rejected without burning an auth comparison, and -- crucially -- so the
        # rejection happens on a path that auth failure cannot skip past.
        if self._failed_auth_throttle.enabled:
            blocked, retry_after = self._failed_auth_throttle.check(client_ip)
            if blocked:
                logger.warning("Too many failed authentication attempts from %s; throttled for %ss", client_ip, retry_after)
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": f"Too many failed authentication attempts. Try again in {retry_after} seconds."},
                    headers={HEADER_RETRY_AFTER: str(retry_after)},
                )

        api_key = None
        try:
            if self._api_key_auth.enabled:
                api_key = await self._api_key_auth(request)

            if self._rate_limiter.enabled:
                await self._rate_limiter(request, api_key)
        except HTTPException as exc:
            # Record the attempt only for authentication failures. A 429 from the identity-keyed
            # limiter is a quota outcome, not a credential guess, and counting it here would let
            # a caller throttle itself out of the auth path by exceeding its own quota.
            if exc.status_code == status.HTTP_401_UNAUTHORIZED and self._failed_auth_throttle.enabled:
                self._failed_auth_throttle.record_failure(client_ip)
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )

        response = await call_next(request)

        if self._rate_limiter.enabled and hasattr(request.state, "rate_limit_remaining"):
            response.headers[HEADER_X_RATELIMIT_LIMIT] = str(self._rate_limiter.limit)
            response.headers[HEADER_X_RATELIMIT_REMAINING] = str(request.state.rate_limit_remaining)
            response.headers[HEADER_X_RATELIMIT_RESET] = str(request.state.rate_limit_reset)

        return response

    def _is_exempt(self, path: str) -> bool:
        """Check if a path is exempt from security checks.

        Args:
            path: The request path.

        Returns:
            True if the path is exempt, False otherwise.
        """
        return path in EXEMPT_PATHS
