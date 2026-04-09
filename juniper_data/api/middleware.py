"""FastAPI middleware for security and request processing."""

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

from .security import APIKeyAuth, RateLimiter

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
    """Reject requests whose Content-Length exceeds a configurable limit."""

    def __init__(self, app: ASGIApp, max_bytes: int = _MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self._max_bytes:
            return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": "Request body too large"})
        return await call_next(request)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and rate limiting.

    Applies authentication and rate limiting to all requests except
    explicitly exempt paths (health checks, docs).
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key_auth: APIKeyAuth,
        rate_limiter: RateLimiter,
    ) -> None:
        """Initialize the security middleware.

        Args:
            app: The ASGI application.
            api_key_auth: API key authentication handler.
            rate_limiter: Rate limiter instance.
        """
        super().__init__(app)
        self._api_key_auth = api_key_auth
        self._rate_limiter = rate_limiter

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

        api_key = None
        try:
            if self._api_key_auth.enabled:
                api_key = await self._api_key_auth(request)

            if self._rate_limiter.enabled:
                await self._rate_limiter(request, api_key)
        except HTTPException as exc:
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
