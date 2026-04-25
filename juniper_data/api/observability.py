"""Observability module for structured logging, Prometheus metrics, and Sentry integration."""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from juniper_data.api.constants import (
    DATASET_GENERATION_DURATION_BUCKETS,
    DEFAULT_LOG_FORMAT_PLAIN,
    DEFAULT_NAMESPACE,
    DEFAULT_SENTRY_TRACES_SAMPLE_RATE,
    DEFAULT_SERVICE_NAME,
    GENERATION_STATUS_SUCCESS,
    HEADER_X_REQUEST_ID,
    LOG_FORMAT_JSON,
)

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_SERVICE_NAME_DEFAULT: str = DEFAULT_SERVICE_NAME
_NAMESPACE_DEFAULT: str = DEFAULT_NAMESPACE


class JuniperJsonFormatter(logging.Formatter):
    """JSON log formatter with request_id propagation."""

    def __init__(self, service: str = _SERVICE_NAME_DEFAULT) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service,
            "request_id": request_id_var.get(""),
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID into ContextVar and response header."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        rid = request.headers.get(HEADER_X_REQUEST_ID, str(uuid.uuid4()))
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers[HEADER_X_REQUEST_ID] = rid
            return response
        finally:
            request_id_var.reset(token)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Tracks http_requests_total and http_request_duration_seconds with namespace prefix."""

    def __init__(self, app: object, service_name: str = _SERVICE_NAME_DEFAULT, namespace: str = _NAMESPACE_DEFAULT) -> None:
        super().__init__(app)
        from prometheus_client import Counter, Histogram

        prefix = f"{namespace}_" if namespace else ""
        self._request_count = Counter(
            f"{prefix}http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )
        self._request_duration = Histogram(
            f"{prefix}http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # BUG-JD-09: use route template for fixed cardinality (avoid unbounded labels from path params)
        route = request.scope.get("route")
        endpoint = route.path if route is not None and hasattr(route, "path") else request.url.path
        method = request.method
        status = str(response.status_code)

        self._request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self._request_duration.labels(method=method, endpoint=endpoint).observe(duration)

        return response


def configure_logging(log_level: str, log_format: str, service_name: str = _SERVICE_NAME_DEFAULT) -> None:
    """Configure logging — JSON when log_format='json', plain text otherwise.

    Args:
        log_level: Logging level string (e.g. "INFO", "DEBUG").
        log_format: Format mode — "json" for structured JSON, anything else for plain text.
        service_name: Service name included in JSON log entries.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if log_format == LOG_FORMAT_JSON:
        handler.setFormatter(JuniperJsonFormatter(service=service_name))
    else:
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT_PLAIN))

    root.addHandler(handler)


# SEC-10: header names that may carry API keys or session identifiers.
# ``before_send`` scrubs these from every Sentry event regardless of the
# ``send_default_pii`` flag, so a future integration that re-enables
# per-event header capture (custom logging integration, replay, etc.)
# cannot leak authentication material to Sentry.
_SENTRY_SENSITIVE_HEADERS = frozenset({"x-api-key", "authorization", "cookie"})


def _strip_sensitive_headers(event, hint):  # noqa: ARG001 — Sentry hook signature
    """Redact sensitive request headers in a Sentry event with ``[Filtered]``.

    Sentry calls this via ``before_send`` for every outbound event. The
    filter only rewrites keys in ``_SENTRY_SENSITIVE_HEADERS`` so
    non-sensitive diagnostic headers (user-agent, trace IDs, etc.) still
    reach Sentry unchanged.
    """
    request_data = event.get("request", {}) if isinstance(event, dict) else {}
    headers = request_data.get("headers", {}) if isinstance(request_data, dict) else {}
    if isinstance(headers, dict):
        for key in list(headers.keys()):
            if key.lower() in _SENTRY_SENSITIVE_HEADERS:
                headers[key] = "[Filtered]"
    return event


def configure_sentry(
    dsn: str | None,
    service_name: str,
    version: str,
    *,
    send_pii: bool = False,
    traces_sample_rate: float = DEFAULT_SENTRY_TRACES_SAMPLE_RATE,
) -> None:
    """Initialize Sentry with FastAPI integration. No-op when dsn is None or empty.

    Args:
        dsn: Sentry DSN URL. Pass None or empty string to skip initialization.
        service_name: Service name for Sentry environment tag.
        version: Application version string.
        send_pii: Whether to send default PII (IP addresses, etc.) to Sentry.
            Defaults to False (SEC-10); operators can opt in explicitly via
            ``JUNIPER_DATA_SENTRY_SEND_PII=true`` when they accept the risk.
        traces_sample_rate: Fraction of transactions to send to Sentry (0.0 to 1.0).
    """
    if not dsn:
        return

    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        # SEC-10: honor the operator choice but still run ``before_send``
        # so that when ``send_pii=True`` is set intentionally, API keys
        # never hit Sentry regardless. With the default ``send_pii=False``
        # nothing is sent anyway; the filter acts as defense-in-depth.
        send_default_pii=send_pii,
        enable_logs=True,
        traces_sample_rate=traces_sample_rate,
        release=f"{service_name}@{version}",
        before_send=_strip_sensitive_headers,
    )


def get_prometheus_app():
    """Return ASGI app for /metrics endpoint via prometheus_client.make_asgi_app().

    Returns:
        ASGI application serving Prometheus metrics.
    """
    from prometheus_client import make_asgi_app

    return make_asgi_app()


# SEC-16: default allowlist matches ``Settings.metrics_trusted_ips``. We
# duplicate it as a module-level default so ``MetricsAuthMiddleware`` can
# be constructed without passing trusted IPs explicitly (useful in tests).
METRICS_DEFAULT_TRUSTED_IPS = ("127.0.0.1", "::1")


class MetricsAuthMiddleware:
    """ASGI wrapper that restricts ``/metrics`` to a trusted IP allowlist.

    The Prometheus ASGI app is mounted as a sub-app and therefore bypasses
    ``SecurityMiddleware`` (which only runs on the router stack). This
    wrapper inspects the raw ASGI scope so it can reject untrusted scrapers
    without depending on FastAPI. An empty allowlist blocks everything —
    operators who don't want ``/metrics`` exposed should simply flip
    ``metrics_enabled=False`` instead.
    """

    def __init__(self, app, trusted_ips: list[str] | tuple[str, ...] | None = None) -> None:
        self.app = app
        self.trusted_ips: frozenset[str] = frozenset(trusted_ips if trusted_ips is not None else METRICS_DEFAULT_TRUSTED_IPS)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client = scope.get("client")
            client_ip = client[0] if client else None
            if not client_ip or client_ip not in self.trusted_ips:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                    }
                )
                await send({"type": "http.response.body", "body": b"Forbidden"})
                return
        await self.app(scope, receive, send)


def set_build_info(namespace: str, version: str) -> None:
    """Set build information as a Prometheus Info metric.

    Args:
        namespace: Metric namespace prefix (e.g. "juniper_data").
        version: Application version string.
    """
    from prometheus_client import Info

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info = Info(f"{namespace}_build", f"Build information for {namespace.replace('_', '-')} service")
    info.info({"version": version, "python_version": python_version})


# ---------------------------------------------------------------------------
# Custom application metrics — lazily initialized to avoid requiring
# prometheus_client at import time (it is an optional dependency).
# ---------------------------------------------------------------------------

_dataset_metrics: dict | None = None


def _ensure_dataset_metrics() -> dict:
    """Create dataset-related Prometheus metrics on first access."""
    global _dataset_metrics
    if _dataset_metrics is None:
        from prometheus_client import Counter, Gauge, Histogram

        _dataset_metrics = {
            "generations_total": Counter(
                "juniper_data_dataset_generations_total",
                "Total dataset generation requests",
                ["generator", "status"],
            ),
            "generation_duration_seconds": Histogram(
                "juniper_data_dataset_generation_duration_seconds",
                "Dataset generation duration in seconds",
                ["generator"],
                buckets=DATASET_GENERATION_DURATION_BUCKETS,
            ),
            "datasets_cached": Gauge(
                "juniper_data_datasets_cached",
                "Number of datasets currently cached in storage",
            ),
        }
    return _dataset_metrics


def record_dataset_generation(generator: str, status: str, duration: float) -> None:
    """Record a dataset generation event in Prometheus metrics.

    Args:
        generator: Generator type name (e.g. "spiral").
        status: Outcome — "success" or "error".
        duration: Generation duration in seconds.
    """
    m = _ensure_dataset_metrics()
    m["generations_total"].labels(generator=generator, status=status).inc()
    if status == GENERATION_STATUS_SUCCESS:
        m["generation_duration_seconds"].labels(generator=generator).observe(duration)


def set_datasets_cached(count: int) -> None:
    """Update the cached datasets gauge.

    Args:
        count: Current number of datasets in cache/storage.
    """
    _ensure_dataset_metrics()["datasets_cached"].set(count)
