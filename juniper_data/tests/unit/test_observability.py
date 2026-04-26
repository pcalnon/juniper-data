"""Unit tests for the observability module."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from juniper_data.api.observability import (
    UNMATCHED_ENDPOINT_LABEL,
    JuniperJsonFormatter,
    PrometheusMiddleware,
    RequestIdMiddleware,
    configure_logging,
    configure_sentry,
    get_prometheus_app,
    record_dataset_generation,
    request_id_var,
    set_build_info,
    set_datasets_cached,
)


@pytest.mark.unit
class TestJuniperJsonFormatter:
    """Tests for JuniperJsonFormatter."""

    def test_format_produces_valid_json(self):
        formatter = JuniperJsonFormatter(service="test-service")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["service"] == "test-service"
        assert "timestamp" in parsed
        assert "request_id" in parsed

    def test_format_includes_request_id_from_contextvar(self):
        formatter = JuniperJsonFormatter(service="test-service")
        token = request_id_var.set("abc-123")
        try:
            record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg="hi", args=None, exc_info=None)
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["request_id"] == "abc-123"
        finally:
            request_id_var.reset(token)

    def test_format_includes_exception_info(self):
        formatter = JuniperJsonFormatter(service="test-service")
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(name="test", level=logging.ERROR, pathname="", lineno=0, msg="error", args=None, exc_info=exc_info)
            output = formatter.format(record)
            parsed = json.loads(output)
            assert "exception" in parsed
            assert "ValueError" in parsed["exception"]

    def test_format_default_service_name(self):
        formatter = JuniperJsonFormatter()
        record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg="hi", args=None, exc_info=None)
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["service"] == "juniper-data"


@pytest.mark.unit
class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self):
        """Reset root logger before each test."""
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_text_mode_uses_standard_formatter(self):
        configure_logging("INFO", "text", "test-service")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert not isinstance(handler.formatter, JuniperJsonFormatter)

    def test_json_mode_uses_json_formatter(self):
        configure_logging("INFO", "json", "test-service")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JuniperJsonFormatter)

    def test_sets_log_level(self):
        configure_logging("DEBUG", "text", "test-service")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_removes_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        stream_handlers_before = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and type(h) is logging.StreamHandler]
        assert len(stream_handlers_before) == 2
        configure_logging("INFO", "text", "test-service")
        # configure_logging removes all handlers and adds exactly one
        assert len(root.handlers) == 1


@pytest.mark.unit
class TestConfigureSentry:
    """Tests for configure_sentry function."""

    def test_noop_when_dsn_is_none(self):
        configure_sentry(None, "test-service", "1.0.0")

    def test_noop_when_dsn_is_empty(self):
        configure_sentry("", "test-service", "1.0.0")

    def test_initializes_when_dsn_provided(self):
        pytest.importorskip("sentry_sdk")
        with patch("sentry_sdk.init") as mock_init:
            configure_sentry("https://examplePublicKey@o0.ingest.sentry.io/0", "test-service", "1.0.0")
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["dsn"] == "https://examplePublicKey@o0.ingest.sentry.io/0"
            assert call_kwargs["release"] == "test-service@1.0.0"
            assert call_kwargs["send_default_pii"] is False
            assert call_kwargs["traces_sample_rate"] == 0.1

    def test_send_pii_enabled(self):
        pytest.importorskip("sentry_sdk")
        with patch("sentry_sdk.init") as mock_init:
            configure_sentry("https://examplePublicKey@o0.ingest.sentry.io/0", "test-service", "1.0.0", send_pii=True)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["send_default_pii"] is True

    def test_custom_traces_sample_rate(self):
        pytest.importorskip("sentry_sdk")
        with patch("sentry_sdk.init") as mock_init:
            configure_sentry("https://examplePublicKey@o0.ingest.sentry.io/0", "test-service", "1.0.0", traces_sample_rate=0.5)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["traces_sample_rate"] == 0.5


@pytest.mark.unit
class TestRequestIdMiddleware:
    """Tests for RequestIdMiddleware."""

    @pytest.mark.asyncio
    async def test_generates_request_id_when_not_provided(self):
        middleware = RequestIdMiddleware(app=MagicMock())
        captured_rid = None

        async def mock_call_next(request):
            nonlocal captured_rid
            captured_rid = request_id_var.get("")
            response = MagicMock()
            response.headers = {}
            return response

        request = MagicMock()
        request.headers = {}
        middleware.dispatch = RequestIdMiddleware.dispatch.__get__(middleware, RequestIdMiddleware)

        response = await middleware.dispatch(request, mock_call_next)
        assert captured_rid != ""
        assert "X-Request-ID" in response.headers

    @pytest.mark.asyncio
    async def test_uses_provided_request_id(self):
        middleware = RequestIdMiddleware(app=MagicMock())
        captured_rid = None

        async def mock_call_next(request):
            nonlocal captured_rid
            captured_rid = request_id_var.get("")
            response = MagicMock()
            response.headers = {}
            return response

        request = MagicMock()
        request.headers = {"X-Request-ID": "custom-id-123"}

        response = await middleware.dispatch(request, mock_call_next)
        assert captured_rid == "custom-id-123"
        assert response.headers["X-Request-ID"] == "custom-id-123"


@pytest.mark.unit
class TestPrometheusMiddleware:
    """Tests for PrometheusMiddleware."""

    @staticmethod
    def _build_request(*, method: str, url_path: str, route_template: str | None) -> MagicMock:
        request = MagicMock()
        request.url.path = url_path
        request.method = method
        if route_template is None:
            request.scope = {}
        else:
            route = MagicMock()
            route.path = route_template
            request.scope = {"route": route}
        return request

    @pytest.mark.asyncio
    async def test_matched_route_uses_template_for_endpoint_label(self):
        """When Starlette resolves a route, the endpoint label is the template, not the raw URL."""
        pytest.importorskip("prometheus_client")
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            request_count = MagicMock()
            unmatched_count = MagicMock()
            MockCounter.side_effect = [request_count, unmatched_count]
            mock_histogram = MagicMock()
            MockHistogram.return_value = mock_histogram

            middleware = PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_data")

            response = MagicMock()
            response.status_code = 200

            async def mock_call_next(request):
                return response

            request = self._build_request(method="GET", url_path="/v1/datasets/abc-123", route_template="/v1/datasets/{dataset_id}")
            result = await middleware.dispatch(request, mock_call_next)

            request_count.labels.assert_called_once_with(method="GET", endpoint="/v1/datasets/{dataset_id}", status="200")
            request_count.labels().inc.assert_called_once()
            mock_histogram.labels.assert_called_once_with(method="GET", endpoint="/v1/datasets/{dataset_id}")
            mock_histogram.labels().observe.assert_called_once()
            unmatched_count.labels.assert_not_called()
            assert result == response

    @pytest.mark.asyncio
    async def test_unmatched_route_collapses_to_single_label(self):
        """No resolved route → endpoint label collapses to UNMATCHED_ENDPOINT_LABEL and unmatched counter increments."""
        pytest.importorskip("prometheus_client")
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            request_count = MagicMock()
            unmatched_count = MagicMock()
            MockCounter.side_effect = [request_count, unmatched_count]
            MockHistogram.return_value = MagicMock()

            middleware = PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_data")

            response = MagicMock()
            response.status_code = 404

            async def mock_call_next(request):
                return response

            request = self._build_request(method="GET", url_path="/totally/unknown/path", route_template=None)
            await middleware.dispatch(request, mock_call_next)

            request_count.labels.assert_called_once_with(method="GET", endpoint=UNMATCHED_ENDPOINT_LABEL, status="404")
            unmatched_count.labels.assert_called_once_with(method="GET")
            unmatched_count.labels().inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_cardinality_bounded_under_high_entropy_paths(self):
        """Sending N distinct unmatched URLs must still produce only one endpoint label value."""
        pytest.importorskip("prometheus_client")
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            request_count = MagicMock()
            unmatched_count = MagicMock()
            MockCounter.side_effect = [request_count, unmatched_count]
            MockHistogram.return_value = MagicMock()

            middleware = PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_data")

            response = MagicMock()
            response.status_code = 404

            async def mock_call_next(request):
                return response

            for i in range(50):
                request = self._build_request(method="GET", url_path=f"/attacker/{i}/abc", route_template=None)
                await middleware.dispatch(request, mock_call_next)

            distinct_endpoints = {call.kwargs["endpoint"] for call in request_count.labels.call_args_list}
            assert distinct_endpoints == {UNMATCHED_ENDPOINT_LABEL}, f"endpoint label cardinality leaked: {distinct_endpoints}"
            assert unmatched_count.labels.call_count == 50

    @pytest.mark.asyncio
    async def test_namespace_prefix_applied_to_metric_names(self):
        """Verify that the namespace parameter prefixes all three metric names."""
        pytest.importorskip("prometheus_client")
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            MockCounter.return_value = MagicMock()
            MockHistogram.return_value = MagicMock()

            PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="juniper_data")

            counter_calls = MockCounter.call_args_list
            counter_names = [call.args[0] for call in counter_calls]
            assert "juniper_data_http_requests_total" in counter_names
            assert "juniper_data_http_unmatched_requests_total" in counter_names
            MockHistogram.assert_called_once_with(
                "juniper_data_http_request_duration_seconds",
                "HTTP request duration in seconds",
                ["method", "endpoint"],
            )

    @pytest.mark.asyncio
    async def test_empty_namespace_produces_unprefixed_names(self):
        """Verify that an empty namespace does not add a prefix."""
        pytest.importorskip("prometheus_client")
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram:
            MockCounter.return_value = MagicMock()
            MockHistogram.return_value = MagicMock()

            PrometheusMiddleware(app=MagicMock(), service_name="test", namespace="")

            counter_names = [call.args[0] for call in MockCounter.call_args_list]
            assert "http_requests_total" in counter_names
            assert "http_unmatched_requests_total" in counter_names


@pytest.mark.unit
class TestGetPrometheusApp:
    """Tests for get_prometheus_app function."""

    def test_returns_asgi_app(self):
        pytest.importorskip("prometheus_client")
        app = get_prometheus_app()
        assert callable(app)


@pytest.mark.unit
class TestSetBuildInfo:
    """Tests for set_build_info function."""

    def test_creates_info_metric(self):
        pytest.importorskip("prometheus_client")
        with patch("prometheus_client.Info") as MockInfo:
            mock_info = MagicMock()
            MockInfo.return_value = mock_info
            set_build_info("juniper_data", "0.4.2")
            MockInfo.assert_called_once_with("juniper_data_build", "Build information for juniper-data service")
            mock_info.info.assert_called_once()
            call_args = mock_info.info.call_args[0][0]
            assert call_args["version"] == "0.4.2"
            assert "python_version" in call_args


@pytest.mark.unit
class TestDatasetMetrics:
    """Tests for custom dataset metrics helpers."""

    def test_record_dataset_generation_success(self):
        pytest.importorskip("prometheus_client")
        import juniper_data.api.observability as obs

        obs._dataset_metrics = None  # Reset lazy cache
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram, patch("prometheus_client.Gauge"):
            mock_counter = MagicMock()
            mock_histogram = MagicMock()
            MockCounter.return_value = mock_counter
            MockHistogram.return_value = mock_histogram

            record_dataset_generation("spiral", "success", 1.5)

            mock_counter.labels.assert_called_with(generator="spiral", status="success")
            mock_counter.labels().inc.assert_called_once()
            mock_histogram.labels.assert_called_with(generator="spiral")
            mock_histogram.labels().observe.assert_called_once_with(1.5)

        obs._dataset_metrics = None  # Clean up

    def test_record_dataset_generation_error_skips_histogram(self):
        pytest.importorskip("prometheus_client")
        import juniper_data.api.observability as obs

        obs._dataset_metrics = None
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram") as MockHistogram, patch("prometheus_client.Gauge"):
            mock_counter = MagicMock()
            mock_histogram = MagicMock()
            MockCounter.return_value = mock_counter
            MockHistogram.return_value = mock_histogram

            record_dataset_generation("spiral", "error", 0.0)

            mock_counter.labels.assert_called_with(generator="spiral", status="error")
            mock_histogram.labels.assert_not_called()

        obs._dataset_metrics = None

    def test_set_datasets_cached(self):
        pytest.importorskip("prometheus_client")
        import juniper_data.api.observability as obs

        obs._dataset_metrics = None
        with patch("prometheus_client.Counter"), patch("prometheus_client.Histogram"), patch("prometheus_client.Gauge") as MockGauge:
            mock_gauge = MagicMock()
            MockGauge.return_value = mock_gauge

            set_datasets_cached(42)
            mock_gauge.set.assert_called_once_with(42)

        obs._dataset_metrics = None
