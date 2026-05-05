"""Unit tests for ``juniper_data.api.observability``.

METRICS-MON R2.1.2 / seed-06: the cross-cutting machinery
(JuniperJsonFormatter, RequestIdMiddleware, PrometheusMiddleware,
configure_logging, configure_sentry, get_prometheus_app,
set_build_info) now lives in ``juniper-observability`` and is covered
by that package's own test suite. This module retains:

  * ``TestDatasetMetrics`` — juniper-data-specific dataset-generation
    metrics (record_dataset_generation, set_datasets_cached, lazy
    init).
  * ``TestObservabilityShim`` — sanity-check that the re-export shim
    in ``juniper_data.api.observability`` exposes every symbol that
    existing call sites depended on, sourced from the shared lib.
"""

from unittest.mock import MagicMock, patch

import juniper_data.api.observability as obs
import juniper_observability as jobs
import pytest

from juniper_data.api.observability import (
    DEFAULT_LOG_FORMAT_PLAIN,
    DEFAULT_SENTRY_TRACES_SAMPLE_RATE,
    LOG_FORMAT_JSON,
    UNMATCHED_ENDPOINT_LABEL,
    JuniperJsonFormatter,
    MetricsAuthMiddleware,
    PrometheusMiddleware,
    RequestIdMiddleware,
    _strip_sensitive_headers,
    configure_logging,
    configure_sentry,
    get_prometheus_app,
    record_dataset_generation,
    record_dataset_post,
    request_id_var,
    set_build_info,
    set_datasets_cached,
)


@pytest.mark.unit
class TestObservabilityShim:
    """METRICS-MON R2.1.2: re-export shim points at juniper_observability."""

    def test_cross_cutting_symbols_are_juniper_observability_objects(self):
        """The symbols re-exported here must be the *same* objects exposed by juniper_observability."""
        assert JuniperJsonFormatter is jobs.JuniperJsonFormatter
        assert RequestIdMiddleware is jobs.RequestIdMiddleware
        assert PrometheusMiddleware is jobs.PrometheusMiddleware
        assert configure_logging is jobs.configure_logging
        assert configure_sentry is jobs.configure_sentry
        assert get_prometheus_app is jobs.get_prometheus_app
        assert set_build_info is jobs.set_build_info
        assert request_id_var is jobs.request_id_var
        assert UNMATCHED_ENDPOINT_LABEL == jobs.UNMATCHED_ENDPOINT_LABEL
        assert LOG_FORMAT_JSON == jobs.LOG_FORMAT_JSON
        assert DEFAULT_LOG_FORMAT_PLAIN == jobs.DEFAULT_LOG_FORMAT_PLAIN
        assert DEFAULT_SENTRY_TRACES_SAMPLE_RATE == jobs.DEFAULT_SENTRY_TRACES_SAMPLE_RATE

    def test_strip_sensitive_headers_re_exported(self):
        """The private SEC-10 hook is re-exported so existing tests still find it."""
        from juniper_observability.sentry import _strip_sensitive_headers as upstream

        assert _strip_sensitive_headers is upstream

    def test_metrics_auth_middleware_stays_juniper_data_specific(self):
        """``MetricsAuthMiddleware`` (SEC-16 IP allowlist) is intentionally NOT in the shared lib.

        Promotion to juniper-observability is tracked as a roadmap §R5
        gating issue. Until then, juniper-data is the sole owner.
        """
        assert MetricsAuthMiddleware.__module__ == "juniper_data.api.observability"


@pytest.mark.unit
class TestDatasetMetrics:
    """Tests for juniper-data-specific dataset-generation metrics."""

    def test_record_dataset_generation_success(self):
        pytest.importorskip("prometheus_client")

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

    def test_record_dataset_post_labels_by_generator_status_and_cache(self):
        pytest.importorskip("prometheus_client")

        obs._dataset_metrics = None
        with patch("prometheus_client.Counter") as MockCounter, patch("prometheus_client.Histogram"), patch("prometheus_client.Gauge"):
            mock_generation_counter = MagicMock()
            mock_post_counter = MagicMock()
            MockCounter.side_effect = [mock_generation_counter, mock_post_counter]

            record_dataset_post("spiral", "error", "miss")

            mock_post_counter.labels.assert_called_once_with(generator="spiral", status="error", cache="miss")
            mock_post_counter.labels().inc.assert_called_once()
            mock_generation_counter.labels.assert_not_called()

        obs._dataset_metrics = None

    def test_set_datasets_cached(self):
        pytest.importorskip("prometheus_client")

        obs._dataset_metrics = None
        with patch("prometheus_client.Counter"), patch("prometheus_client.Histogram"), patch("prometheus_client.Gauge") as MockGauge:
            mock_gauge = MagicMock()
            MockGauge.return_value = mock_gauge

            set_datasets_cached(42)
            mock_gauge.set.assert_called_once_with(42)

        obs._dataset_metrics = None
