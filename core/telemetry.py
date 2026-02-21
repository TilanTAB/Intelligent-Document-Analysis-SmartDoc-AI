"""
OpenTelemetry bootstrap and runtime helpers for SmartDoc.

This module is intentionally defensive:
- Telemetry is opt-in (disabled by default).
- Missing exporter dependencies do not crash the app.
- Metrics avoid high-cardinality labels by design.
"""

from __future__ import annotations

import atexit
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from configuration.parameters import parameters

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_initialized = False

_tracer = None
_span_provider = None
_meter_provider = None
_logger_provider = None
_otel_log_handler = None

_request_counter = None
_request_error_counter = None
_request_duration_hist = None
_stage_duration_hist = None
_stage_alert_counter = None


def _parse_kv(raw: Optional[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not raw:
        return parsed
    for token in str(raw).split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def _build_signal_endpoint(base_endpoint: str, signal: str) -> str:
    base = (base_endpoint or "").strip().rstrip("/")
    suffix = f"/v1/{signal}"
    if base.lower().endswith(suffix):
        return base
    return f"{base}{suffix}"


def _resolve_signal_endpoint(signal: str) -> str:
    specific_map = {
        "traces": (parameters.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or "").strip(),
        "metrics": (parameters.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or "").strip(),
        "logs": (parameters.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT or "").strip(),
    }
    if specific_map.get(signal):
        return specific_map[signal]

    base_endpoint = (parameters.OTEL_EXPORTER_OTLP_ENDPOINT or "").strip()
    if not base_endpoint:
        base_endpoint = "http://127.0.0.1:4318"
    return _build_signal_endpoint(base_endpoint, signal)


def _to_attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def initialize_telemetry() -> None:
    """Initialize OpenTelemetry providers/exporters once."""
    global _initialized
    global _tracer, _span_provider, _meter_provider, _logger_provider, _otel_log_handler
    global _request_counter, _request_error_counter, _request_duration_hist
    global _stage_duration_hist, _stage_alert_counter

    with _lock:
        if _initialized:
            return
        _initialized = True

        if not parameters.OTEL_ENABLED:
            logger.info("OpenTelemetry disabled (OTEL_ENABLED=false).")
            return

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.resources import Resource
        except Exception as exc:
            logger.warning("Telemetry not initialized: missing OpenTelemetry core packages (%s)", exc)
            return

        resource_attrs: Dict[str, str] = {
            "service.name": parameters.OTEL_SERVICE_NAME,
            "service.version": parameters.OTEL_SERVICE_VERSION,
        }
        if parameters.OTEL_SERVICE_NAMESPACE:
            resource_attrs["service.namespace"] = parameters.OTEL_SERVICE_NAMESPACE
        resource_attrs.update(_parse_kv(parameters.OTEL_RESOURCE_ATTRIBUTES))
        resource = Resource.create(resource_attrs)

        headers = _parse_kv(parameters.OTEL_EXPORTER_OTLP_HEADERS)
        timeout_s = max(1.0, float(parameters.OTEL_EXPORT_TIMEOUT_S))
        any_signal_enabled = False

        if parameters.OTEL_TRACES_ENABLED:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

                span_provider = TracerProvider(resource=resource)
                trace_endpoint = _resolve_signal_endpoint("traces")
                span_provider.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(
                            endpoint=trace_endpoint,
                            headers=headers or None,
                            timeout=timeout_s,
                        )
                    )
                )
                if parameters.OTEL_CONSOLE_DEBUG_EXPORT:
                    span_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
                trace.set_tracer_provider(span_provider)
                _span_provider = span_provider
                _tracer = trace.get_tracer("smartdoc.telemetry")
                any_signal_enabled = True
                logger.info("OpenTelemetry traces exporter configured: %s", trace_endpoint)
            except Exception as exc:
                logger.warning("OpenTelemetry traces setup failed: %s", exc)

        if parameters.OTEL_METRICS_ENABLED:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                from opentelemetry.sdk.metrics import MeterProvider
                from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

                metric_readers = []
                metrics_endpoint = _resolve_signal_endpoint("metrics")
                metric_readers.append(
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(
                            endpoint=metrics_endpoint,
                            headers=headers or None,
                            timeout=timeout_s,
                        ),
                        export_interval_millis=max(1000, int(parameters.OTEL_METRICS_EXPORT_INTERVAL_MS)),
                    )
                )
                if parameters.OTEL_CONSOLE_DEBUG_EXPORT:
                    metric_readers.append(
                        PeriodicExportingMetricReader(
                            ConsoleMetricExporter(),
                            export_interval_millis=max(1000, int(parameters.OTEL_METRICS_EXPORT_INTERVAL_MS)),
                        )
                    )

                meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
                metrics.set_meter_provider(meter_provider)
                meter = metrics.get_meter("smartdoc.telemetry", "1.0.0")
                _meter_provider = meter_provider

                _request_counter = meter.create_counter(
                    name="smartdoc.requests.total",
                    description="Total SmartDoc request attempts.",
                )
                _request_error_counter = meter.create_counter(
                    name="smartdoc.requests.errors.total",
                    description="Total SmartDoc request failures.",
                )
                _request_duration_hist = meter.create_histogram(
                    name="smartdoc.requests.duration.seconds",
                    description="End-to-end request duration in seconds.",
                    unit="s",
                )
                _stage_duration_hist = meter.create_histogram(
                    name="smartdoc.stage.duration.seconds",
                    description="Stage duration in seconds.",
                    unit="s",
                )
                _stage_alert_counter = meter.create_counter(
                    name="smartdoc.stage.alerts.total",
                    description="Number of stage p95 threshold alert events.",
                )

                any_signal_enabled = True
                logger.info("OpenTelemetry metrics exporter configured: %s", metrics_endpoint)
            except Exception as exc:
                logger.warning("OpenTelemetry metrics setup failed: %s", exc)

        if parameters.OTEL_LOGS_ENABLED:
            try:
                from opentelemetry._logs import set_logger_provider
                from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
                from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
                from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

                try:
                    from opentelemetry.sdk._logs.export import ConsoleLogRecordExporter
                except Exception:
                    from opentelemetry.sdk._logs.export import ConsoleLogExporter as ConsoleLogRecordExporter

                logger_provider = LoggerProvider(resource=resource)
                logs_endpoint = _resolve_signal_endpoint("logs")
                logger_provider.add_log_record_processor(
                    BatchLogRecordProcessor(
                        OTLPLogExporter(
                            endpoint=logs_endpoint,
                            headers=headers or None,
                            timeout=timeout_s,
                        )
                    )
                )
                if parameters.OTEL_CONSOLE_DEBUG_EXPORT:
                    logger_provider.add_log_record_processor(
                        BatchLogRecordProcessor(ConsoleLogRecordExporter())
                    )

                set_logger_provider(logger_provider)

                # Keep existing handlers and add OTEL as an additional sink.
                otel_handler = LoggingHandler(
                    level=logging.INFO,
                    logger_provider=logger_provider,
                )
                logging.getLogger().addHandler(otel_handler)

                _logger_provider = logger_provider
                _otel_log_handler = otel_handler
                any_signal_enabled = True
                logger.info("OpenTelemetry logs exporter configured: %s", logs_endpoint)
            except Exception as exc:
                logger.warning("OpenTelemetry logs setup failed: %s", exc)

        if not any_signal_enabled:
            logger.warning("OpenTelemetry was enabled but no signal initialized successfully.")
            return

        logger.info("OpenTelemetry initialization complete.")


def shutdown_telemetry() -> None:
    """Best-effort telemetry shutdown and flush."""
    global _logger_provider, _meter_provider, _span_provider, _otel_log_handler
    with _lock:
        if _otel_log_handler is not None:
            try:
                logging.getLogger().removeHandler(_otel_log_handler)
            except Exception:
                pass
            _otel_log_handler = None

        if _logger_provider is not None:
            try:
                _logger_provider.shutdown()
            except Exception:
                pass
            _logger_provider = None

        if _meter_provider is not None:
            try:
                _meter_provider.shutdown()
            except Exception:
                pass
            _meter_provider = None

        if _span_provider is not None:
            try:
                _span_provider.shutdown()
            except Exception:
                pass
            _span_provider = None


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Start a trace span if tracing is enabled; otherwise yield None."""
    if _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(str(key), _to_attribute_value(value))
        yield span


def mark_span_error(span: Any, exc: BaseException) -> None:
    """Annotate span with exception details when available."""
    if span is None:
        return
    try:
        span.record_exception(exc)
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(exc).__name__)
        span.set_attribute("error.message", str(exc))
    except Exception:
        pass


def record_request_metrics(duration_s: float, success: bool, extra_attributes: Optional[Dict[str, Any]] = None) -> None:
    """Record end-to-end request counters and duration metrics."""
    if duration_s is None or duration_s < 0:
        return

    attrs: Dict[str, Any] = {"success": bool(success)}
    if extra_attributes:
        for key, value in extra_attributes.items():
            attrs[str(key)] = _to_attribute_value(value)

    if _request_counter is not None:
        _request_counter.add(1, attributes=attrs)
    if _request_duration_hist is not None:
        _request_duration_hist.record(duration_s, attributes=attrs)
    if not success and _request_error_counter is not None:
        _request_error_counter.add(1, attributes=attrs)


def record_stage_latency(stage: str, duration_s: float, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Record stage latency histogram with low-cardinality labels only."""
    if _stage_duration_hist is None or duration_s is None or duration_s < 0:
        return

    attrs: Dict[str, Any] = {"stage": str(stage)}
    mode = (metadata or {}).get("mode")
    if isinstance(mode, str) and mode:
        attrs["mode"] = mode.lower()

    _stage_duration_hist.record(duration_s, attributes=attrs)


def record_stage_alert(stage: str, threshold_s: float) -> None:
    """Record alert counter when stage p95 breaches threshold."""
    if _stage_alert_counter is None:
        return
    attrs = {
        "stage": str(stage),
        "threshold_s": float(threshold_s),
    }
    _stage_alert_counter.add(1, attributes=attrs)


atexit.register(shutdown_telemetry)
