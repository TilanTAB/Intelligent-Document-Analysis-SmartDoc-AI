"""
Lightweight stage-level latency monitor with p50/p95 tracking and alerting.

Designed for:
- parse.total
- chart.phase1
- chart.phase2
- embed.batch
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional

from configuration.parameters import parameters
from core.telemetry import record_stage_alert, record_stage_latency

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class StageLatencyMonitor:
    """Thread-safe in-process monitor for stage latencies."""

    def __init__(self) -> None:
        window = max(10, _safe_int(parameters.PERF_METRICS_WINDOW, 200))
        self._latencies: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._last_alert_at: Dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = (len(sorted_values) - 1) * (percentile / 100.0)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return sorted_values[lower]
        weight = rank - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    @staticmethod
    def _alert_threshold(stage: str) -> Optional[float]:
        mapping = {
            "parse.total": _safe_float(parameters.PERF_ALERT_PARSE_P95_S, 90.0),
            "chart.phase1": _safe_float(parameters.PERF_ALERT_CHART_PHASE1_P95_S, 120.0),
            "chart.phase2": _safe_float(parameters.PERF_ALERT_CHART_PHASE2_P95_S, 45.0),
            "embed.batch": _safe_float(parameters.PERF_ALERT_EMBED_BATCH_P95_S, 10.0),
        }
        return mapping.get(stage)

    def record(self, stage: str, duration_s: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record latency for a stage and emit metrics/alerts."""
        if duration_s is None:
            return
        duration = _safe_float(duration_s, 0.0)
        if duration < 0:
            return

        min_samples = max(1, _safe_int(parameters.PERF_ALERT_MIN_SAMPLES, 5))
        cooldown_s = max(5.0, _safe_float(parameters.PERF_ALERT_COOLDOWN_S, 300.0))
        threshold = self._alert_threshold(stage)

        with self._lock:
            self._latencies[stage].append(duration)
            samples = list(self._latencies[stage])
            samples.sort()
            count = len(samples)
            p50 = self._percentile(samples, 50)
            p95 = self._percentile(samples, 95)

            logger.info(
                "[METRICS] stage=%s samples=%d p50_s=%.3f p95_s=%.3f latest_s=%.3f",
                stage,
                count,
                p50,
                p95,
                duration,
            )
            record_stage_latency(stage=stage, duration_s=duration, metadata=metadata)

            if threshold is None or count < min_samples:
                return

            last_alert = self._last_alert_at.get(stage, 0.0)
            now = time.time()
            if p95 > threshold and (now - last_alert) >= cooldown_s:
                logger.warning(
                    "[ALERT] stage=%s p95_s=%.3f threshold_s=%.3f samples=%d latest_s=%.3f metadata=%s",
                    stage,
                    p95,
                    threshold,
                    count,
                    duration,
                    metadata or {},
                )
                record_stage_alert(stage=stage, threshold_s=threshold)
                self._last_alert_at[stage] = now


latency_monitor = StageLatencyMonitor()
