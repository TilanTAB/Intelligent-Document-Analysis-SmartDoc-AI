#!/usr/bin/env python3
"""Compare SmartDoc benchmark reports and enforce latency gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def load_report(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric(report: Dict, key: str) -> float:
    summary = report.get("summary", {})
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def improvement_pct(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return ((baseline - candidate) / baseline) * 100.0


def compare_metrics(baseline: Dict, candidate: Dict) -> Dict[str, Tuple[float, float, float]]:
    keys = [
        "ingest_build_p95_s",
        "ingest_build_mean_s",
        "parse_p95_s",
        "retriever_build_p95_s",
        "query_p95_s",
    ]
    compared = {}
    for key in keys:
        b = get_metric(baseline, key)
        c = get_metric(candidate, key)
        compared[key] = (b, c, improvement_pct(b, c))
    return compared


def quality_non_regression(baseline: Dict, candidate: Dict) -> Tuple[bool, Dict[str, Tuple[float, float]]]:
    summary_baseline = baseline.get("summary", {})
    summary_candidate = candidate.get("summary", {})
    checks = {}
    required_keys = ["retrieval_hit_rate", "answer_verification_pass_rate"]
    available = all(isinstance(summary_baseline.get(k), (int, float)) and isinstance(summary_candidate.get(k), (int, float)) for k in required_keys)
    if not available:
        return True, checks
    passed = True
    for key in required_keys:
        b = float(summary_baseline[key])
        c = float(summary_candidate[key])
        checks[key] = (b, c)
        if c + 1e-9 < b:
            passed = False
    return passed, checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate benchmark JSON reports.")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark report path.")
    parser.add_argument("--candidate", required=True, help="Candidate benchmark report path.")
    parser.add_argument("--target-improvement", type=float, default=35.0, help="Required improvement percent for ingest_build_p95_s.")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline report not found: {baseline_path}")
    if not candidate_path.exists():
        raise SystemExit(f"Candidate report not found: {candidate_path}")

    baseline = load_report(baseline_path)
    candidate = load_report(candidate_path)
    compared = compare_metrics(baseline, candidate)
    quality_ok, quality_checks = quality_non_regression(baseline, candidate)

    print("Benchmark comparison:")
    for key, (b, c, pct) in compared.items():
        print(f"- {key}: baseline={b:.6f}s candidate={c:.6f}s improvement={pct:.2f}%")

    gate_key = "ingest_build_p95_s"
    gate_improvement = compared[gate_key][2]
    latency_passed = gate_improvement >= args.target_improvement
    quality_passed = quality_ok
    passed = latency_passed and quality_passed
    print(
        f"Latency gate ({gate_key} >= {args.target_improvement:.2f}% improvement): "
        f"{'PASS' if latency_passed else 'FAIL'}"
    )
    if quality_checks:
        for key, (b, c) in quality_checks.items():
            print(f"- quality.{key}: baseline={b:.6f} candidate={c:.6f}")
    print(f"Quality non-regression gate: {'PASS' if quality_passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
