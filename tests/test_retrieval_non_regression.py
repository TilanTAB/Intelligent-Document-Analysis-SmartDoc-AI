import importlib.util
import json
import sys
from pathlib import Path


def _load_compare_module():
    module_path = Path("scripts/compare_benchmarks.py").resolve()
    spec = importlib.util.spec_from_file_location("compare_benchmarks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path, ingest_p95: float, retrieval_hit: float, answer_pass: float):
    payload = {
        "summary": {
            "ingest_build_p95_s": ingest_p95,
            "ingest_build_mean_s": ingest_p95,
            "parse_p95_s": 1.0,
            "retriever_build_p95_s": 1.0,
            "query_p95_s": 1.0,
            "retrieval_hit_rate": retrieval_hit,
            "answer_verification_pass_rate": answer_pass,
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_reports_passes_with_latency_gain_and_quality_parity(monkeypatch, tmp_path):
    module = _load_compare_module()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"

    _write_report(baseline, ingest_p95=10.0, retrieval_hit=0.8, answer_pass=0.9)
    _write_report(candidate, ingest_p95=6.0, retrieval_hit=0.8, answer_pass=0.9)

    argv = [
        "compare_benchmarks.py",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--target-improvement",
        "35",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 0


def test_compare_reports_fails_on_quality_regression(monkeypatch, tmp_path):
    module = _load_compare_module()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"

    _write_report(baseline, ingest_p95=10.0, retrieval_hit=0.9, answer_pass=0.9)
    _write_report(candidate, ingest_p95=6.0, retrieval_hit=0.8, answer_pass=0.7)

    argv = [
        "compare_benchmarks.py",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--target-improvement",
        "35",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 1
