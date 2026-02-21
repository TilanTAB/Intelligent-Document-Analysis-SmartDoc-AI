import importlib.util
import json
import sys
from pathlib import Path


def _load_benchmark_module():
    module_path = Path("scripts/benchmark_latency.py").resolve()
    spec = importlib.util.spec_from_file_location("benchmark_latency", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_benchmark_script_emits_stage_timings(monkeypatch, tmp_path):
    module = _load_benchmark_module()

    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello world", encoding="utf-8")

    monkeypatch.setattr(module, "_iter_input_files", lambda roots: [sample_file])

    fake_result = module.RunMetrics(
        run_idx=1,
        files_count=1,
        docs_total=1,
        chunks_total=1,
        parse_total_s=0.4,
        chunk_total_s=0.2,
        ingest_file_total_s=0.6,
        retriever_build_total_s=0.8,
        first_query_latency_s=0.1,
        first_query_docs=3,
        ingest_build_total_s=1.4,
        retrieval_hit_rate=1.0,
        answer_verification_pass_rate=1.0,
    )
    monkeypatch.setattr(module, "run_once", lambda files, query, run_idx, quality_queries=None: (fake_result, {}, []))

    output_dir = tmp_path / "bench_out"
    argv = [
        "benchmark_latency.py",
        "--input-dir",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--runs",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = module.main()
    assert rc == 0

    reports = list(output_dir.glob("*.json"))
    assert reports, "Expected benchmark report output"
    payload = json.loads(reports[0].read_text(encoding="utf-8"))

    summary = payload["summary"]
    assert "ingest_build_p95_s" in summary
    assert "parse_p95_s" in summary
    assert "retriever_build_p95_s" in summary
    assert "query_p95_s" in summary
