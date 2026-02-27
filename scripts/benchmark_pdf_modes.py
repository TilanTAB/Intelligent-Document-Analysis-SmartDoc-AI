#!/usr/bin/env python3
"""Benchmark PDF ingestion modes: pdf_only vs both vs vision_only.

Outputs:
- JSON report under docs/perf/benchmarks/
- Markdown table printed to stdout for quick comparison
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALID_MODES = ("pdf_only", "both", "vision_only")
VISION_COST_PER_CALL_USD = 0.0125  # aligns with existing chart extraction estimate logs


@dataclass
class ModeRunResult:
    mode: str
    run_idx: int
    files_count: int
    chunks_total: int
    ingest_total_s: float
    retriever_build_total_s: float
    first_query_latency_s: float
    first_query_docs: int
    retrieval_hit_rate: float
    answer_verification_pass_rate: float
    vision_docs: int
    estimated_vision_calls: float
    estimated_vision_cost_usd: float


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _evaluate_retrieval_quality(retriever: Any, quality_queries: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not quality_queries:
        return 0.0, 0.0

    retrieval_hits = 0
    answer_pass = 0
    evaluated = 0
    for item in quality_queries:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        evaluated += 1
        expected_sources = [str(src).lower() for src in item.get("expected_sources", [])]
        docs = retriever.invoke(question)
        retrieved_sources = [str(doc.metadata.get("source", "")).lower() for doc in docs]
        if expected_sources:
            has_hit = any(expected in source for expected in expected_sources for source in retrieved_sources)
        else:
            has_hit = bool(docs)
        retrieval_hits += int(has_hit)
        answer_pass += int(bool(docs))

    total = max(1, evaluated)
    return round(retrieval_hits / total, 6), round(answer_pass / total, 6)


def _estimate_vision_cost(chunks: List[Any]) -> Tuple[int, float, float]:
    vision_docs = 0
    estimated_calls = 0.0
    seen_items = set()
    for doc in chunks:
        metadata = getattr(doc, "metadata", {}) or {}
        extraction_method = str(metadata.get("extraction_method", "")).strip().lower()
        if extraction_method not in {"hybrid_batch", "hybrid_sequential", "vision_only_full_page"}:
            continue
        origin_chunk_id = str(metadata.get("origin_chunk_id", "")).strip()
        item_key = origin_chunk_id or (
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            extraction_method,
        )
        if item_key in seen_items:
            continue
        seen_items.add(item_key)
        vision_docs += 1
        batch_size_raw = metadata.get("batch_size", 1)
        try:
            batch_size = max(1.0, float(batch_size_raw))
        except Exception:
            batch_size = 1.0
        estimated_calls += 1.0 / batch_size
    estimated_cost = estimated_calls * VISION_COST_PER_CALL_USD
    return vision_docs, round(estimated_calls, 6), round(estimated_cost, 6)


def _build_default_quality_queries(files: List[Path], query: str) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for file_path in files:
        checks.append(
            {
                "question": query,
                "expected_sources": [file_path.name.lower()],
            }
        )
    return checks


def run_once(
    files: List[Path],
    mode: str,
    query: str,
    run_idx: int,
    quality_queries: List[Dict[str, Any]],
) -> ModeRunResult:
    from content_analyzer.document_parser import DocumentProcessor
    from search_engine.indexer import RetrieverBuilder

    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()

    all_chunks: List[Any] = []
    seen_hashes = set()

    t_ingest_start = time.perf_counter()
    for file_path in files:
        file_obj = SimpleNamespace(name=str(file_path))
        chunks = processor._process_file(file_obj, pdf_analysis_mode=mode)
        for chunk in chunks:
            chunk_hash = processor._generate_hash(str(chunk.page_content).encode("utf-8"))
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            all_chunks.append(chunk)
    t_ingest_end = time.perf_counter()

    session_id = f"mode_bench_{mode}_{int(time.time())}_{run_idx}"
    t_build_start = time.perf_counter()
    retriever = retriever_builder.build_hybrid_retriever(all_chunks, session_id=session_id)
    t_build_end = time.perf_counter()
    profile = retriever_builder.last_build_profile or {}
    retriever_build_total_s = profile.get("retriever_build_total_s")
    if not isinstance(retriever_build_total_s, (int, float)):
        retriever_build_total_s = t_build_end - t_build_start

    t_query_start = time.perf_counter()
    docs = retriever.invoke(query)
    t_query_end = time.perf_counter()

    retrieval_hit_rate, answer_pass_rate = _evaluate_retrieval_quality(
        retriever=retriever,
        quality_queries=quality_queries,
    )
    vision_docs, estimated_calls, estimated_cost = _estimate_vision_cost(all_chunks)

    return ModeRunResult(
        mode=mode,
        run_idx=run_idx,
        files_count=len(files),
        chunks_total=len(all_chunks),
        ingest_total_s=round(t_ingest_end - t_ingest_start, 6),
        retriever_build_total_s=round(float(retriever_build_total_s), 6),
        first_query_latency_s=round(t_query_end - t_query_start, 6),
        first_query_docs=len(docs),
        retrieval_hit_rate=retrieval_hit_rate,
        answer_verification_pass_rate=answer_pass_rate,
        vision_docs=vision_docs,
        estimated_vision_calls=estimated_calls,
        estimated_vision_cost_usd=estimated_cost,
    )


def _aggregate(results: List[ModeRunResult], mode: str) -> Dict[str, Any]:
    rows = [r for r in results if r.mode == mode]
    if not rows:
        return {}
    return {
        "mode": mode,
        "runs": len(rows),
        "chunks_total_mean": round(statistics.mean(r.chunks_total for r in rows), 3),
        "ingest_total_mean_s": round(statistics.mean(r.ingest_total_s for r in rows), 6),
        "ingest_total_p95_s": round(_percentile([r.ingest_total_s for r in rows], 95), 6),
        "retriever_build_mean_s": round(statistics.mean(r.retriever_build_total_s for r in rows), 6),
        "first_query_mean_s": round(statistics.mean(r.first_query_latency_s for r in rows), 6),
        "first_query_docs_mean": round(statistics.mean(r.first_query_docs for r in rows), 3),
        "retrieval_hit_rate": round(statistics.mean(r.retrieval_hit_rate for r in rows), 6),
        "answer_verification_pass_rate": round(statistics.mean(r.answer_verification_pass_rate for r in rows), 6),
        "vision_docs_mean": round(statistics.mean(r.vision_docs for r in rows), 3),
        "estimated_vision_calls_mean": round(statistics.mean(r.estimated_vision_calls for r in rows), 6),
        "estimated_vision_cost_mean_usd": round(statistics.mean(r.estimated_vision_cost_usd for r in rows), 6),
    }


def _print_markdown_table(aggregates: List[Dict[str, Any]]) -> None:
    print("\n### PDF Analysis Mode Benchmark (Latency / Cost / Quality)\n")
    print(
        "| Mode | Ingest mean (s) | Ingest p95 (s) | Retriever build mean (s) | First query mean (s) | Est. vision cost/run (USD) | Retrieval hit rate | Quality pass rate |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in aggregates:
        print(
            f"| {row['mode']} | {row['ingest_total_mean_s']:.3f} | {row['ingest_total_p95_s']:.3f} | "
            f"{row['retriever_build_mean_s']:.3f} | {row['first_query_mean_s']:.3f} | "
            f"${row['estimated_vision_cost_mean_usd']:.4f} | {row['retrieval_hit_rate']:.3f} | "
            f"{row['answer_verification_pass_rate']:.3f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SmartDoc PDF analysis modes.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["samples/OIT-NASK-IAGen_WP140_web.pdf"],
        help="Files to benchmark.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(VALID_MODES),
        help="Subset of modes to run: pdf_only both vision_only",
    )
    parser.add_argument(
        "--query",
        default="Which occupations are most likely to be automated by AI?",
        help="Query used to benchmark retrieval latency.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per mode (>=2 recommended).")
    parser.add_argument(
        "--quality-queries",
        default="",
        help="Optional JSON file: [{\"question\":...,\"expected_sources\":[...]}]",
    )
    parser.add_argument("--label", default="pdf_mode_compare", help="Output label.")
    parser.add_argument("--output-dir", default="docs/perf/benchmarks", help="Output directory.")
    args = parser.parse_args()

    files = [Path(p) for p in args.files]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"Missing input files: {missing}")

    requested_modes = [str(m).strip().lower() for m in args.modes]
    invalid_modes = [m for m in requested_modes if m not in VALID_MODES]
    if invalid_modes:
        raise SystemExit(f"Unsupported mode(s): {invalid_modes}; valid={VALID_MODES}")

    quality_queries: List[Dict[str, Any]] = []
    if args.quality_queries:
        quality_queries = json.loads(Path(args.quality_queries).read_text(encoding="utf-8"))
    if not quality_queries:
        quality_queries = _build_default_quality_queries(files=files, query=args.query)

    all_results: List[ModeRunResult] = []
    started_at = datetime.now(UTC).isoformat()

    for mode in requested_modes:
        for run_idx in range(1, max(1, int(args.runs)) + 1):
            result = run_once(
                files=files,
                mode=mode,
                query=args.query,
                run_idx=run_idx,
                quality_queries=quality_queries,
            )
            all_results.append(result)

    aggregates = [_aggregate(all_results, mode) for mode in requested_modes]
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "started_at": started_at,
        "label": args.label,
        "files": [str(p) for p in files],
        "modes": requested_modes,
        "query": args.query,
        "runs_per_mode": max(1, int(args.runs)),
        "quality_queries": quality_queries,
        "run_results": [asdict(r) for r in all_results],
        "summary": aggregates,
        "notes": {
            "estimated_vision_cost_basis": f"${VISION_COST_PER_CALL_USD} per vision call (project heuristic)",
            "quality_definition": "retrieval hit/pass rates from retrieval checks (not full LLM answer grading)",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{stamp}_{args.label}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Benchmark report written to: {output_path}")
    _print_markdown_table(aggregates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
