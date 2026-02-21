#!/usr/bin/env python3
"""Benchmark SmartDoc ingestion/build latency and emit JSON reports."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class FileMetrics:
    path: str
    parse_s: float
    chunk_s: float
    docs_extracted: int
    chunks_created: int

    @property
    def ingest_s(self) -> float:
        return self.parse_s + self.chunk_s


@dataclass
class RunMetrics:
    run_idx: int
    files_count: int
    docs_total: int
    chunks_total: int
    parse_total_s: float
    chunk_total_s: float
    ingest_file_total_s: float
    retriever_build_total_s: float
    first_query_latency_s: float
    first_query_docs: int
    ingest_build_total_s: float
    retrieval_hit_rate: float
    answer_verification_pass_rate: float


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _iter_input_files(dirs: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in dirs:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in ALLOWED_EXTS:
                files.append(candidate)
    return sorted(files)


def _load_non_pdf(file_path: Path) -> List[Any]:
    from langchain_community.document_loaders import Docx2txtLoader, TextLoader

    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        return Docx2txtLoader(str(file_path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(file_path)).load()
    raise ValueError(f"Unsupported non-pdf file extension: {suffix}")


def _chunk_documents(processor: Any, docs: List[Any], file_hash: str, stable_source: str) -> List[Any]:
    from langchain_core.documents import Document

    chunks: List[Any] = []
    for i, doc in enumerate(docs):
        page_chunks = processor.splitter.split_text(doc.page_content)
        for j, chunk in enumerate(page_chunks):
            chunk_id = f"txt_{file_hash}_{doc.metadata.get('page', i + 1)}_{j}"
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": stable_source,
                        "page": doc.metadata.get("page", i + 1),
                        "type": doc.metadata.get("type", "text"),
                        "chunk_id": chunk_id,
                    },
                )
            )
    return chunks


def _parse_and_chunk(processor: Any, file_path: Path) -> tuple[List[Any], FileMetrics]:
    file_bytes = file_path.read_bytes()
    file_hash = processor._generate_hash(file_bytes)
    stable_source = f"{file_path.name}::{file_hash}"

    t_parse_start = time.perf_counter()
    if file_path.suffix.lower() == ".pdf":
        docs = processor._load_pdf_by_mode(str(file_path))
    else:
        docs = _load_non_pdf(file_path)
    t_parse_end = time.perf_counter()

    t_chunk_start = time.perf_counter()
    chunks = _chunk_documents(processor, docs, file_hash=file_hash, stable_source=stable_source)
    t_chunk_end = time.perf_counter()

    metrics = FileMetrics(
        path=str(file_path),
        parse_s=round(t_parse_end - t_parse_start, 6),
        chunk_s=round(t_chunk_end - t_chunk_start, 6),
        docs_extracted=len(docs),
        chunks_created=len(chunks),
    )
    return chunks, metrics


def _evaluate_retrieval_quality(retriever, quality_queries: List[Dict]) -> tuple[float, float]:
    if not quality_queries:
        return 0.0, 0.0
    retrieval_hits = 0
    answer_pass = 0
    for item in quality_queries:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        expected_sources = [str(src).lower() for src in item.get("expected_sources", [])]
        docs = retriever.invoke(question)
        retrieved_sources = [str(doc.metadata.get("source", "")).lower() for doc in docs]
        if expected_sources:
            has_hit = any(expected in source for expected in expected_sources for source in retrieved_sources)
        else:
            has_hit = bool(docs)
        retrieval_hits += int(has_hit)
        answer_pass += int(bool(docs))
    total = max(1, len(quality_queries))
    return round(retrieval_hits / total, 6), round(answer_pass / total, 6)


def run_once(
    files: List[Path],
    query: str,
    run_idx: int,
    quality_queries: List[Dict] | None = None,
) -> tuple[RunMetrics, Dict, List[Dict]]:
    from content_analyzer.document_parser import DocumentProcessor
    from search_engine.indexer import RetrieverBuilder

    processor = DocumentProcessor()
    processor.chart_extraction_enabled = False
    retriever_builder = RetrieverBuilder()

    all_chunks: List[Any] = []
    seen_content_hashes = set()
    file_metrics: List[FileMetrics] = []

    for file_path in files:
        chunks, metrics = _parse_and_chunk(processor, file_path)
        file_metrics.append(metrics)
        for chunk in chunks:
            content_hash = processor._generate_hash(chunk.page_content.encode("utf-8"))
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            all_chunks.append(chunk)

    parse_total = sum(m.parse_s for m in file_metrics)
    chunk_total = sum(m.chunk_s for m in file_metrics)

    session_id = f"bench_{int(time.time())}_{run_idx}"
    t_build_start = time.perf_counter()
    retriever = retriever_builder.build_hybrid_retriever(all_chunks, session_id=session_id)
    t_build_end = time.perf_counter()

    t_query_start = time.perf_counter()
    first_docs = retriever.invoke(query)
    t_query_end = time.perf_counter()
    retrieval_hit_rate, answer_pass_rate = _evaluate_retrieval_quality(
        retriever, quality_queries or []
    )

    build_profile = retriever_builder.last_build_profile
    retriever_build_total = build_profile.get("retriever_build_total_s")
    if not isinstance(retriever_build_total, (int, float)):
        retriever_build_total = round(t_build_end - t_build_start, 6)

    run_metrics = RunMetrics(
        run_idx=run_idx,
        files_count=len(files),
        docs_total=sum(m.docs_extracted for m in file_metrics),
        chunks_total=len(all_chunks),
        parse_total_s=round(parse_total, 6),
        chunk_total_s=round(chunk_total, 6),
        ingest_file_total_s=round(parse_total + chunk_total, 6),
        retriever_build_total_s=round(float(retriever_build_total), 6),
        first_query_latency_s=round(t_query_end - t_query_start, 6),
        first_query_docs=len(first_docs),
        ingest_build_total_s=round(parse_total + chunk_total + float(retriever_build_total), 6),
        retrieval_hit_rate=retrieval_hit_rate,
        answer_verification_pass_rate=answer_pass_rate,
    )

    return run_metrics, build_profile, [asdict(m) for m in file_metrics]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SmartDoc parse + retriever latency.")
    parser.add_argument("--input-dir", default="samples", help="Primary dataset directory.")
    parser.add_argument("--large-dir", default="samples/perf_large", help="Optional large-doc dataset directory.")
    parser.add_argument("--no-large", action="store_true", help="Disable loading files from --large-dir.")
    parser.add_argument("--query", default="Summarize the key findings from the document.", help="Query for first retrieval latency.")
    parser.add_argument("--runs", type=int, default=1, help="Benchmark runs; use >=3 for meaningful P95.")
    parser.add_argument("--label", default="candidate", help="Label included in output filename.")
    parser.add_argument("--output-dir", default="docs/perf/benchmarks", help="Output directory for JSON report.")
    parser.add_argument(
        "--quality-queries",
        default="",
        help="Optional JSON file with retrieval quality checks: [{\"question\":..., \"expected_sources\":[...]}].",
    )
    args = parser.parse_args()

    roots = [Path(args.input_dir)]
    large_dir = Path(args.large_dir)
    if not args.no_large:
        roots.append(large_dir)

    files = _iter_input_files(roots)
    if not files:
        raise SystemExit("No benchmark files found. Provide files in samples/ or --input-dir.")

    runs = max(1, args.runs)
    quality_queries: List[Dict] = []
    if args.quality_queries:
        quality_queries = json.loads(Path(args.quality_queries).read_text(encoding="utf-8"))
    run_results: List[RunMetrics] = []
    run_build_profiles: List[Dict] = []
    run_file_metrics: List[List[Dict]] = []

    started_at = datetime.now(UTC).isoformat()
    for run_idx in range(1, runs + 1):
        result, profile, file_metrics = run_once(
            files=files,
            query=args.query,
            run_idx=run_idx,
            quality_queries=quality_queries,
        )
        run_results.append(result)
        run_build_profiles.append(profile)
        run_file_metrics.append(file_metrics)

    ingest_build_values = [r.ingest_build_total_s for r in run_results]
    parse_values = [r.parse_total_s for r in run_results]
    chunk_values = [r.chunk_total_s for r in run_results]
    retriever_values = [r.retriever_build_total_s for r in run_results]
    query_values = [r.first_query_latency_s for r in run_results]
    retrieval_hit_values = [r.retrieval_hit_rate for r in run_results]
    answer_pass_values = [r.answer_verification_pass_rate for r in run_results]

    summary = {
        "runs": runs,
        "ingest_build_mean_s": round(statistics.mean(ingest_build_values), 6),
        "ingest_build_p95_s": round(percentile(ingest_build_values, 95), 6),
        "parse_mean_s": round(statistics.mean(parse_values), 6),
        "chunk_mean_s": round(statistics.mean(chunk_values), 6),
        "retriever_build_mean_s": round(statistics.mean(retriever_values), 6),
        "query_mean_s": round(statistics.mean(query_values), 6),
        "parse_p95_s": round(percentile(parse_values, 95), 6),
        "chunk_p95_s": round(percentile(chunk_values, 95), 6),
        "retriever_build_p95_s": round(percentile(retriever_values, 95), 6),
        "query_p95_s": round(percentile(query_values, 95), 6),
        "retrieval_hit_rate": round(statistics.mean(retrieval_hit_values), 6),
        "answer_verification_pass_rate": round(statistics.mean(answer_pass_values), 6),
    }

    report = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "started_at": started_at,
        "label": args.label,
        "input_roots": [str(r) for r in roots if r.exists()],
        "files": [str(f) for f in files],
        "run_results": [asdict(r) for r in run_results],
        "run_build_profiles": run_build_profiles,
        "run_file_metrics": run_file_metrics,
        "summary": summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{stamp}_{args.label}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Benchmark report written to: {output_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
