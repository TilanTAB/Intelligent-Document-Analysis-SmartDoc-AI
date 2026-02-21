# Performance Recheck - 2026-02-16

## App Run Check
- App started successfully.
- Log evidence:
  - `Launching Gradio on port 7861`
  - `Access the app at: http://127.0.0.1:7861`

## Effective Runtime Settings
- `EMBEDDING_CACHE_ENABLED=True`
- `VECTOR_INGEST_PARALLEL_WORKERS=1`
- `MAX_INDEX_CHUNKS=2000`
- `CHROMA_INGEST_BATCH_SIZE=100`

## Benchmark Run
Command:
```bash
python scripts/benchmark_latency.py --input-dir samples --no-large --runs 2 --label perf_recheck_20260216 --query "What is the accuracy of AI models in coding?"
```

Report:
- `docs/perf/benchmarks/20260216T091015Z_perf_recheck_20260216.json`

Summary (mean):
- ingest+build: `134.018s`
- parse: `111.700s`
- chunk: `0.045s`
- retriever build: `22.273s`
- first query retrieval: `1.355s`

Per-run retriever profile:
- run1: `docs_total=2559`, `docs_indexed=2000`, `vector_ingest_mode=sequential`, `vector_ingest_total_s=23.290`, `retriever_build_total_s=24.608`
- run2: `docs_total=2559`, `docs_indexed=2000`, `vector_ingest_mode=sequential`, `vector_ingest_total_s=18.993`, `retriever_build_total_s=19.938`

## Delta vs Original Baseline
Baseline report:
- `docs/perf/benchmarks/20260214T044923Z_bottleneck_example.json`

Changes:
- ingest+build: `215.416s -> 134.018s` (`-37.79%`)
- retriever build: `118.021s -> 22.273s` (`-81.13%`)
- parse: `97.350s -> 111.700s` (`+14.74%`)

## Notes
- The PDF parsing warnings still appear (`unpack requires a buffer of 4 bytes` on multiple pages), and parse remains the largest stage now.

## Fresh Example Sweep Recheck (Clean Cache + Vector DB)
- Runtime check confirmed active vector DB path is `CHROMA_DB_PATH=./chroma_db` (not `vector_store`).
- Per request, cache/vector artifacts were cleared before rerun:
  - `document_cache/`
  - `chroma_db/`
  - stale `vector_store/` removed

Attempted command:
```bash
python scripts/benchmark_latency.py --input-dir samples --no-large --runs 1 --label fresh_research_examples_20260216_rerun --query "What is the accuracy of AI models in coding?"
```

Observed execution issue:
- The rerun exceeded `30m` and timed out in this shell session.
- It also left a stale Python process that held a Chroma file lock (`data_level0.bin`), requiring process termination before cleanup.

Latest completed fresh example benchmark available:
- `docs/perf/benchmarks/20260216T144717Z_fresh_research_examples_20260216.json`

Summary (`runs=1`):
- ingest+build: `197.288s`
- parse: `106.719s`
- chunk: `0.045s`
- retriever build: `90.524s`
- first query retrieval: `0.962s`
- retriever profile: `docs_total=2609`, `docs_indexed=2000`, `vector_ingest_mode=sequential`, `vector_ingest_total_s=88.795`

Delta vs original baseline (`20260214T044923Z_bottleneck_example.json`):
- ingest+build: `215.416s -> 197.288s` (`-8.42%`)
- retriever build: `118.021s -> 90.524s` (`-23.30%`)
- parse: `97.350s -> 106.719s` (`+9.62%`, slower)
- first query retrieval: `0.365s -> 0.962s` (slower)

Failure/regression check:
- ✅ No immediate benchmark crash in the latest completed fresh report.
- ⚠️ Operational regression remains: long-running benchmark jobs can exceed shell timeout and leave Chroma lock contention.
