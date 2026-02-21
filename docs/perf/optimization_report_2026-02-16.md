# Optimization Report - 2026-02-16

## Implemented

### 2) Reduce chunks before vector ingest
- Added `MAX_INDEX_CHUNKS` config (default: `2000`) in `configuration/parameters.py`.
- Added pre-ingest trimming in `search_engine/indexer.py` (`_trim_docs_for_indexing`).
- Trim strategy:
  - keep chart/table chunks first
  - evenly sample remaining text chunks
  - preserve original order for selected chunks

### 3) Reuse retriever for unchanged file set
- Added file-set signature reuse in `main.py`.
- If uploaded files are unchanged in the current session:
  - skip chunking/re-indexing
  - reuse session retriever directly

### Parallel vector ingest
- Added `VECTOR_INGEST_PARALLEL_WORKERS` config (default: `1` for safe baseline).
- Added parallel ingest path in `search_engine/indexer.py` (`_ingest_batches_parallel`).
- Added resilient fallback to sequential mode on failure.
- Added client refresh after parallel writes to ensure the retriever sees latest vectors.

## Benchmarks

Baseline (before these changes):
- `docs/perf/benchmarks/20260214T044923Z_bottleneck_example.json`
- ingest+build: `215.416s`
- retriever build: `118.021s`

Post-change (parallel=2, embedding cache disabled):
- `docs/perf/benchmarks/20260216T084835Z_post_opt_impl_parallel2_nocache.json`
- ingest+build: `186.516s` (`-13.42%`)
- retriever build: `84.887s` (`-28.08%`)
- docs total/indexed: `2559 -> 2000`
- ingest mode: `parallel`, workers: `2`

Additional warm-cache validation run:
- `docs/perf/benchmarks/20260216T084227Z_post_opt_impl_parallel2_refresh.json`

## Notes
- PDF parsing errors still appear (`unpack requires a buffer of 4 bytes`) and remain a separate bottleneck.
- Retrieval quality metrics in benchmark JSON remain `0.0` because no quality query set was supplied.
