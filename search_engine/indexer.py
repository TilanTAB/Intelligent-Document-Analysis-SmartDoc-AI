"""
Retriever indexer module for DocChat.

Provides utilities for building different types of retrievers:
- Vector-based retriever (ChromaDB + embeddings)
- Hybrid retriever (BM25 + Vector with ensemble)
"""
import logging
from typing import List, Any
import time
import hashlib
import os
import json
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from core.embedding_factory import get_embeddings
from core.perf_monitor import latency_monitor
from configuration.parameters import parameters

logger = logging.getLogger(__name__)

# Thread lock for manifest file access
_manifest_lock = threading.Lock()


def doc_id(doc) -> str:
    """Generate a unique ID for a document based on source, page, chunk_id, and content hash."""
    src = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    chunk = doc.metadata.get("chunk_id", "")
    # Include content hash to ensure uniqueness even if chunk_id is missing
    content = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
    base = f"{src}::{page}::{chunk}::{content}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def content_hash(doc) -> str:
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def load_manifest(path):
    """Thread-safe manifest loading."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load manifest, starting fresh: {e}")
            return {}
    return {}


def save_manifest(path, manifest):
    """Thread-safe manifest saving with atomic write."""
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(manifest, f)
        os.replace(temp_path, path)  # Atomic rename
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


class EnsembleRetriever(BaseRetriever):
    """
    Custom Ensemble Retriever combining multiple retrievers with weighted RRF.
    
    Attributes:
        retrievers: List of retriever instances
        weights: List of weights (should sum to 1.0)
        c: RRF constant (default: 60)
        k: Max documents to return (default: 10)
    """
    
    retrievers: List[Any]
    weights: List[float]
    c: int = 60
    k: int = 10
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve and combine documents using weighted RRF, deduplicating charts by doc_id and aggregating page numbers."""
        logger.debug(f"[ENSEMBLE] Query: {query[:80]}...")
        all_docs_with_scores = {}
        retriever_names = ["BM25", "Vector"]
        for idx, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            retriever_name = retriever_names[idx] if idx < len(retriever_names) else f"Retriever_{idx}"
            try:
                docs = retriever.invoke(query)
                logger.debug(f"[ENSEMBLE] {retriever_name}: {len(docs)} docs (weight: {weight})")
                for rank, doc in enumerate(docs):
                    # Deduplicate by doc_id only
                    doc_key = doc_id(doc)
                    rrf_score = weight / (rank + 1 + self.c)
                    if doc_key in all_docs_with_scores:
                        existing_doc, existing_score = all_docs_with_scores[doc_key]
                        # Aggregate page numbers
                        existing_pages = set()
                        if isinstance(existing_doc.metadata.get('page'), list):
                            existing_pages.update(existing_doc.metadata['page'])
                        else:
                            existing_pages.add(existing_doc.metadata.get('page'))
                        existing_pages.add(doc.metadata.get('page'))
                        # Update metadata to include all pages
                        existing_doc.metadata['page'] = sorted(p for p in existing_pages if p is not None)
                        all_docs_with_scores[doc_key] = (existing_doc, existing_score + rrf_score)
                    else:
                        all_docs_with_scores[doc_key] = (doc, rrf_score)
            except Exception as e:
                logger.warning(f"[ENSEMBLE] {retriever_name} failed: {e}")
                continue
        sorted_docs = sorted(all_docs_with_scores.values(), key=lambda x: x[1], reverse=True)
        result = [doc for doc, score in sorted_docs[:self.k]]
        logger.debug(f"[ENSEMBLE] Returning {len(result)} documents")
        return result


class RetrieverBuilder:
    """Builder class for creating document retrievers with caching."""
    
    def __init__(self):
        """Initialize with embeddings model."""
        self.embeddings = get_embeddings()
        self._retriever_cache = {}  # {docset_hash: retriever}
        self._bm25_cache = {}  # {docset_hash: bm25_retriever} - NEW: Cache BM25 retrievers
        self._vector_store_cache = {}  # {chroma_dir: vector_store} - NEW: Reuse ChromaDB connections
        self.last_build_profile = {}
        logger.debug("RetrieverBuilder initialized with caching enabled")

    def _hash_docs(self, docs):
        # Create a hash of all document contents and metadata
        m = hashlib.sha256()
        for doc in docs:
            m.update(doc.page_content.encode('utf-8'))
            for k, v in sorted(doc.metadata.items()):
                m.update(str(k).encode('utf-8'))
                m.update(str(v).encode('utf-8'))
        return m.hexdigest()

    def _trim_docs_for_indexing(self, docs: List[Document]) -> List[Document]:
        """
        Optionally trim the chunk set before vector ingest.
        Keeps chart/table chunks first, then evenly samples text chunks.
        """
        max_chunks = max(0, int(getattr(parameters, "MAX_INDEX_CHUNKS", 0) or 0))
        if max_chunks <= 0 or len(docs) <= max_chunks:
            return docs

        priority: List[tuple[int, Document]] = []
        normal: List[tuple[int, Document]] = []
        for pos, doc in enumerate(docs):
            doc_type = str(doc.metadata.get("type", "text")).lower()
            if "chart" in doc_type or "table" in doc_type:
                priority.append((pos, doc))
            else:
                normal.append((pos, doc))

        selected: List[tuple[int, Document]] = []
        selected.extend(priority[:max_chunks])
        remaining = max_chunks - len(selected)

        if remaining > 0 and normal:
            if len(normal) <= remaining:
                selected.extend(normal)
            else:
                # Evenly sample across the normal chunk list to preserve corpus coverage.
                seen = set()
                total = len(normal)
                for i in range(remaining):
                    idx = min(int(i * total / remaining), total - 1)
                    while idx in seen and idx + 1 < total:
                        idx += 1
                    if idx in seen:
                        for j in range(total):
                            if j not in seen:
                                idx = j
                                break
                    seen.add(idx)
                    selected.append(normal[idx])

        selected.sort(key=lambda item: item[0])
        trimmed_docs = [doc for _, doc in selected[:max_chunks]]
        logger.info(
            "[PROFILE] stage=vector.pre_ingest.trim original=%s indexed=%s max=%s",
            len(docs),
            len(trimmed_docs),
            max_chunks,
        )
        return trimmed_docs

    def _build_chroma_metadata(self) -> dict:
        return {
            "hnsw:M": int(parameters.CHROMA_HNSW_M),
            "hnsw:construction_ef": int(parameters.CHROMA_HNSW_CONSTRUCTION_EF),
            "hnsw:search_ef": int(parameters.CHROMA_HNSW_SEARCH_EF),
            "hnsw:num_threads": int(parameters.CHROMA_HNSW_NUM_THREADS),
            "hnsw:batch_size": int(parameters.CHROMA_HNSW_BATCH_SIZE),
            "hnsw:sync_threshold": int(parameters.CHROMA_HNSW_SYNC_THRESHOLD),
        }

    def _create_vector_store(self, chroma_dir: str):
        metadata = self._build_chroma_metadata()
        common_kwargs = {
            "embedding_function": self.embeddings,
            "persist_directory": chroma_dir,
            "collection_name": parameters.CHROMA_COLLECTION_NAME,
        }
        try:
            return Chroma(collection_metadata=metadata, **common_kwargs)
        except TypeError:
            logger.warning("collection_metadata not supported by installed langchain-chroma; continuing without HNSW metadata.")
            return Chroma(**common_kwargs)

    def _create_or_get_vector_store(self, chroma_dir: str):
        if chroma_dir in self._vector_store_cache:
            logger.debug(f"Reusing cached vector store connection for {chroma_dir}")
            return self._vector_store_cache[chroma_dir]

        vector_store = self._create_vector_store(chroma_dir)
        self._vector_store_cache[chroma_dir] = vector_store
        logger.debug(f"Created new vector store connection for {chroma_dir}")
        return vector_store

    def _resolve_ingest_batch_size(self, vector_store) -> int:
        configured = max(1, int(parameters.CHROMA_INGEST_BATCH_SIZE))
        client = getattr(vector_store, "_client", None)
        max_batch = getattr(client, "max_batch_size", None)
        if callable(max_batch):
            try:
                max_batch = max_batch()
            except Exception:
                max_batch = None
        if isinstance(max_batch, int) and max_batch > 0:
            effective = min(configured, max_batch)
            if effective != configured:
                logger.info(f"[PROFILE] stage=vector.ingest.batch_size configured={configured} limited_by_client={max_batch} effective={effective}")
            return effective
        return configured

    def _resolve_parallel_workers(self, total_batches: int) -> int:
        configured = max(1, int(getattr(parameters, "VECTOR_INGEST_PARALLEL_WORKERS", 1)))
        return min(configured, max(1, total_batches))

    @staticmethod
    def _is_retryable_embedding_error(exc: Exception) -> bool:
        """
        Heuristic check for transient embedding/indexing failures.
        """
        retryable_type_names = {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "ServiceUnavailableError",
            "TimeoutError",
            "ConnectionError",
        }
        if type(exc).__name__ in retryable_type_names:
            return True
        message = str(exc).lower()
        transient_markers = (
            "rate limit",
            "429",
            "too many requests",
            "timeout",
            "timed out",
            "temporar",
            "service unavailable",
            "connection",
            "server error",
            "503",
            "502",
            "retry",
            "locked",
        )
        return any(marker in message for marker in transient_markers)

    @staticmethod
    def _compute_backoff_delay(attempt: int) -> float:
        base = max(0.1, float(parameters.EMBEDDING_BACKOFF_BASE_S))
        max_delay = max(base, float(parameters.EMBEDDING_BACKOFF_MAX_S))
        jitter_ratio = min(1.0, max(0.0, float(parameters.EMBEDDING_BACKOFF_JITTER_RATIO)))
        exponential = min(max_delay, base * (2 ** max(0, attempt - 1)))
        jittered = exponential * (1 + random.uniform(-jitter_ratio, jitter_ratio))
        return max(0.05, min(max_delay, jittered))

    def _add_batch_with_retry(
        self,
        vector_store,
        batch_docs: List[Document],
        batch_ids: List[str],
        batch_idx: int,
    ) -> float:
        batch_start = time.time()
        max_attempts = max(1, int(parameters.EMBEDDING_INGEST_MAX_RETRIES))
        for attempt in range(1, max_attempts + 1):
            try:
                vector_store.add_documents(batch_docs, ids=batch_ids)
                return time.time() - batch_start
            except Exception as e:
                retryable = self._is_retryable_embedding_error(e)
                if (not retryable) or attempt >= max_attempts:
                    logger.error(
                        "Vector ingest batch failed idx=%s attempt=%s/%s retryable=%s: %s",
                        batch_idx,
                        attempt,
                        max_attempts,
                        retryable,
                        e,
                    )
                    raise

                delay_s = self._compute_backoff_delay(attempt=attempt)
                logger.warning(
                    "[PROFILE] stage=vector.ingest.batch.retry idx=%s attempt=%s/%s delay_s=%.3f reason=%s",
                    batch_idx,
                    attempt,
                    max_attempts,
                    delay_s,
                    type(e).__name__,
                )
                time.sleep(delay_s)

        # Defensive: loop should always return or raise.
        raise RuntimeError(f"Vector ingest batch {batch_idx} failed unexpectedly")

    def _ingest_batches_sequential(
        self,
        vector_store,
        uniq_docs: List[Document],
        uniq_ids: List[str],
        batch_size: int,
        build_profile: dict,
    ) -> None:
        total_batches = (len(uniq_docs) - 1) // batch_size + 1
        for i in range(0, len(uniq_docs), batch_size):
            batch_idx = i // batch_size + 1
            batch_docs = uniq_docs[i:i + batch_size]
            batch_ids = uniq_ids[i:i + batch_size]
            batch_duration = self._add_batch_with_retry(
                vector_store=vector_store,
                batch_docs=batch_docs,
                batch_ids=batch_ids,
                batch_idx=batch_idx,
            )
            build_profile["vector_ingest_batch_durations_s"].append(round(batch_duration, 6))
            latency_monitor.record(
                stage="embed.batch",
                duration_s=batch_duration,
                metadata={"batch_idx": batch_idx, "docs": len(batch_docs), "mode": "sequential"},
            )
            if total_batches > 1:
                logger.info(
                    "[PROFILE] stage=vector.ingest.batch idx=%s duration_s=%.3f docs=%s mode=sequential",
                    batch_idx,
                    batch_duration,
                    len(batch_docs),
                )

    def _ingest_batches_parallel(
        self,
        chroma_dir: str,
        uniq_docs: List[Document],
        uniq_ids: List[str],
        batch_size: int,
        build_profile: dict,
    ) -> None:
        total_batches = (len(uniq_docs) - 1) // batch_size + 1
        workers = self._resolve_parallel_workers(total_batches=total_batches)
        if workers <= 1 or total_batches <= 1:
            build_profile["vector_ingest_mode"] = "sequential"
            vector_store = self._create_or_get_vector_store(chroma_dir)
            self._ingest_batches_sequential(
                vector_store=vector_store,
                uniq_docs=uniq_docs,
                uniq_ids=uniq_ids,
                batch_size=batch_size,
                build_profile=build_profile,
            )
            return

        build_profile["vector_ingest_mode"] = "parallel"
        build_profile["vector_ingest_parallel_workers"] = workers
        logger.info(
            "[PROFILE] stage=vector.ingest.parallel workers=%s batches=%s",
            workers,
            total_batches,
        )

        batches = []
        for i in range(0, len(uniq_docs), batch_size):
            batch_idx = i // batch_size + 1
            batches.append((batch_idx, uniq_docs[i:i + batch_size], uniq_ids[i:i + batch_size]))

        batch_durations = [0.0] * total_batches

        def ingest_one(batch_tuple):
            batch_idx, batch_docs, batch_ids = batch_tuple
            worker_store = self._create_vector_store(chroma_dir)
            duration = self._add_batch_with_retry(
                vector_store=worker_store,
                batch_docs=batch_docs,
                batch_ids=batch_ids,
                batch_idx=batch_idx,
            )
            return batch_idx, duration, len(batch_docs)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vec-ingest") as executor:
            futures = {executor.submit(ingest_one, batch): batch[0] for batch in batches}
            try:
                for future in as_completed(futures):
                    batch_idx, duration, docs_count = future.result()
                    batch_durations[batch_idx - 1] = round(duration, 6)
                    latency_monitor.record(
                        stage="embed.batch",
                        duration_s=duration,
                        metadata={"batch_idx": batch_idx, "docs": docs_count, "mode": "parallel"},
                    )
                    if total_batches > 1:
                        logger.info(
                            "[PROFILE] stage=vector.ingest.batch idx=%s duration_s=%.3f docs=%s mode=parallel",
                            batch_idx,
                            duration,
                            docs_count,
                        )
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise

        build_profile["vector_ingest_batch_durations_s"].extend(batch_durations)

    # ──────────────────────────────────────────────────────────────────────────
    # B2: Streaming chunker — consumer-side helpers
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_streaming(
        self,
        chunk_queue: "queue.Queue",  # queue.Queue fed by _process_file(chunk_queue=...)
        chroma_dir: str,
    ) -> tuple:
        """
        B2: Consume chunks from a queue and ingest into ChromaDB as they arrive.

        Runs in the consumer thread while _process_file runs in the producer thread.
        Returns (collected_chunks, vector_store) so the caller can build BM25 +
        EnsembleRetriever after both threads join.

        Key design: batches are flushed to ChromaDB as soon as batch_size chunks
        accumulate — this overlaps embedding API calls with the tail of parsing,
        which is the entire point of B2.

        Protocol: producer puts Document objects one by one, then puts None as sentinel.
        """
        import queue as _q

        os.makedirs(chroma_dir, exist_ok=True)
        manifest_path = os.path.join(chroma_dir, "indexed_manifest.json")

        # Load manifest under lock — same pattern as build_hybrid_retriever
        with _manifest_lock:
            manifest = load_manifest(manifest_path)

        vector_store = self._create_or_get_vector_store(chroma_dir)
        batch_size = self._resolve_ingest_batch_size(vector_store)

        collected_chunks: List[Document] = []  # full list returned for BM25 build
        current_doc_ids: set = set()

        accumulator: List[Document] = []       # in-flight batch waiting to reach batch_size
        accumulator_ids: List[str] = []
        batch_idx = 0
        t_start = time.time()

        def flush():
            """Send accumulated batch to ChromaDB via existing retry logic.

            On failure, rolls back manifest entries for the un-persisted batch
            so a future retry doesn't skip them (manifest says 'indexed' but
            ChromaDB never received them).
            """
            nonlocal batch_idx
            if not accumulator:
                return
            batch_idx += 1
            pending_ids = list(accumulator_ids)  # snapshot before clear
            try:
                self._add_batch_with_retry(
                    vector_store=vector_store,
                    batch_docs=list(accumulator),
                    batch_ids=pending_ids,
                    batch_idx=batch_idx,
                )
            except Exception:
                # Rollback: remove manifest entries for chunks that never reached ChromaDB
                for pid in pending_ids:
                    manifest.pop(pid, None)
                logger.warning(
                    "[STREAMING] flush() failed — rolled back %d manifest entries for batch %d",
                    len(pending_ids), batch_idx,
                )
                raise
            logger.info(
                "[STREAMING] stage=embed.batch idx=%s size=%s elapsed_s=%.1f",
                batch_idx, len(accumulator), time.time() - t_start,
            )
            accumulator.clear()
            accumulator_ids.clear()

        # ── Drain queue, flushing batches as they fill ──────────────────────────
        # Each chunk is evaluated against the manifest for idempotency and
        # appended to the accumulator.  When the accumulator reaches batch_size
        # we flush immediately — the embedding API call runs while the producer
        # thread continues parsing the next pages.
        while True:
            try:
                item = chunk_queue.get(timeout=120)  # 2-min safety timeout
            except _q.Empty:
                raise RuntimeError(
                    "[STREAMING] Consumer timed out waiting for producer (120s) — aborting"
                )

            if item is None:   # sentinel: producer finished
                break

            chunk = item
            collected_chunks.append(chunk)

            _id = doc_id(chunk)
            _hash = content_hash(chunk)
            current_doc_ids.add(_id)

            # Idempotent: skip chunks already in manifest with same content hash
            # (protects against partial re-runs after a previous failure)
            if manifest.get(_id) == _hash:
                continue

            manifest[_id] = _hash
            accumulator.append(chunk)
            accumulator_ids.append(_id)

            if len(accumulator) >= batch_size:
                flush()

        # Flush any remaining partial batch after sentinel
        flush()

        logger.info(
            "[STREAMING] Queue drained: chunks=%s batches=%s elapsed_s=%.1f",
            len(collected_chunks), batch_idx, time.time() - t_start,
        )

        # ── Prune stale docs and save manifest ──────────────────────────────────
        stale_ids = [eid for eid in list(manifest.keys()) if eid not in current_doc_ids]
        if stale_ids:
            try:
                vector_store.delete(ids=stale_ids)
                for sid in stale_ids:
                    manifest.pop(sid, None)
                logger.info("[STREAMING] stage=vector.prune stale_removed=%s", len(stale_ids))
            except Exception as e:
                logger.warning("[STREAMING] Failed to prune stale IDs: %s", e)

        with _manifest_lock:
            save_manifest(manifest_path, manifest)

        logger.info(
            "[STREAMING] Ingest complete: consumed=%s batches=%s elapsed_s=%.1f",
            len(collected_chunks), batch_idx, time.time() - t_start,
        )
        return collected_chunks, vector_store

    def _assemble_ensemble(
        self,
        docs_for_index: List[Document],
        cache_key: str,
        vector_store,
    ) -> EnsembleRetriever:
        """Shared helper: BM25 build + vector retriever wrap + EnsembleRetriever.

        Both build_bm25_and_ensemble (B2 streaming) and build_hybrid_retriever
        delegate here so the BM25/ensemble construction stays in one place.
        """
        # BM25 — needs all chunks at once, cannot be done incrementally
        t_bm25_start = time.time()
        if cache_key in self._bm25_cache:
            bm25_retriever = self._bm25_cache[cache_key]
            logger.debug("Reusing cached BM25 for docset %s...", cache_key[:8])
        else:
            texts = [doc.page_content for doc in docs_for_index]
            metadatas = [doc.metadata for doc in docs_for_index]
            bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
            bm25_retriever.k = parameters.BM25_SEARCH_K
            self._bm25_cache[cache_key] = bm25_retriever
        logger.info(
            "[PROFILE] stage=bm25.build duration_s=%.3f docs=%s",
            time.time() - t_bm25_start, len(docs_for_index),
        )

        # Vector retriever — wraps the already-populated store
        vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": parameters.VECTOR_SEARCH_K_CHROMA,
                "fetch_k": parameters.VECTOR_FETCH_K,
                "lambda_mult": 0.7,
            },
        )

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=parameters.HYBRID_RETRIEVER_WEIGHTS,
            k=parameters.VECTOR_SEARCH_K,
        )

        self._retriever_cache[cache_key] = hybrid_retriever
        logger.info("EnsembleRetriever built (k=%s)", parameters.VECTOR_SEARCH_K)
        return hybrid_retriever

    def build_bm25_and_ensemble(
        self,
        all_chunks: List[Document],
        vector_store,
    ) -> EnsembleRetriever:
        """
        B2: Build BM25 retriever + EnsembleRetriever from a complete chunk list
        and an already-populated vector store.

        Called in the main thread after both producer and consumer threads join.
        """
        if not all_chunks:
            raise ValueError("No chunks provided to build_bm25_and_ensemble")

        docs_for_index = self._trim_docs_for_indexing(all_chunks)
        cache_key = self._hash_docs(docs_for_index)
        return self._assemble_ensemble(docs_for_index, cache_key, vector_store)

    def build_hybrid_retriever(self, docs, session_id: str = None) -> EnsembleRetriever:
        """
        Build hybrid retriever using BM25 and vector search.
        
        Args:
            docs: List of documents to index
            session_id: Optional session ID for user isolation (recommended for multi-user)
            
        Returns:
            EnsembleRetriever combining BM25 and vector search
        """
        logger.info(f"Building hybrid retriever with {len(docs)} documents...")
        if not docs:
            raise ValueError("No documents provided")
        docs_for_index = self._trim_docs_for_indexing(docs)
        build_profile = {
            "docs_total": len(docs),
            "docs_indexed": len(docs_for_index),
            "cache_hit": False,
            "vector_ingest_batch_durations_s": [],
            "vector_ingest_docs_added": 0,
            "vector_ingest_batches": 0,
        }
        
        # Generate cache key from document content hashes
        cache_key = self._hash_docs(docs_for_index)
        
        # Check retriever cache first (10-200× speedup for repeat queries)
        if cache_key in self._retriever_cache:
            logger.info(f"✅ Using cached retriever for docset {cache_key[:8]}... (CACHE HIT)")
            build_profile["cache_hit"] = True
            self.last_build_profile = build_profile
            return self._retriever_cache[cache_key]
        
        logger.debug(f"Cache miss for docset {cache_key[:8]}..., building new retriever")
        
        # Use session-specific directory if provided (for multi-user isolation)
        if session_id:
            chroma_dir = os.path.join(parameters.CHROMA_DB_PATH, f"session_{session_id}")
        else:
            chroma_dir = parameters.CHROMA_DB_PATH
            
        manifest_path = os.path.join(chroma_dir, "indexed_manifest.json")
        os.makedirs(chroma_dir, exist_ok=True)
        
        # Thread-safe manifest access
        with _manifest_lock:
            manifest = load_manifest(manifest_path)
        
        t_vector_start = time.time()
        
        vector_store = self._create_or_get_vector_store(chroma_dir)

        to_add = []
        ids_to_add = []
        to_delete_ids = []
        current_doc_ids = set()
        
        for d in docs_for_index:
            _id = doc_id(d)
            _hash = content_hash(d)
            current_doc_ids.add(_id)
            if _id not in manifest:
                to_add.append(d)
                ids_to_add.append(_id)
                manifest[_id] = _hash
            elif manifest[_id] != _hash:
                to_delete_ids.append(_id)
                to_add.append(d)
                ids_to_add.append(_id)
                manifest[_id] = _hash

        # Remove stale vectors that are no longer part of this docset.
        stale_ids = [existing_id for existing_id in list(manifest.keys()) if existing_id not in current_doc_ids]
        if stale_ids:
            to_delete_ids.extend(stale_ids)
            for stale_id in stale_ids:
                manifest.pop(stale_id, None)
            build_profile["vector_pruned_ids"] = len(stale_ids)
            logger.info(
                "[PROFILE] stage=vector.prune count=%s",
                len(stale_ids),
            )

        if to_delete_ids:
            t_delete_start = time.time()
            try:
                vector_store.delete(ids=to_delete_ids)
            except Exception as e:
                logger.warning(f"Failed to delete stale vector ids ({len(to_delete_ids)}): {e}")
            t_delete_end = time.time()
            logger.info(f"[PROFILE] stage=vector.delete duration_s={t_delete_end - t_delete_start:.3f} count={len(to_delete_ids)}")
            build_profile["vector_deleted_ids"] = len(to_delete_ids)

        if to_add:
            # Safety net: de-dupe before add_documents
            seen = set()
            uniq_docs, uniq_ids = [], []
            for doc, _id in zip(to_add, ids_to_add):
                if _id in seen:
                    continue
                seen.add(_id)
                uniq_docs.append(doc)
                uniq_ids.append(_id)
            
            # Log duplicate count for debugging
            dupe_count = len(to_add) - len(uniq_docs)
            if dupe_count > 0:
                logger.debug(f"Filtered {dupe_count} duplicate documents before indexing")
            
            # Batch add documents for better performance.
            batch_size = self._resolve_ingest_batch_size(vector_store)
            total_batches = (len(uniq_docs) - 1) // batch_size + 1
            logger.info(f"[PROFILE] stage=vector.ingest.start docs={len(uniq_docs)} batch_size={batch_size} batches={total_batches}")
            t_add_start = time.time()
            build_profile["vector_ingest_docs_added"] = len(uniq_docs)
            build_profile["vector_ingest_batch_size"] = batch_size
            build_profile["vector_ingest_batches"] = total_batches
            try:
                self._ingest_batches_parallel(
                    chroma_dir=chroma_dir,
                    uniq_docs=uniq_docs,
                    uniq_ids=uniq_ids,
                    batch_size=batch_size,
                    build_profile=build_profile,
                )
            except Exception as e:
                logger.warning(
                    "Parallel vector ingest failed; falling back to sequential mode. reason=%s",
                    e,
                )
                build_profile["vector_ingest_parallel_fallback"] = True
                build_profile["vector_ingest_batch_durations_s"] = []
                build_profile["vector_ingest_mode"] = "sequential_fallback"
                self._ingest_batches_sequential(
                    vector_store=vector_store,
                    uniq_docs=uniq_docs,
                    uniq_ids=uniq_ids,
                    batch_size=batch_size,
                    build_profile=build_profile,
                )
            
            t_add_end = time.time()
            logger.info(f"[PROFILE] stage=vector.ingest.total duration_s={t_add_end - t_add_start:.3f} docs={len(uniq_docs)}")
            build_profile["vector_ingest_total_s"] = round(t_add_end - t_add_start, 6)

            # Parallel workers use independent Chroma clients; refresh the cached client
            # so downstream retrieval always sees the latest writes.
            if build_profile.get("vector_ingest_mode") == "parallel":
                vector_store = self._create_vector_store(chroma_dir)
                self._vector_store_cache[chroma_dir] = vector_store
        
        t_vector_end = time.time()
        logger.info(f"[PROFILE] stage=vector.setup.total duration_s={t_vector_end - t_vector_start:.3f}")
        build_profile["vector_setup_total_s"] = round(t_vector_end - t_vector_start, 6)
        
        # Thread-safe manifest save
        with _manifest_lock:
            save_manifest(manifest_path, manifest)
        
        # BM25 + vector retriever + ensemble — delegated to shared helper
        t_ensemble_start = time.time()
        hybrid_retriever = self._assemble_ensemble(docs_for_index, cache_key, vector_store)
        t_ensemble_end = time.time()
        logger.info(f"[PROFILE] stage=retriever.build.total duration_s={t_ensemble_end - t_vector_start:.3f}")
        build_profile["retriever_build_total_s"] = round(t_ensemble_end - t_vector_start, 6)

        self.last_build_profile = build_profile
        logger.debug(f"Cached retriever for docset {cache_key[:8]}... (future requests will be instant)")

        return hybrid_retriever
