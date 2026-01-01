"""
Retriever indexer module for DocChat.

Provides utilities for building different types of retrievers:
- Vector-based retriever (ChromaDB + embeddings)
- Hybrid retriever (BM25 + Vector with ensemble)
"""
import logging
import sys
from typing import List, Any
import time
import hashlib
import os
import json
import threading

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
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
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=parameters.GOOGLE_API_KEY,
            batch_size=100,  # Increased from 32 to 100 for 3× faster embedding (Google supports up to 100)
        )
        self._retriever_cache = {}  # {docset_hash: retriever}
        self._bm25_cache = {}  # {docset_hash: bm25_retriever} - NEW: Cache BM25 retrievers
        self._vector_store_cache = {}  # {chroma_dir: vector_store} - NEW: Reuse ChromaDB connections
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
        
        # Generate cache key from document content hashes
        cache_key = self._hash_docs(docs)
        
        # Check retriever cache first (10-200× speedup for repeat queries)
        if cache_key in self._retriever_cache:
            logger.info(f"✅ Using cached retriever for docset {cache_key[:8]}... (CACHE HIT)")
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
        
        # Check vector store cache (reuse ChromaDB connections)
        if chroma_dir in self._vector_store_cache:
            logger.debug(f"Reusing cached vector store connection for {chroma_dir}")
            vector_store = self._vector_store_cache[chroma_dir]
        else:
            vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=chroma_dir,
            )
            self._vector_store_cache[chroma_dir] = vector_store
            logger.debug(f"Created new vector store connection for {chroma_dir}")

        to_add = []
        ids_to_add = []
        to_delete_ids = []
        current_ids = set()
        
        for d in docs:
            _id = doc_id(d)
            _hash = content_hash(d)
            current_ids.add(_id)
            if _id not in manifest:
                to_add.append(d)
                ids_to_add.append(_id)
                manifest[_id] = _hash
            elif manifest[_id] != _hash:
                to_delete_ids.append(_id)
                to_add.append(d)
                ids_to_add.append(_id)
                manifest[_id] = _hash             
        
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
            
            # Batch add documents for better performance
            logger.info(f"[PROFILE] Adding {len(uniq_docs)} new documents to vector store...")
            t_add_start = time.time()
            
            # Add in batches for progress tracking and memory efficiency
            batch_size = 100
            for i in range(0, len(uniq_docs), batch_size):
                batch_docs = uniq_docs[i:i+batch_size]
                batch_ids = uniq_ids[i:i+batch_size]
                vector_store.add_documents(batch_docs, ids=batch_ids)
                if len(uniq_docs) > batch_size:
                    logger.debug(f"[PROFILE] Indexed batch {i//batch_size + 1}/{(len(uniq_docs)-1)//batch_size + 1}")
            
            t_add_end = time.time()
            logger.info(f"[PROFILE] Vector store add_documents: {t_add_end - t_add_start:.2f}s")
        
        t_vector_end = time.time()
        logger.info(f"[PROFILE] Total vector store setup: {t_vector_end - t_vector_start:.2f}s")
        
        # Thread-safe manifest save
        with _manifest_lock:
            save_manifest(manifest_path, manifest)
        
        # Create BM25 retriever
        t_bm25_start = time.time()
        
        # Check BM25 cache (avoid rebuilding for same documents)
        if cache_key in self._bm25_cache:
            logger.debug(f"Reusing cached BM25 retriever for docset {cache_key[:8]}...")
            bm25_retriever = self._bm25_cache[cache_key]
        else:
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
            bm25_retriever.k = parameters.BM25_SEARCH_K
            self._bm25_cache[cache_key] = bm25_retriever
            logger.debug(f"Created new BM25 retriever for docset {cache_key[:8]}...")
        
        t_bm25_end = time.time()
        logger.info(f"[PROFILE] BM25 retriever creation: {t_bm25_end - t_bm25_start:.2f}s")
        logger.debug(f"BM25 indexed {len(docs)} texts, k={bm25_retriever.k}")
        
        t_vec_retr_start = time.time()
        vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": parameters.VECTOR_SEARCH_K_CHROMA,
                "fetch_k": parameters.VECTOR_FETCH_K,
                "lambda_mult": 0.7,
            },
        )
        t_vec_retr_end = time.time()
        logger.info(f"[PROFILE] Vector retriever creation: {t_vec_retr_end - t_vec_retr_start:.2f}s")
        logger.debug("Vector retriever created")
        
        t_ensemble_start = time.time()
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=parameters.HYBRID_RETRIEVER_WEIGHTS,                                                                
            k=parameters.VECTOR_SEARCH_K,
        )
        t_ensemble_end = time.time()
        logger.info(f"[PROFILE] Ensemble retriever creation: {t_ensemble_end - t_ensemble_start:.2f}s")
        logger.info(f"Hybrid retriever created (k={parameters.VECTOR_SEARCH_K})")
        logger.info(f"[PROFILE] Total hybrid retriever build: {t_ensemble_end - t_vector_start:.2f}s")
        
        # Cache the complete retriever for future use
        self._retriever_cache[cache_key] = hybrid_retriever
        logger.debug(f"Cached retriever for docset {cache_key[:8]}... (future requests will be instant)")
        
        return hybrid_retriever
