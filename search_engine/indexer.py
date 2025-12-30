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

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from configuration.parameters import parameters

logger = logging.getLogger(__name__)


def doc_id(doc) -> str:
    src = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    chunk = doc.metadata.get("chunk_id", "")
    base = f"{src}::{page}::{chunk}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def content_hash(doc) -> str:
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def load_manifest(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_manifest(path, manifest):
    with open(path, "w") as f:
        json.dump(manifest, f)


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
            batch_size=32,  # Enable batching for faster embedding computation
        )
        self._retriever_cache = {}  # {docset_hash: retriever}

    def _hash_docs(self, docs):
        # Create a hash of all document contents and metadata
        m = hashlib.sha256()
        for doc in docs:
            m.update(doc.page_content.encode('utf-8'))
            for k, v in sorted(doc.metadata.items()):
                m.update(str(k).encode('utf-8'))
                m.update(str(v).encode('utf-8'))
        return m.hexdigest()

    def build_hybrid_retriever(self, docs) -> EnsembleRetriever:
        """
        Build hybrid retriever using BM25 and vector search.
        
        Args:
            docs: List of documents to index
            
        Returns:
            EnsembleRetriever combining BM25 and vector search
        """
        logger.info(f"Building hybrid retriever with {len(docs)} documents...")
        if not docs:
            raise ValueError("No documents provided")
        chroma_dir = parameters.CHROMA_DB_PATH
        manifest_path = os.path.join(chroma_dir, "indexed_manifest.json")
        os.makedirs(chroma_dir, exist_ok=True)
        manifest = load_manifest(manifest_path)
        vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )
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
            # Debugging: show duplicate IDs and their sources
            from collections import Counter
            counts = Counter(ids_to_add)
            dupes = [i for i, c in counts.items() if c > 1]
            if dupes:
                print("Duplicate IDs:", len(dupes))
                for d in dupes[:10]:
                    idxs = [k for k, x in enumerate(ids_to_add) if x == d]
                    print("ID:", d, "examples:")
                    for k in idxs[:3]:
                        md = to_add[k].metadata
                        print("  ", md.get("source"), md.get("page"), md.get("chunk_index"))
            vector_store.add_documents(uniq_docs, ids=uniq_ids)
        save_manifest(manifest_path, manifest)
        # Create BM25 retriever
        t_bm25_start = time.time()
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
        bm25_retriever.k = parameters.BM25_SEARCH_K
        t_bm25_end = time.time()
        logger.info(f"[PROFILE] BM25 retriever creation: {t_bm25_end - t_bm25_start:.2f}s")
        logger.debug(f"BM25 indexed {len(texts)} texts, k={bm25_retriever.k}")
        t_vec_retr_start = time.time()
        vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": parameters.VECTOR_Search_K_CHROMA,
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
        logger.info(f"[PROFILE] Total hybrid retriever build: {t_ensemble_end - t_bm25_start:.2f}s")
        return hybrid_retriever
