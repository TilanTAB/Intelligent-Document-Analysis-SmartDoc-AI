from langchain_core.documents import Document

import search_engine.indexer as indexer


class FakeVectorRetriever:
    def invoke(self, _query):
        return []


class FakeBM25Retriever:
    def __init__(self):
        self.k = 0

    def invoke(self, _query):
        return []


class FakeChroma:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def add_documents(self, docs, ids=None):
        return None

    def delete(self, ids):
        return None

    def as_retriever(self, **_kwargs):
        return FakeVectorRetriever()


def test_hnsw_collection_metadata_is_applied(monkeypatch, tmp_path):
    created_kwargs = []

    def fake_chroma(*args, **kwargs):
        created_kwargs.append(kwargs)
        return FakeChroma(*args, **kwargs)

    monkeypatch.setattr(indexer, "Chroma", fake_chroma)
    monkeypatch.setattr(indexer, "get_embeddings", lambda: object())
    monkeypatch.setattr(indexer.BM25Retriever, "from_texts", staticmethod(lambda texts, metadatas: FakeBM25Retriever()))

    monkeypatch.setattr(indexer.parameters, "CHROMA_DB_PATH", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_COLLECTION_NAME", "documents_perf", raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_HNSW_M", 24, raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_HNSW_CONSTRUCTION_EF", 120, raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_HNSW_SEARCH_EF", 70, raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_HNSW_NUM_THREADS", 6, raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_HNSW_BATCH_SIZE", 200, raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_HNSW_SYNC_THRESHOLD", 1500, raising=False)

    builder = indexer.RetrieverBuilder()
    docs = [
        Document(
            page_content="content",
            metadata={"source": "s", "page": 1, "chunk_id": "c1", "type": "text"},
        )
    ]

    builder.build_hybrid_retriever(docs, session_id="hnsw")

    assert created_kwargs
    metadata = created_kwargs[0]["collection_metadata"]
    assert metadata["hnsw:M"] == 24
    assert metadata["hnsw:construction_ef"] == 120
    assert metadata["hnsw:search_ef"] == 70
    assert metadata["hnsw:num_threads"] == 6
    assert metadata["hnsw:batch_size"] == 200
    assert metadata["hnsw:sync_threshold"] == 1500
