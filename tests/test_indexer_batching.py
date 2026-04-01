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
        self.added_batches = []
        self.deleted_ids = []
        self._client = type("Client", (), {"max_batch_size": 4})()

    def add_documents(self, docs, ids=None):
        self.added_batches.append(len(docs))

    def delete(self, ids):
        self.deleted_ids.extend(ids)

    def as_retriever(self, **_kwargs):
        return FakeVectorRetriever()


def test_indexer_respects_configured_and_client_limited_batch_size(monkeypatch, tmp_path):
    created = []

    def fake_chroma(*args, **kwargs):
        store = FakeChroma(*args, **kwargs)
        created.append(store)
        return store

    monkeypatch.setattr(indexer, "Chroma", fake_chroma)
    monkeypatch.setattr(indexer, "get_embeddings", lambda: object())
    monkeypatch.setattr(indexer.BM25Retriever, "from_texts", staticmethod(lambda texts, metadatas: FakeBM25Retriever()))
    monkeypatch.setattr(indexer.parameters, "CHROMA_DB_PATH", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_INGEST_BATCH_SIZE", 6, raising=False)

    builder = indexer.RetrieverBuilder()
    docs = [
        Document(
            page_content=f"content {i}",
            metadata={"source": "s", "page": i, "chunk_id": f"c{i}", "type": "text"},
        )
        for i in range(9)
    ]

    builder.build_hybrid_retriever(docs, session_id="batching")

    assert created, "Chroma should be instantiated"
    added_batches = created[0].added_batches
    assert added_batches == [4, 4, 1]
    assert builder.last_build_profile["vector_ingest_batch_size"] == 4
    assert builder.last_build_profile["vector_ingest_batches"] == 3


def test_trim_docs_for_indexing_is_uncapped_when_max_index_chunks_is_zero(monkeypatch):
    monkeypatch.setattr(indexer, "get_embeddings", lambda: object())
    monkeypatch.setattr(indexer.parameters, "MAX_INDEX_CHUNKS", 0, raising=False)

    builder = indexer.RetrieverBuilder()
    docs = [
        Document(
            page_content=f"content {i}",
            metadata={"source": "s", "page": i, "chunk_id": f"c{i}", "type": "text"},
        )
        for i in range(5)
    ]

    trimmed = builder._trim_docs_for_indexing(docs)

    assert trimmed == docs


def test_ingest_streaming_flushes_per_batch(monkeypatch, tmp_path):
    """Verify ingest_streaming calls add_documents once per batch_size
    (flush-as-you-go), NOT once after draining the entire queue."""
    import queue

    created = []

    def fake_chroma(*args, **kwargs):
        store = FakeChroma(*args, **kwargs)
        created.append(store)
        return store

    monkeypatch.setattr(indexer, "Chroma", fake_chroma)
    monkeypatch.setattr(indexer, "get_embeddings", lambda: object())
    monkeypatch.setattr(indexer.parameters, "CHROMA_DB_PATH", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(indexer.parameters, "CHROMA_INGEST_BATCH_SIZE", 3, raising=False)

    builder = indexer.RetrieverBuilder()

    # Simulate a producer pushing 10 chunks then a None sentinel
    q = queue.Queue()
    for i in range(10):
        q.put(Document(
            page_content=f"streaming content {i}",
            metadata={"source": "s", "page": i, "chunk_id": f"sc{i}", "type": "text"},
        ))
    q.put(None)  # sentinel

    collected, vstore = builder.ingest_streaming(
        chunk_queue=q,
        chroma_dir=str(tmp_path / "chroma"),
    )

    assert len(collected) == 10
    assert created, "Chroma should be instantiated"
    # client max_batch_size=4, configured=3 → effective batch_size=3
    # 10 chunks / 3 per batch = 3 full batches + 1 tail batch of 1
    added_batches = created[0].added_batches
    assert added_batches == [3, 3, 3, 1], f"Expected [3,3,3,1] but got {added_batches}"
