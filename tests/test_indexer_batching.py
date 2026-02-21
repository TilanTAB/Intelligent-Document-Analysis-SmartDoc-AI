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
