from types import SimpleNamespace
import re

import core.embedding_factory as embedding_factory


def test_cache_namespace_is_localfilestore_safe():
    namespace = embedding_factory._cache_namespace(
        provider="az:ure openai",
        model_name="text-embedding-3-small:prod/v1",
    )

    assert namespace == "smartdoc_embeddings/az-ure-openai/text-embedding-3-small-prod/v1/"
    assert ":" not in namespace
    assert re.match(r"^[a-zA-Z0-9_.\-/]+$", namespace)


def test_embedding_cache_reuses_cached_vectors(monkeypatch, tmp_path):
    class FakeOpenAIEmbeddings:
        def __init__(self, **_kwargs):
            self.embed_calls = 0

        def embed_documents(self, texts):
            self.embed_calls += 1
            return [[float(len(t))] for t in texts]

        def embed_query(self, text):
            return [float(len(text))]

    class FakeCacheBackedEmbeddings:
        def __init__(self, underlying_embeddings):
            self.underlying = underlying_embeddings
            self.cache = {}

        @classmethod
        def from_bytes_store(
            cls,
            underlying_embeddings=None,
            document_embedding_cache=None,
            namespace=None,
            batch_size=None,
            *args,
            **kwargs,
        ):
            # Support both old and new signatures.
            underlying = underlying_embeddings or (args[0] if args else kwargs.get("embeddings"))
            return cls(underlying)

        def embed_documents(self, texts):
            missing = [t for t in texts if t not in self.cache]
            if missing:
                embeds = self.underlying.embed_documents(missing)
                for text, embedding in zip(missing, embeds):
                    self.cache[text] = embedding
            return [self.cache[t] for t in texts]

        def embed_query(self, text):
            if text not in self.cache:
                self.cache[text] = self.underlying.embed_query(text)
            return self.cache[text]

    class FakeLocalFileStore:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(embedding_factory, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    monkeypatch.setattr(embedding_factory, "CacheBackedEmbeddings", FakeCacheBackedEmbeddings)
    monkeypatch.setattr(embedding_factory, "LocalFileStore", FakeLocalFileStore)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "OPENAI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "OPENAI_EMBEDDING_MODEL_NAME", "test-model", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_CACHE_ENABLED", True, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_CACHE_DIR", str(tmp_path / "emb_cache"), raising=False)

    embeddings = embedding_factory.get_embeddings()
    assert isinstance(embeddings, FakeCacheBackedEmbeddings)

    embeddings.embed_documents(["alpha", "beta"])
    first_calls = embeddings.underlying.embed_calls
    embeddings.embed_documents(["alpha", "beta"])

    assert first_calls == 1
    assert embeddings.underlying.embed_calls == 1


def test_azure_embeddings_use_dedicated_endpoint(monkeypatch):
    class FakeAzureOpenAIEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_documents(self, texts):
            return [[float(len(t))] for t in texts]

        def embed_query(self, text):
            return [float(len(text))]

    monkeypatch.setattr(embedding_factory, "AzureOpenAIEmbeddings", FakeAzureOpenAIEmbeddings)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_PROVIDER", "azure", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_OPENAI_API_KEY", "chat-key", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_EMBEDDING_API_KEY", "embedding-key", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_OPENAI_API_VERSION", "2024-08-01-preview", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_OPENAI_ENDPOINT", "https://chat-resource.openai.azure.com/", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_EMBEDDING_ENDPOINT", "https://embedding-resource.openai.azure.com/", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_EMBEDDING_DEPLOYMENT", "embed-deployment", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_CACHE_ENABLED", False, raising=False)

    embeddings = embedding_factory.get_embeddings()
    assert isinstance(embeddings, FakeAzureOpenAIEmbeddings)
    assert embeddings.kwargs["api_key"] == "embedding-key"
    assert embeddings.kwargs["azure_endpoint"] == "https://embedding-resource.openai.azure.com/"
    assert embeddings.kwargs["deployment"] == "embed-deployment"


def test_azure_embeddings_fallback_to_chat_key(monkeypatch):
    class FakeAzureOpenAIEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_documents(self, texts):
            return [[float(len(t))] for t in texts]

        def embed_query(self, text):
            return [float(len(text))]

    monkeypatch.setattr(embedding_factory, "AzureOpenAIEmbeddings", FakeAzureOpenAIEmbeddings)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_PROVIDER", "azure", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_OPENAI_API_KEY", "chat-key", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_EMBEDDING_API_KEY", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_OPENAI_API_VERSION", "2024-08-01-preview", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_OPENAI_ENDPOINT", "https://chat-resource.openai.azure.com/", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_EMBEDDING_ENDPOINT", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AZURE_EMBEDDING_DEPLOYMENT", "embed-deployment", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_CACHE_ENABLED", False, raising=False)

    embeddings = embedding_factory.get_embeddings()
    assert isinstance(embeddings, FakeAzureOpenAIEmbeddings)
    assert embeddings.kwargs["api_key"] == "chat-key"
    assert embeddings.kwargs["azure_endpoint"] == "https://chat-resource.openai.azure.com/"
