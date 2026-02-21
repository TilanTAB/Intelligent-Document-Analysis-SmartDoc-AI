from configuration.parameters import Settings


def test_settings_accepts_separate_embedding_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "test-embedding-key")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", "test-embedding-deployment")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://embedding-resource.openai.azure.com/")

    settings = Settings()
    assert settings.AZURE_OPENAI_ENDPOINT in (None, "")
    assert settings.AZURE_EMBEDDING_API_KEY == "test-embedding-key"
    assert settings.AZURE_EMBEDDING_ENDPOINT == "https://embedding-resource.openai.azure.com/"


def test_settings_rejects_azure_embeddings_without_any_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_EMBEDDING_API_KEY", "test-embedding-key")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", "test-embedding-deployment")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_EMBEDDING_ENDPOINT", raising=False)

    try:
        Settings()
    except Exception as exc:  # pydantic wraps model-level validation
        assert "AZURE_EMBEDDING_ENDPOINT or AZURE_OPENAI_ENDPOINT" in str(exc)
    else:
        raise AssertionError("Expected Settings() to fail when Azure embedding endpoints are missing")


def test_settings_rejects_azure_embeddings_without_any_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_EMBEDDING_DEPLOYMENT", "test-embedding-deployment")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_EMBEDDING_ENDPOINT", "https://embedding-resource.openai.azure.com/")

    try:
        Settings()
    except Exception as exc:
        assert "AZURE_EMBEDDING_API_KEY or AZURE_OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected Settings() to fail when Azure embedding keys are missing")

