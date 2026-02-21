import core.llm_factory as llm_factory


class _CaptureOpenAI:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _CaptureAzure:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


def test_openai_gpt5_uses_max_completion_tokens(monkeypatch):
    monkeypatch.setattr(llm_factory, "ChatOpenAI", _CaptureOpenAI)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "OPENAI_API_KEY", "test-openai-key", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "LLM_MODEL_NAME", "gpt-5-nano", raising=False)

    llm_factory.get_chat_llm(
        role="router",
        model_name="gpt-5-nano",
        max_output_tokens=777,
    )

    kwargs = _CaptureOpenAI.last_kwargs
    assert kwargs["max_completion_tokens"] == 777
    assert "max_tokens" not in kwargs


def test_openai_non_gpt5_uses_max_tokens(monkeypatch):
    monkeypatch.setattr(llm_factory, "ChatOpenAI", _CaptureOpenAI)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "OPENAI_API_KEY", "test-openai-key", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "LLM_MODEL_NAME", "gpt-4o-mini", raising=False)

    llm_factory.get_chat_llm(
        role="research",
        model_name="gpt-4o-mini",
        max_output_tokens=333,
    )

    kwargs = _CaptureOpenAI.last_kwargs
    assert kwargs["max_tokens"] == 333
    assert "max_completion_tokens" not in kwargs


def test_azure_gpt5_uses_max_completion_tokens(monkeypatch):
    monkeypatch.setattr(llm_factory, "AzureChatOpenAI", _CaptureAzure)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "azure", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AZURE_OPENAI_API_KEY", "test-azure-key", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AZURE_OPENAI_API_VERSION", "2024-08-01-preview", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AZURE_OPENAI_DEPLOYMENT", "gpt5-deploy", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "LLM_MODEL_NAME", "gpt-5-nano", raising=False)

    llm_factory.get_chat_llm(
        role="verification",
        model_name="gpt-5-nano",
        max_output_tokens=555,
    )

    kwargs = _CaptureAzure.last_kwargs
    assert kwargs["max_completion_tokens"] == 555
    assert "max_tokens" not in kwargs
