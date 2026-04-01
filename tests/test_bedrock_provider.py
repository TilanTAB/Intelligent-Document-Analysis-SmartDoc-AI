"""
Unit tests for Amazon Bedrock provider integration across LLM, embedding, and vision factories.
Uses monkeypatching to avoid real AWS calls.
"""
import pytest
import warnings

import core.llm_factory as llm_factory
import core.embedding_factory as embedding_factory
import core.vision_client as vision_client


# ---------------------------------------------------------------------------
# Capture stubs
# ---------------------------------------------------------------------------

class _CaptureBedrock:
    """Captures kwargs passed to ChatBedrockConverse.__init__."""
    last_kwargs = None
    last_content = None  # set by .invoke() in vision tests

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        # Return a fake AIMessage-like object
        class _FakeResponse:
            content = "bedrock chart analysis result"
        return _FakeResponse()


class _CaptureBedrockEmbeddings:
    """Captures kwargs passed to BedrockEmbeddings.__init__."""
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def embed_documents(self, texts):
        return [[0.0] * 1024] * len(texts)

    def embed_query(self, text):
        return [0.0] * 1024


# ---------------------------------------------------------------------------
# LLM factory tests
# ---------------------------------------------------------------------------

def test_bedrock_llm_basic_routing(monkeypatch):
    """get_chat_llm with LLM_PROVIDER=bedrock instantiates ChatBedrockConverse with correct kwargs."""
    monkeypatch.setattr(llm_factory, "ChatBedrockConverse", _CaptureBedrock)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    llm_factory.get_chat_llm(role="research", temperature=0.1)

    kwargs = _CaptureBedrock.last_kwargs
    assert kwargs["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert kwargs["region_name"] == "us-east-1"
    assert kwargs["temperature"] == 0.1
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs


def test_bedrock_llm_with_explicit_credentials(monkeypatch):
    """Explicit AWS credentials are passed through to ChatBedrockConverse."""
    monkeypatch.setattr(llm_factory, "ChatBedrockConverse", _CaptureBedrock)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_BEDROCK_REGION", "eu-west-1", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_ACCESS_KEY_ID", "AKIATEST", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SECRET_ACCESS_KEY", "supersecret", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SESSION_TOKEN", "mytoken", raising=False)

    llm_factory.get_chat_llm(role="router")

    kwargs = _CaptureBedrock.last_kwargs
    assert kwargs["aws_access_key_id"] == "AKIATEST"
    assert kwargs["aws_secret_access_key"] == "supersecret"
    assert kwargs["aws_session_token"] == "mytoken"
    assert kwargs["region_name"] == "eu-west-1"


def test_bedrock_llm_max_tokens_passed(monkeypatch):
    """max_output_tokens is forwarded as max_tokens (not max_completion_tokens)."""
    monkeypatch.setattr(llm_factory, "ChatBedrockConverse", _CaptureBedrock)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    llm_factory.get_chat_llm(role="verification", max_output_tokens=2048)

    kwargs = _CaptureBedrock.last_kwargs
    assert kwargs["max_tokens"] == 2048
    assert "max_completion_tokens" not in kwargs


def test_bedrock_llm_non_bedrock_model_name_falls_back(monkeypatch):
    """
    If an agent-specific model override like 'gpt-4o-mini' is passed when
    provider=bedrock, the factory warns and substitutes BEDROCK_MODEL_ID.
    """
    monkeypatch.setattr(llm_factory, "ChatBedrockConverse", _CaptureBedrock)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    # The factory logs a WARNING (not a Python warning) and does not raise
    llm_factory.get_chat_llm(role="research", model_name="gpt-4o-mini")

    kwargs = _CaptureBedrock.last_kwargs
    # Should have fallen back to BEDROCK_MODEL_ID
    assert kwargs["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_bedrock_llm_explicit_bedrock_model_name_is_used(monkeypatch):
    """A valid Bedrock model ID passed as model_name is used as-is."""
    monkeypatch.setattr(llm_factory, "ChatBedrockConverse", _CaptureBedrock)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    llm_factory.get_chat_llm(role="research", model_name="anthropic.claude-3-haiku-20240307-v1:0")

    kwargs = _CaptureBedrock.last_kwargs
    assert kwargs["model"] == "anthropic.claude-3-haiku-20240307-v1:0"


def test_bedrock_llm_missing_package_raises(monkeypatch):
    """If langchain-aws is not installed, get_chat_llm raises ValueError."""
    monkeypatch.setattr(llm_factory, "ChatBedrockConverse", None)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    with pytest.raises(ValueError, match="langchain-aws"):
        llm_factory.get_chat_llm(role="router")


def test_bedrock_structured_llm_calls_with_structured_output(monkeypatch):
    """get_structured_llm wraps the Bedrock LLM with .with_structured_output()."""
    monkeypatch.setattr(llm_factory, "ChatBedrockConverse", _CaptureBedrock)
    monkeypatch.setattr(llm_factory.parameters, "LLM_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(llm_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    from pydantic import BaseModel

    class _Schema(BaseModel):
        answer: str

    result = llm_factory.get_structured_llm(schema=_Schema, role="verification")
    # _CaptureBedrock.with_structured_output returns self, so result should be a _CaptureBedrock
    assert isinstance(result, _CaptureBedrock)


# ---------------------------------------------------------------------------
# Embedding factory tests
# ---------------------------------------------------------------------------

def test_bedrock_embeddings_routing(monkeypatch):
    """get_embeddings with EMBEDDING_PROVIDER=bedrock instantiates BedrockEmbeddings."""
    monkeypatch.setattr(embedding_factory, "BedrockEmbeddings", _CaptureBedrockEmbeddings)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_CACHE_ENABLED", False, raising=False)

    result = embedding_factory.get_embeddings()

    kwargs = _CaptureBedrockEmbeddings.last_kwargs
    assert kwargs["model_id"] == "amazon.titan-embed-text-v2:0"
    assert kwargs["region_name"] == "us-east-1"
    assert "aws_access_key_id" not in kwargs


def test_bedrock_embeddings_with_credentials(monkeypatch):
    """Explicit credentials are forwarded to BedrockEmbeddings."""
    monkeypatch.setattr(embedding_factory, "BedrockEmbeddings", _CaptureBedrockEmbeddings)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_BEDROCK_REGION", "us-west-2", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_ACCESS_KEY_ID", "AKIAEMBED", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_SECRET_ACCESS_KEY", "embedsecret", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_CACHE_ENABLED", False, raising=False)

    embedding_factory.get_embeddings()

    kwargs = _CaptureBedrockEmbeddings.last_kwargs
    assert kwargs["aws_access_key_id"] == "AKIAEMBED"
    assert kwargs["aws_secret_access_key"] == "embedsecret"
    assert kwargs["region_name"] == "us-west-2"


def test_bedrock_embeddings_custom_model_name(monkeypatch):
    """Passing model_name overrides the default BEDROCK_EMBEDDING_MODEL_ID."""
    monkeypatch.setattr(embedding_factory, "BedrockEmbeddings", _CaptureBedrockEmbeddings)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "AWS_SESSION_TOKEN", None, raising=False)
    monkeypatch.setattr(embedding_factory.parameters, "EMBEDDING_CACHE_ENABLED", False, raising=False)

    embedding_factory.get_embeddings(model_name="cohere.embed-english-v3")

    kwargs = _CaptureBedrockEmbeddings.last_kwargs
    assert kwargs["model_id"] == "cohere.embed-english-v3"


# ---------------------------------------------------------------------------
# Vision client tests
# ---------------------------------------------------------------------------

def test_bedrock_vision_client_creation(monkeypatch):
    """get_vision_client with VISION_PROVIDER=bedrock returns a ChatBedrockConverse instance."""
    monkeypatch.setattr(vision_client.parameters, "VISION_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(vision_client.parameters, "BEDROCK_VISION_MODEL_ID", None, raising=False)
    monkeypatch.setattr(vision_client.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(vision_client.parameters, "CHART_MAX_TOKENS", 1500, raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    # Patch the lazy import inside get_vision_client
    import sys
    import types
    fake_langchain_aws = types.ModuleType("langchain_aws")
    fake_langchain_aws.ChatBedrockConverse = _CaptureBedrock
    monkeypatch.setitem(sys.modules, "langchain_aws", fake_langchain_aws)

    client = vision_client.get_vision_client()
    assert isinstance(client, _CaptureBedrock)
    kwargs = _CaptureBedrock.last_kwargs
    assert kwargs["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert kwargs["region_name"] == "us-east-1"
    assert kwargs["max_tokens"] == 1500


def test_bedrock_vision_model_id_override(monkeypatch):
    """BEDROCK_VISION_MODEL_ID takes precedence over BEDROCK_MODEL_ID when set."""
    monkeypatch.setattr(vision_client.parameters, "VISION_PROVIDER", "bedrock", raising=False)
    monkeypatch.setattr(vision_client.parameters, "BEDROCK_VISION_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0", raising=False)
    monkeypatch.setattr(vision_client.parameters, "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0", raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_BEDROCK_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(vision_client.parameters, "CHART_MAX_TOKENS", 1500, raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_ACCESS_KEY_ID", None, raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_SECRET_ACCESS_KEY", None, raising=False)
    monkeypatch.setattr(vision_client.parameters, "AWS_SESSION_TOKEN", None, raising=False)

    import sys, types
    fake_langchain_aws = types.ModuleType("langchain_aws")
    fake_langchain_aws.ChatBedrockConverse = _CaptureBedrock
    monkeypatch.setitem(sys.modules, "langchain_aws", fake_langchain_aws)

    vision_client.get_vision_client()
    assert _CaptureBedrock.last_kwargs["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_bedrock_analyze_chart_images(monkeypatch, tmp_path):
    """analyze_chart_images with bedrock constructs a multimodal HumanMessage and returns content."""
    # Create a fake image file
    img_path = tmp_path / "chart.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # fake JPEG bytes

    fake_client = _CaptureBedrock()

    # Patch langchain_core.messages.HumanMessage
    import sys, types
    fake_core_messages = types.ModuleType("langchain_core.messages")

    captured_messages = []

    class _FakeHumanMessage:
        def __init__(self, content):
            self.content = content
            captured_messages.append(self)

    fake_core_messages.HumanMessage = _FakeHumanMessage
    monkeypatch.setitem(sys.modules, "langchain_core.messages", fake_core_messages)

    result = vision_client.analyze_chart_images(
        client=fake_client,
        image_paths=[str(img_path)],
        prompt="Describe this chart.",
        model_name="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        max_output_tokens=1500,
        provider="bedrock",
    )

    assert result == "bedrock chart analysis result"
    assert len(captured_messages) == 1
    content = captured_messages[0].content
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Describe this chart."
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# Parameter validator tests
# ---------------------------------------------------------------------------

def test_parameters_accept_bedrock_llm_provider():
    """'bedrock' is a valid LLM_PROVIDER value."""
    from configuration.parameters import Settings
    s = Settings(
        LLM_PROVIDER="bedrock",
        EMBEDDING_PROVIDER="openai",
        VISION_PROVIDER="none",
        OPENAI_API_KEY="sk-test",
    )
    assert s.LLM_PROVIDER == "bedrock"


def test_parameters_accept_bedrock_embedding_provider():
    """'bedrock' is a valid EMBEDDING_PROVIDER value."""
    from configuration.parameters import Settings
    s = Settings(
        LLM_PROVIDER="openai",
        EMBEDDING_PROVIDER="bedrock",
        VISION_PROVIDER="none",
        OPENAI_API_KEY="sk-test",
    )
    assert s.EMBEDDING_PROVIDER == "bedrock"


def test_parameters_accept_bedrock_vision_provider():
    """'bedrock' is a valid VISION_PROVIDER value."""
    from configuration.parameters import Settings
    s = Settings(
        LLM_PROVIDER="openai",
        EMBEDDING_PROVIDER="openai",
        VISION_PROVIDER="bedrock",
        OPENAI_API_KEY="sk-test",
    )
    assert s.VISION_PROVIDER == "bedrock"


def test_parameters_bedrock_defaults():
    """Bedrock config fields have expected defaults."""
    from configuration.parameters import Settings
    s = Settings(
        LLM_PROVIDER="openai",
        EMBEDDING_PROVIDER="openai",
        VISION_PROVIDER="none",
        OPENAI_API_KEY="sk-test",
    )
    assert s.AWS_BEDROCK_REGION == "us-east-1"
    assert s.BEDROCK_MODEL_ID == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert s.BEDROCK_EMBEDDING_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert s.BEDROCK_VISION_MODEL_ID is None
    assert s.AWS_ACCESS_KEY_ID is None


def test_parameters_bedrock_credential_warning(recwarn):
    """When provider=bedrock with no AWS_ACCESS_KEY_ID, a warning is emitted (not ValueError)."""
    from configuration.parameters import Settings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = Settings(
            LLM_PROVIDER="bedrock",
            EMBEDDING_PROVIDER="openai",
            VISION_PROVIDER="none",
            OPENAI_API_KEY="sk-test",
        )
    bedrock_warnings = [w for w in caught if "AWS_ACCESS_KEY_ID" in str(w.message)]
    assert len(bedrock_warnings) == 1, "Expected exactly one Bedrock credential warning"
