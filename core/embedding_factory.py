import logging
from typing import Optional
from pathlib import Path
import re

from configuration.parameters import parameters

# Keep provider classes patchable in tests while still avoiding heavy import
# side effects at module import time.
GoogleGenerativeAIEmbeddings = None
OpenAIEmbeddings = None
AzureOpenAIEmbeddings = None
BedrockEmbeddings = None

try:
    # LangChain v1+ split: cache utilities moved to langchain_classic.
    from langchain_classic.embeddings import CacheBackedEmbeddings
except ImportError:  # pragma: no cover
    try:
        from langchain.embeddings import CacheBackedEmbeddings
    except ImportError:  # pragma: no cover
        try:
            from langchain.embeddings.cache import CacheBackedEmbeddings
        except ImportError:  # pragma: no cover
            CacheBackedEmbeddings = None
try:
    # LangChain v1+ split: storage moved to langchain_classic.
    from langchain_classic.storage import LocalFileStore
except ImportError:  # pragma: no cover
    try:
        from langchain.storage import LocalFileStore
    except ImportError:  # pragma: no cover
        try:
            # Legacy/alternate package path (kept as last-resort fallback).
            from langchain_community.storage import LocalFileStore
        except ImportError:  # pragma: no cover
            LocalFileStore = None

logger = logging.getLogger(__name__)
_CACHE_COMPONENT_SANITIZER = re.compile(r"[^a-zA-Z0-9_.\-/]+")


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _sanitize_cache_component(value: str) -> str:
    """Return a LocalFileStore-safe cache key component."""
    sanitized = _CACHE_COMPONENT_SANITIZER.sub("-", (value or "").strip())
    sanitized = sanitized.strip("./-")
    return sanitized or "unknown"


def _cache_namespace(provider: str, model_name: str) -> str:
    safe_provider = _sanitize_cache_component(provider.lower())
    safe_model = _sanitize_cache_component(model_name.lower())
    # LocalFileStore keys may only use [a-zA-Z0-9_.\\-/], so avoid separators like ":".
    return f"smartdoc_embeddings/{safe_provider}/{safe_model}/"


def _wrap_with_embedding_cache(embeddings, provider: str, model_name: str):
    if not parameters.EMBEDDING_CACHE_ENABLED:
        return embeddings
    if CacheBackedEmbeddings is None or LocalFileStore is None:
        logger.warning("Embedding cache is enabled but cache dependencies are unavailable; using uncached embeddings.")
        return embeddings

    cache_path = Path(parameters.EMBEDDING_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    store = LocalFileStore(str(cache_path))
    namespace = _cache_namespace(provider=provider, model_name=model_name)
    logger.info(f"[PROFILE] stage=embeddings.cache status=enabled namespace={namespace}")
    try:
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=store,
            namespace=namespace,
            batch_size=parameters.EMBEDDING_CACHE_BATCH_SIZE,
        )
    except TypeError:
        # Backward compatibility for older LangChain signatures.
        return CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            store,
            namespace=namespace,
        )


def get_embeddings(model_name: Optional[str] = None):
    """Return an embeddings model based on configuration."""
    global GoogleGenerativeAIEmbeddings, OpenAIEmbeddings, AzureOpenAIEmbeddings, BedrockEmbeddings
    provider = parameters.EMBEDDING_PROVIDER.lower()
    if provider == "google":
        if GoogleGenerativeAIEmbeddings is None:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings as _GoogleEmbeddings
            except ImportError as e:  # pragma: no cover
                raise ValueError("langchain-google-genai is required for Google embeddings") from e
            GoogleGenerativeAIEmbeddings = _GoogleEmbeddings
        _require(parameters.GOOGLE_API_KEY, "GOOGLE_API_KEY is required for Google embeddings")
        name = model_name or parameters.EMBEDDING_MODEL_NAME
        logger.debug(f"Using Google embeddings model={name}")
        base = GoogleGenerativeAIEmbeddings(
            model=name,
            google_api_key=parameters.GOOGLE_API_KEY,
            batch_size=100,
        )
        return _wrap_with_embedding_cache(base, provider=provider, model_name=name)

    if provider == "openai":
        if OpenAIEmbeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddings
            except ImportError as e:  # pragma: no cover
                raise ValueError("langchain-openai is required for OpenAI embeddings") from e
            OpenAIEmbeddings = _OpenAIEmbeddings
        _require(parameters.OPENAI_API_KEY, "OPENAI_API_KEY is required for OpenAI embeddings")
        name = model_name or parameters.OPENAI_EMBEDDING_MODEL_NAME
        logger.debug(f"Using OpenAI embeddings model={name}")
        base = OpenAIEmbeddings(
            model=name,
            api_key=parameters.OPENAI_API_KEY,
        )
        return _wrap_with_embedding_cache(base, provider=provider, model_name=name)

    if provider == "azure":
        if AzureOpenAIEmbeddings is None:
            try:
                from langchain_openai import AzureOpenAIEmbeddings as _AzureOpenAIEmbeddings
            except ImportError as e:  # pragma: no cover
                raise ValueError("langchain-openai is required for Azure OpenAI embeddings") from e
            AzureOpenAIEmbeddings = _AzureOpenAIEmbeddings
        embedding_api_key = (parameters.AZURE_EMBEDDING_API_KEY or parameters.AZURE_OPENAI_API_KEY or "").strip()
        _require(
            bool(embedding_api_key),
            "Azure embeddings require AZURE_EMBEDDING_API_KEY or AZURE_OPENAI_API_KEY.",
        )
        embedding_endpoint = (parameters.AZURE_EMBEDDING_ENDPOINT or parameters.AZURE_OPENAI_ENDPOINT or "").strip()
        _require(
            bool(embedding_endpoint),
            "Azure embeddings require AZURE_EMBEDDING_ENDPOINT or AZURE_OPENAI_ENDPOINT.",
        )
        deployment = parameters.AZURE_EMBEDDING_DEPLOYMENT or model_name
        _require(deployment, "AZURE_EMBEDDING_DEPLOYMENT (or model_name) is required for Azure embeddings")
        logger.debug(
            "Using Azure OpenAI embeddings deployment=%s endpoint=%s",
            deployment,
            embedding_endpoint,
        )
        base = AzureOpenAIEmbeddings(
            api_key=embedding_api_key,
            azure_endpoint=embedding_endpoint,
            api_version=parameters.AZURE_OPENAI_API_VERSION,
            deployment=deployment,
        )
        return _wrap_with_embedding_cache(base, provider=provider, model_name=deployment)

    if provider == "bedrock":
        if BedrockEmbeddings is None:
            try:
                from langchain_aws import BedrockEmbeddings as _BedrockEmbeddings
            except ImportError as e:
                raise ValueError(
                    "langchain-aws is required for Bedrock embeddings. Install with: pip install langchain-aws boto3"
                ) from e
            BedrockEmbeddings = _BedrockEmbeddings
        name = model_name or parameters.BEDROCK_EMBEDDING_MODEL_ID
        logger.debug("Using Bedrock embeddings model=%s", name)
        creds: dict = {}
        if parameters.AWS_ACCESS_KEY_ID:
            creds["credentials_profile_name"] = None
            creds["aws_access_key_id"] = parameters.AWS_ACCESS_KEY_ID
        if parameters.AWS_SECRET_ACCESS_KEY:
            creds["aws_secret_access_key"] = parameters.AWS_SECRET_ACCESS_KEY
        if parameters.AWS_SESSION_TOKEN:
            creds["aws_session_token"] = parameters.AWS_SESSION_TOKEN
        base = BedrockEmbeddings(
            model_id=name,
            region_name=parameters.AWS_BEDROCK_REGION,
            **creds,
        )
        return _wrap_with_embedding_cache(base, provider=provider, model_name=name)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
