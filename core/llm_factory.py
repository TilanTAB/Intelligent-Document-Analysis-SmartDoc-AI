import logging
import warnings
from typing import Optional, Type

from configuration.parameters import parameters

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None
try:
    from langchain_openai import AzureChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    AzureChatOpenAI = None

logger = logging.getLogger(__name__)


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _supports_temperature_override(provider: str, model: str) -> bool:
    """Some GPT-5 deployments reject explicit temperature overrides."""
    model_lower = (model or "").lower()
    if provider in {"openai", "azure"} and model_lower.startswith("gpt-5"):
        return False
    return True


def _resolved_temperature(provider: str, model: str, requested: Optional[float]) -> Optional[float]:
    if requested is None:
        return None
    if _supports_temperature_override(provider=provider, model=model):
        return requested
    if requested not in (1, 1.0):
        logger.info(
            f"Model {model} does not support explicit temperature={requested}; "
            "using provider default temperature."
        )
    # Omit temperature parameter for GPT-5 style deployments.
    return None


def _is_gpt5_model(provider: str, model: str) -> bool:
    return provider in {"openai", "azure"} and (model or "").lower().startswith("gpt-5")


def _gpt5_inference_profile(role: str) -> dict:
    """
    Conservative GPT-5 profile to avoid burning the completion budget on
    hidden reasoning tokens (which can return empty visible text).
    """
    role_lower = (role or "").lower()

    # Router/decomposition needs less depth than research/verification.
    if "router" in role_lower:
        return {"reasoning_effort": "low", "verbosity": "low"}

    # Default profile for research/relevance/verification.
    return {"reasoning_effort": "low", "verbosity": "low"}


def _model_max_tokens(provider: str, model: str) -> Optional[int]:
    """Known hard completion-token caps by provider/model family."""
    if _is_gpt5_model(provider=provider, model=model):
        # Azure/OpenAI GPT-5 deployments currently cap completion tokens at 128k.
        return 128_000
    return None


def _resolved_max_output_tokens(provider: str, model: str, requested: Optional[int]) -> Optional[int]:
    if requested is None:
        return None
    cap = _model_max_tokens(provider=provider, model=model)
    if cap is None:
        return requested
    if requested > cap:
        logger.warning(
            "Requested max_tokens=%d exceeds model limit for %s; clamping to %d.",
            requested,
            model,
            cap,
        )
        return cap
    return requested


def _apply_completion_token_limit(
    kwargs: dict,
    *,
    provider: str,
    model: str,
    resolved_max_tokens: Optional[int],
) -> None:
    """
    Apply provider/model-specific completion token parameter.

    GPT-5 style deployments reject `max_tokens` and require
    `max_completion_tokens`.
    """
    if resolved_max_tokens is None:
        return
    if _is_gpt5_model(provider=provider, model=model):
        kwargs["max_completion_tokens"] = resolved_max_tokens
        return
    kwargs["max_tokens"] = resolved_max_tokens


def _construct_chat_model(factory, **kwargs):
    """
    Construct a langchain-openai chat client while suppressing known warnings
    for GPT-5-only passthrough params (handled via model_kwargs internally).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*max_completion_tokens.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*verbosity.*",
            category=UserWarning,
        )
        return factory(**kwargs)


def get_chat_llm(
    role: str,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
):
    """
    Return a LangChain chat model configured per provider.

    Args:
        role: descriptive role for logging (research/verification/router/etc.)
        model_name: override model; defaults to parameters.LLM_MODEL_NAME
        temperature: sampling temperature
        max_output_tokens: provider-specific limit (mapped to max_tokens for OpenAI)
    """

    provider = parameters.LLM_PROVIDER.lower()
    model = model_name or parameters.LLM_MODEL_NAME
    resolved_max_tokens = _resolved_max_output_tokens(
        provider=provider,
        model=model,
        requested=max_output_tokens,
    )

    if provider == "google":
        _require(ChatGoogleGenerativeAI is not None, "langchain-google-genai is required for Google provider")
        _require(parameters.GOOGLE_API_KEY, "GOOGLE_API_KEY is required for Google provider")
        logger.debug(f"Using Google LLM for role={role}, model={model}")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=parameters.GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=resolved_max_tokens,
        )

    if provider == "openai":
        _require(ChatOpenAI is not None, "langchain-openai is required for OpenAI provider")
        _require(parameters.OPENAI_API_KEY, "OPENAI_API_KEY is required for OpenAI provider")
        logger.debug(f"Using OpenAI LLM for role={role}, model={model}")
        resolved_temperature = _resolved_temperature(provider=provider, model=model, requested=temperature)
        kwargs = {
            "model": model,
            "api_key": parameters.OPENAI_API_KEY,
        }
        _apply_completion_token_limit(
            kwargs,
            provider=provider,
            model=model,
            resolved_max_tokens=resolved_max_tokens,
        )
        if _is_gpt5_model(provider=provider, model=model):
            kwargs.update(_gpt5_inference_profile(role=role))
        if resolved_temperature is not None:
            kwargs["temperature"] = resolved_temperature
        return _construct_chat_model(ChatOpenAI, **kwargs)

    if provider == "azure":
        _require(AzureChatOpenAI is not None, "langchain-openai is required for Azure OpenAI provider")
        _require(parameters.AZURE_OPENAI_API_KEY, "AZURE_OPENAI_API_KEY is required for Azure provider")
        _require(parameters.AZURE_OPENAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT is required for Azure provider")
        _require(parameters.AZURE_OPENAI_DEPLOYMENT, "AZURE_OPENAI_DEPLOYMENT is required for Azure provider")
        logger.debug(f"Using Azure OpenAI LLM for role={role}, deployment={parameters.AZURE_OPENAI_DEPLOYMENT}")
        resolved_temperature = _resolved_temperature(provider=provider, model=model, requested=temperature)
        kwargs = {
            "api_key": parameters.AZURE_OPENAI_API_KEY,
            "azure_endpoint": parameters.AZURE_OPENAI_ENDPOINT,
            "api_version": parameters.AZURE_OPENAI_API_VERSION,
            "deployment_name": parameters.AZURE_OPENAI_DEPLOYMENT,
        }
        _apply_completion_token_limit(
            kwargs,
            provider=provider,
            model=model,
            resolved_max_tokens=resolved_max_tokens,
        )
        if _is_gpt5_model(provider=provider, model=model):
            kwargs.update(_gpt5_inference_profile(role=role))
        if resolved_temperature is not None:
            kwargs["temperature"] = resolved_temperature
        return _construct_chat_model(AzureChatOpenAI, **kwargs)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def get_structured_llm(
    schema: Type,
    role: str,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
):
    """Return an LLM wrapped with structured output for the given schema."""
    llm = get_chat_llm(role=role, model_name=model_name, temperature=temperature, max_output_tokens=max_output_tokens)
    try:
        return llm.with_structured_output(schema)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Structured output not supported natively; falling back to base LLM. Error: {e}")
        return llm
