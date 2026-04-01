import base64
import logging
import re
from pathlib import Path
from typing import List, Optional

from configuration.parameters import parameters

logger = logging.getLogger(__name__)


class VisionClientUnavailable(Exception):
    pass


def _image_to_data_url(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    try:
        parts = []
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                item_text = getattr(content_item, "text", None)
                if item_text:
                    parts.append(item_text)
        return "\n".join(parts).strip()
    except Exception:
        return str(response)


def _extract_chat_text(chat_response) -> str:
    try:
        choice = chat_response.choices[0]
        content = choice.message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        texts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if text:
                        texts.append(text)
            return "\n".join(texts).strip()
    except Exception:
        pass
    return str(chat_response)


def _azure_supports_responses_api(api_version: Optional[str]) -> bool:
    """
    Azure Responses API is available only on 2025-03-01-preview and later.
    """
    if not api_version:
        return False
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", api_version)
    if not match:
        return False
    year, month, day = map(int, match.groups())
    return (year, month, day) >= (2025, 3, 1)


def get_vision_client():
    """Return a provider-specific vision client or raise if unavailable."""
    provider = parameters.VISION_PROVIDER.lower()
    if provider == "none":
        raise VisionClientUnavailable("Vision provider disabled")

    if provider == "google":
        try:
            import google.genai as genai
        except ImportError as e:  # pragma: no cover
            raise VisionClientUnavailable("google-genai package not installed") from e
        if not parameters.GOOGLE_API_KEY:
            raise VisionClientUnavailable("GOOGLE_API_KEY is required for Google vision")
        logger.debug("Initializing Google Gemini vision client")
        return genai.Client(api_key=parameters.GOOGLE_API_KEY)

    if provider == "azure":
        try:
            from openai import AzureOpenAI
        except ImportError as e:  # pragma: no cover
            raise VisionClientUnavailable("openai package is required for Azure vision") from e
        if not parameters.AZURE_OPENAI_API_KEY:
            raise VisionClientUnavailable("AZURE_OPENAI_API_KEY is required for Azure vision")
        if not parameters.AZURE_OPENAI_ENDPOINT:
            raise VisionClientUnavailable("AZURE_OPENAI_ENDPOINT is required for Azure vision")
        logger.debug("Initializing Azure OpenAI vision client")
        return AzureOpenAI(
            api_key=parameters.AZURE_OPENAI_API_KEY,
            api_version=parameters.AZURE_OPENAI_API_VERSION,
            azure_endpoint=parameters.AZURE_OPENAI_ENDPOINT,
        )

    if provider == "bedrock":
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as e:
            raise VisionClientUnavailable("langchain-aws is required for Bedrock vision") from e
        logger.debug("Initializing Bedrock vision client via ChatBedrockConverse")
        kwargs: dict = {
            "model": parameters.BEDROCK_VISION_MODEL_ID or parameters.BEDROCK_MODEL_ID,
            "region_name": parameters.AWS_BEDROCK_REGION,
            "max_tokens": parameters.CHART_MAX_TOKENS,
        }
        if parameters.AWS_ACCESS_KEY_ID:
            kwargs["aws_access_key_id"] = parameters.AWS_ACCESS_KEY_ID
        if parameters.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_secret_access_key"] = parameters.AWS_SECRET_ACCESS_KEY
        if parameters.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = parameters.AWS_SESSION_TOKEN
        return ChatBedrockConverse(**kwargs)

    raise VisionClientUnavailable(f"Unsupported VISION_PROVIDER: {provider}")


def analyze_chart_images(
    client,
    image_paths: List[str],
    prompt: str,
    model_name: str,
    max_output_tokens: int,
    provider: Optional[str] = None,
) -> str:
    """Analyze one or more chart images and return text output."""
    current_provider = (provider or parameters.VISION_PROVIDER).lower()

    if current_provider == "google":
        from PIL import Image
        try:
            from google.genai import types
        except Exception as e:  # pragma: no cover
            raise VisionClientUnavailable("google.genai types unavailable") from e

        images = [Image.open(path) for path in image_paths]
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt] + images,
                config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
            )
            return getattr(response, "text", None) or str(response)
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass

    if current_provider == "azure":
        data_urls = [_image_to_data_url(path) for path in image_paths]
        if _azure_supports_responses_api(parameters.AZURE_OPENAI_API_VERSION):
            response_input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        *[
                            {"type": "input_image", "image_url": url}
                            for url in data_urls
                        ],
                    ],
                }
            ]
            try:
                response = client.responses.create(
                    model=model_name,
                    input=response_input,
                    max_output_tokens=max_output_tokens,
                )
                text = _extract_response_text(response)
                if text:
                    return text
                raise RuntimeError("Azure responses API returned empty content")
            except Exception as response_error:
                logger.warning(
                    f"Azure responses API vision call failed ({type(response_error).__name__}); "
                    "falling back to chat.completions."
                )
                chat_kwargs = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                *[
                                    {"type": "image_url", "image_url": {"url": url}}
                                    for url in data_urls
                                ],
                            ],
                        }
                    ],
                    "max_completion_tokens": max_output_tokens,
                }
                if model_name.lower().startswith("gpt-5"):
                    chat_kwargs["reasoning_effort"] = "low"
                    chat_kwargs["verbosity"] = "low"
                chat_response = client.chat.completions.create(**chat_kwargs)
                return _extract_chat_text(chat_response)

        chat_kwargs = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {"type": "image_url", "image_url": {"url": url}}
                            for url in data_urls
                        ],
                    ],
                }
            ],
            "max_completion_tokens": max_output_tokens,
        }
        if model_name.lower().startswith("gpt-5"):
            chat_kwargs["reasoning_effort"] = "low"
            chat_kwargs["verbosity"] = "low"
        chat_response = client.chat.completions.create(**chat_kwargs)
        return _extract_chat_text(chat_response)

    if current_provider == "bedrock":
        from langchain_core.messages import HumanMessage
        content_parts: list = [{"type": "text", "text": prompt}]
        for path in image_paths:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        response = client.invoke([HumanMessage(content=content_parts)])
        return response.content

    raise VisionClientUnavailable(f"Unsupported VISION_PROVIDER: {current_provider}")
