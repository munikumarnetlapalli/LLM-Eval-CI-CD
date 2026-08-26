"""Provider factory — builds the correct LLMProvider from config."""
from __future__ import annotations

from typing import Any


def build_provider(config: dict[str, Any]):
    """Build an LLMProvider from a configuration dictionary.

    Expected config shape:
        provider: mock | openai | azure_openai | gemini
        model: <model name>
        api_key: <key>        (not for mock)
        endpoint: <url>       (azure_openai only)
        deployment: <name>    (azure_openai only)
        api_version: <ver>    (azure_openai only)
    """
    from .mock import MockProvider
    from .openai_provider import OpenAIProvider
    from .azure_openai import AzureOpenAIProvider
    from .gemini import GeminiProvider

    provider_type = config.get("provider", "mock").lower()

    if provider_type == "mock":
        return MockProvider(
            response_text=config.get("mock_response", MockProvider.DEFAULT_RESPONSE),
        )
    elif provider_type == "openai":
        return OpenAIProvider(
            api_key=config["api_key"],
            model=config.get("model", "gpt-4o"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1024),
        )
    elif provider_type == "azure_openai":
        return AzureOpenAIProvider(
            api_key=config["api_key"],
            endpoint=config["endpoint"],
            deployment=config.get("deployment", config.get("model", "gpt-4o")),
            api_version=config.get("api_version", "2024-02-01"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1024),
        )
    elif provider_type == "gemini":
        return GeminiProvider(
            api_key=config["api_key"],
            model=config.get("model", "gemini-1.5-pro"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1024),
        )
    else:
        raise ValueError(
            f"Unknown provider type: '{provider_type}'. "
            "Supported: mock, openai, azure_openai, gemini"
        )
