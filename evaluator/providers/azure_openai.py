"""Azure OpenAI provider — Azure OpenAI API (API-based only)."""
from __future__ import annotations

import time

from .base import LLMProvider, LLMResponse, Message, ModelPricing

try:
    from openai import AzureOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

AZURE_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_per_1k=0.005, output_per_1k=0.015, model="gpt-4o"),
    "gpt-4o-mini": ModelPricing(input_per_1k=0.000165, output_per_1k=0.00066, model="gpt-4o-mini"),
    "gpt-4-turbo": ModelPricing(input_per_1k=0.01, output_per_1k=0.03, model="gpt-4-turbo"),
}
DEFAULT_PRICING = ModelPricing(input_per_1k=0.005, output_per_1k=0.015, model="azure-gpt-4o")


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider. API-based only — no local model weights."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str = "2024-02-01",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages: list[Message], **kwargs) -> LLMResponse:
        oai_messages = [{"role": m.role, "content": m.content} for m in messages]
        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=oai_messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model=self._deployment,
            model_version=response.model,
        )

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding

    @property
    def model_name(self) -> str:
        return self._deployment

    @property
    def pricing(self) -> ModelPricing:
        # Try to match based on deployment name fragment
        for key, pricing in AZURE_PRICING.items():
            if key in self._deployment.lower():
                return pricing
        return DEFAULT_PRICING
