"""Gemini provider — Google Gemini API (API-based only)."""
from __future__ import annotations

import time

from .base import LLMProvider, LLMResponse, Message, ModelPricing

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

GEMINI_PRICING: dict[str, ModelPricing] = {
    "gemini-1.5-pro": ModelPricing(input_per_1k=0.00125, output_per_1k=0.005, model="gemini-1.5-pro"),
    "gemini-1.5-flash": ModelPricing(input_per_1k=0.000075, output_per_1k=0.0003, model="gemini-1.5-flash"),
    "gemini-2.0-flash": ModelPricing(input_per_1k=0.0001, output_per_1k=0.0004, model="gemini-2.0-flash"),
}
DEFAULT_PRICING = ModelPricing(input_per_1k=0.00125, output_per_1k=0.005, model="gemini-1.5-pro")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider. API-based only — no local model weights."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        if not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            )
        genai.configure(api_key=api_key)
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = genai.GenerativeModel(model)

    def complete(self, messages: list[Message], **kwargs) -> LLMResponse:
        # Convert messages to Gemini format
        prompt_parts = []
        system_text = ""
        for m in messages:
            if m.role == "system":
                system_text = m.content
            elif m.role == "user":
                if system_text:
                    prompt_parts.append(f"System: {system_text}\n\nUser: {m.content}")
                    system_text = ""
                else:
                    prompt_parts.append(m.content)
            elif m.role == "assistant":
                prompt_parts.append(f"Assistant: {m.content}")

        prompt = "\n".join(prompt_parts)
        config = genai.GenerationConfig(
            temperature=kwargs.get("temperature", self._temperature),
            max_output_tokens=kwargs.get("max_tokens", self._max_tokens),
        )

        t0 = time.monotonic()
        response = self._client.generate_content(prompt, generation_config=config)
        latency_ms = (time.monotonic() - t0) * 1000

        text = response.text if response.parts else ""
        # Gemini token counts (approximate if not available)
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self._model_name,
            model_version=self._model_name,
        )

    def embed(self, text: str) -> list[float]:
        result = genai.embed_content(model="models/embedding-001", content=text)
        return result["embedding"]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def pricing(self) -> ModelPricing:
        for key, p in GEMINI_PRICING.items():
            if key in self._model_name:
                return p
        return DEFAULT_PRICING
