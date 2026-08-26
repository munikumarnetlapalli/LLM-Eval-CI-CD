"""MockProvider — deterministic provider for unit tests (no API calls)."""
from __future__ import annotations

import time
from typing import Optional

from .base import LLMProvider, LLMResponse, Message, ModelPricing


class MockProvider(LLMProvider):
    """Deterministic mock LLM provider for testing.

    Returns configurable responses without making any API calls.
    """

    DEFAULT_RESPONSE = (
        "This is a mock response. The answer is based on the provided context and question."
    )
    DEFAULT_EMBEDDING = [0.1] * 1536  # Matches OpenAI ada-002 dimensions

    def __init__(
        self,
        response_text: str = DEFAULT_RESPONSE,
        input_tokens: int = 150,
        output_tokens: int = 50,
        latency_ms: float = 120.0,
        fail: bool = False,
        fail_message: str = "MockProvider configured to fail",
    ):
        self._response_text = response_text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._latency_ms = latency_ms
        self._fail = fail
        self._fail_message = fail_message
        self._call_count = 0

    def complete(self, messages: list[Message], **kwargs) -> LLMResponse:
        self._call_count += 1
        if self._fail:
            raise RuntimeError(self._fail_message)
        # Simulate deterministic latency
        time.sleep(self._latency_ms / 1000.0 * 0.01)  # 1% of simulated latency for speed
        return LLMResponse(
            content=self._response_text,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=self._latency_ms,
            model="mock-gpt-4",
            model_version="mock-v1",
        )

    def embed(self, text: str) -> list[float]:
        if self._fail:
            raise RuntimeError(self._fail_message)
        # Return a deterministic embedding based on text length
        base = [0.1] * 1536
        base[0] = len(text) / 10000.0
        return base

    @property
    def model_name(self) -> str:
        return "mock-gpt-4"

    @property
    def pricing(self) -> ModelPricing:
        return ModelPricing(input_per_1k=0.01, output_per_1k=0.03, model="mock-gpt-4")

    @property
    def call_count(self) -> int:
        return self._call_count
