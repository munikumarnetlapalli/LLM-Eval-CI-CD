"""LLM Provider abstraction — base interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ModelPricing:
    """Pricing per 1,000 tokens (USD)."""
    input_per_1k: float
    output_per_1k: float
    model: str


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    model_version: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def estimated_cost(self, pricing: ModelPricing) -> float:
        """Compute estimated cost in USD."""
        input_cost = (self.input_tokens / 1000) * pricing.input_per_1k
        output_cost = (self.output_tokens / 1000) * pricing.output_per_1k
        return round(input_cost + output_cost, 6)


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Implementations must be API-based only — no local weights, no downloads.
    """

    @abstractmethod
    def complete(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Send a completion request and return a structured response."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for the given text."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def pricing(self) -> ModelPricing:
        """Pricing configuration for cost estimation."""
        ...

    def complete_text(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Convenience wrapper: build messages from a plain prompt string."""
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return self.complete(messages)
