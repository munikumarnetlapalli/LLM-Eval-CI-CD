"""EvaluationTarget abstraction — base interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TargetResponse:
    """Response from an evaluation target."""
    answer: str
    retrieved_context: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def has_context(self) -> bool:
        return len(self.retrieved_context) > 0

    @property
    def failed(self) -> bool:
        return self.error is not None


class EvaluationTarget(ABC):
    """Abstract base class for all evaluation targets.

    The same golden dataset can be run against any target —
    local mock, local demo RAG, or a remote HTTP endpoint.
    """

    @abstractmethod
    def query(self, question: str, **kwargs) -> TargetResponse:
        """Send a question to the target and return a structured response."""
        ...

    @property
    @abstractmethod
    def target_id(self) -> str:
        """Unique identifier for this target configuration."""
        ...

    @property
    def target_type(self) -> str:
        """Type name for reporting."""
        return self.__class__.__name__
