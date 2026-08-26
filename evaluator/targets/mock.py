"""MockTarget — deterministic evaluation target for unit tests."""
from __future__ import annotations

from .base import EvaluationTarget, TargetResponse


class MockTarget(EvaluationTarget):
    """Deterministic mock target that returns fixed responses without any API calls.

    Used for all unit and integration tests.
    """

    DEFAULT_ANSWER = (
        "Based on the company policy documents, the answer is clearly defined. "
        "The policy states that the procedure must be followed as documented."
    )
    DEFAULT_CONTEXT = [
        "The company retention policy requires records to be kept for 7 years.",
        "All employees must follow the documented procedures.",
    ]
    DEFAULT_SOURCES = ["policy-document.pdf"]

    def __init__(
        self,
        answer: str = DEFAULT_ANSWER,
        context: list[str] | None = None,
        sources: list[str] | None = None,
        latency_ms: float = 150.0,
        fail: bool = False,
        fail_message: str = "MockTarget configured to fail",
        target_id: str = "mock-target-v1",
    ):
        self._answer = answer
        self._context = context if context is not None else self.DEFAULT_CONTEXT
        self._sources = sources if sources is not None else self.DEFAULT_SOURCES
        self._latency_ms = latency_ms
        self._fail = fail
        self._fail_message = fail_message
        self._target_id = target_id
        self._call_count = 0

    def query(self, question: str, **kwargs) -> TargetResponse:
        self._call_count += 1
        if self._fail:
            return TargetResponse(
                answer="",
                error=self._fail_message,
                latency_ms=self._latency_ms,
            )
        return TargetResponse(
            answer=self._answer,
            retrieved_context=self._context,
            retrieved_sources=self._sources,
            latency_ms=self._latency_ms,
        )

    @property
    def target_id(self) -> str:
        return self._target_id

    @property
    def call_count(self) -> int:
        return self._call_count
