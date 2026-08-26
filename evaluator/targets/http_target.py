"""HTTPRAGTarget — sends questions to a remote RAG HTTP endpoint."""
from __future__ import annotations

import time
from typing import Any

import httpx

from .base import EvaluationTarget, TargetResponse


class HTTPRAGTarget(EvaluationTarget):
    """Evaluation target that calls a remote RAG HTTP API.

    The remote endpoint can be any HTTP service — AegisAI, a custom RAG API,
    or any other deployed system. The evaluator doesn't care which.
    """

    def __init__(
        self,
        base_url: str,
        query_path: str = "/query",
        target_id: str = "http-rag-v1",
        timeout_seconds: float = 60.0,
        headers: dict[str, str] | None = None,
        question_field: str = "question",
        answer_field: str = "answer",
        context_field: str = "context",
        sources_field: str = "sources",
    ):
        self._base_url = base_url.rstrip("/")
        self._query_path = query_path
        self._target_id = target_id
        self._timeout = timeout_seconds
        self._headers = headers or {"Content-Type": "application/json"}
        self._question_field = question_field
        self._answer_field = answer_field
        self._context_field = context_field
        self._sources_field = sources_field

    def query(self, question: str, **kwargs) -> TargetResponse:
        payload: dict[str, Any] = {self._question_field: question}
        payload.update(kwargs)

        t0 = time.monotonic()
        try:
            response = httpx.post(
                f"{self._base_url}{self._query_path}",
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            latency_ms = (time.monotonic() - t0) * 1000
            data = response.json()

            return TargetResponse(
                answer=data.get(self._answer_field, ""),
                retrieved_context=data.get(self._context_field, []),
                retrieved_sources=data.get(self._sources_field, []),
                latency_ms=latency_ms,
                metadata={"status_code": response.status_code},
            )
        except httpx.HTTPError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            return TargetResponse(
                answer="",
                latency_ms=latency_ms,
                error=f"HTTP error: {e}",
            )

    @property
    def target_id(self) -> str:
        return self._target_id


class LocalLLMTarget(EvaluationTarget):
    """Evaluation target that uses an LLMProvider directly (no RAG pipeline)."""

    def __init__(self, provider, target_id: str = "local-llm-v1"):
        self._provider = provider
        self._target_id = target_id

    def query(self, question: str, **kwargs) -> TargetResponse:
        from evaluator.providers.base import Message
        response = self._provider.complete(
            [Message(role="user", content=question)]
        )
        return TargetResponse(
            answer=response.content,
            retrieved_context=[],
            retrieved_sources=[],
            latency_ms=response.latency_ms,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

    @property
    def target_id(self) -> str:
        return self._target_id
