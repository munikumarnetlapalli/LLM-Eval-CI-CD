"""LocalRAGTarget — wraps the demo_rag module for local integration testing."""
from __future__ import annotations

import time
from pathlib import Path

from .base import EvaluationTarget, TargetResponse


class LocalRAGTarget(EvaluationTarget):
    """Evaluation target that uses the lightweight local demo RAG pipeline.

    This is a TEST TARGET ONLY — not the main product.
    It exists to enable local integration tests without any external dependencies.
    """

    def __init__(
        self,
        knowledge_base_path: str | Path = "demo_rag/knowledge_base",
        top_k: int = 3,
        target_id: str = "local-rag-v1",
    ):
        from demo_rag.retriever import KeywordRetriever
        from demo_rag.generator import DemoGenerator

        self._retriever = KeywordRetriever(knowledge_base_path=knowledge_base_path, top_k=top_k)
        self._generator = DemoGenerator()
        self._target_id = target_id
        self._top_k = top_k

    def query(self, question: str, **kwargs) -> TargetResponse:
        t0 = time.monotonic()
        chunks, sources = self._retriever.retrieve(question)
        answer = self._generator.generate(question, chunks)
        latency_ms = (time.monotonic() - t0) * 1000

        return TargetResponse(
            answer=answer,
            retrieved_context=chunks,
            retrieved_sources=sources,
            latency_ms=latency_ms,
        )

    @property
    def target_id(self) -> str:
        return self._target_id
