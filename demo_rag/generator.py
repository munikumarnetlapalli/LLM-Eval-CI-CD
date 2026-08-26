"""
Demo generator for the local test RAG target.
Combines retrieved context with the question to produce a templated answer.
No LLM calls — purely template-based for deterministic testing.
"""
from __future__ import annotations


class DemoGenerator:
    """Template-based answer generator for test purposes.

    In the test target, we don't make real LLM calls — we produce a
    deterministic response that concatenates the retrieved context.
    This is enough to test faithfulness, context relevance, and retrieval
    metrics without any API dependency.
    """

    def generate(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return (
                "Based on the available documentation, I was unable to find "
                "specific information to answer this question."
            )

        context_str = " ".join(context_chunks)
        return (
            f"Based on the available policy documentation: {context_str} "
            f"This information directly addresses your question about: {question[:80]}."
        )
