"""RAG-specific metrics — faithfulness, context relevance, retrieval quality."""
from __future__ import annotations

import re
from typing import Optional

from evaluator.providers.base import LLMProvider, Message
from evaluator.metrics.result import MetricResult, MetricCategory


class FaithfulnessMetric:
    """Model-based: is the answer supported by retrieved context?

    Faithfulness measures whether every claim in the answer can be
    traced back to the retrieved context. High faithfulness = low hallucination
    relative to what was retrieved.
    """

    PROMPT_TEMPLATE = """You are an evaluation judge assessing faithfulness of a RAG system answer.

Retrieved Context:
{context}

AI Answer:
{answer}

Task: What fraction of claims in the answer are directly supported by the retrieved context above?
Do NOT use external knowledge — only evaluate support within the provided context.

Scoring:
- 1.0: Every claim is explicitly supported by the context
- 0.7-0.9: Most claims are supported; minor extrapolations
- 0.4-0.6: Some claims supported, others not in context
- 0.1-0.3: Most claims are not in the context
- 0.0: No claims supported by context (complete fabrication)

Respond with ONLY: {{"faithfulness_score": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    def __init__(self, provider: LLMProvider, threshold: float = 0.90):
        self._provider = provider
        self._threshold = threshold

    def evaluate(self, answer: str, context: list[str]) -> MetricResult:
        if not context:
            return MetricResult.failing(
                "faithfulness", 0.0, self._threshold,
                "No context was retrieved — faithfulness cannot be evaluated",
                MetricCategory.MODEL_BASED,
            )
        if not answer.strip():
            return MetricResult.failing(
                "faithfulness", 0.0, self._threshold,
                "Empty answer — score is 0",
                MetricCategory.MODEL_BASED,
            )

        context_str = "\n---\n".join(context[:5])
        prompt = self.PROMPT_TEMPLATE.format(
            context=context_str[:3000],
            answer=answer[:2000],
        )
        response = self._provider.complete([Message(role="user", content=prompt)])
        score, explanation = self._parse_response(response.content)

        return MetricResult(
            metric="faithfulness",
            score=score,
            threshold=self._threshold,
            passed=score >= self._threshold,
            explanation=explanation,
            category=MetricCategory.MODEL_BASED,
        )

    def _parse_response(self, content: str) -> tuple[float, str]:
        try:
            import json
            match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(0.0, min(1.0, float(data.get("faithfulness_score", 0.8))))
                return score, data.get("explanation", "Faithfulness evaluation complete")
        except Exception:
            pass
        nums = re.findall(r'\b(0\.\d+|1\.0)\b', content)
        if nums:
            return float(nums[0]), "Score extracted from response"
        return 0.8, "Could not parse faithfulness response"


class ContextRelevanceMetric:
    """Retrieval-based: are retrieved chunks relevant to the question?"""

    PROMPT_TEMPLATE = """You are evaluating retrieval quality in a RAG system.

Question: {question}

Retrieved Context:
{context}

Task: Rate how relevant the retrieved context is for answering this question.

Scoring:
- 1.0: Context is perfectly targeted to the question
- 0.7-0.9: Context is mostly relevant with some noise
- 0.4-0.6: Context is partially relevant
- 0.1-0.3: Context is mostly irrelevant
- 0.0: Context has nothing to do with the question

Respond with ONLY: {{"relevance_score": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    def __init__(self, provider: LLMProvider, threshold: float = 0.80):
        self._provider = provider
        self._threshold = threshold

    def evaluate(self, question: str, context: list[str]) -> MetricResult:
        if not context:
            return MetricResult.failing(
                "context_relevance", 0.0, self._threshold,
                "No context retrieved",
                MetricCategory.RETRIEVAL_BASED,
            )

        context_str = "\n---\n".join(context[:5])
        prompt = self.PROMPT_TEMPLATE.format(question=question, context=context_str[:3000])
        response = self._provider.complete([Message(role="user", content=prompt)])

        try:
            import json
            match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(0.0, min(1.0, float(data.get("relevance_score", 0.7))))
                explanation = data.get("explanation", "Context relevance evaluated")
            else:
                score, explanation = 0.7, "Could not parse response"
        except Exception:
            score, explanation = 0.7, "Parse error — defaulting to 0.7"

        return MetricResult(
            metric="context_relevance",
            score=score,
            threshold=self._threshold,
            passed=score >= self._threshold,
            explanation=explanation,
            category=MetricCategory.RETRIEVAL_BASED,
        )


class RetrievalQualityMetric:
    """Retrieval-based: recall of required sources."""

    def __init__(self, threshold: float = 0.80):
        self._threshold = threshold

    def evaluate(self, retrieved_sources: list[str], required_sources: list[str]) -> MetricResult:
        if not required_sources:
            return MetricResult.passing(
                "retrieval_quality", 1.0, self._threshold,
                "No required sources specified — retrieval quality not evaluated",
                MetricCategory.RETRIEVAL_BASED,
            )

        found = sum(
            1 for req in required_sources
            if any(req.lower() in src.lower() or src.lower() in req.lower()
                   for src in retrieved_sources)
        )
        recall = found / len(required_sources)
        passed = recall >= self._threshold
        missing = [r for r in required_sources
                   if not any(r.lower() in s.lower() or s.lower() in r.lower()
                              for s in retrieved_sources)]
        explanation = (
            f"Retrieved {found}/{len(required_sources)} required sources. "
            + (f"Missing: {missing}" if missing else "All required sources found.")
        )
        return MetricResult(
            metric="retrieval_quality",
            score=recall,
            threshold=self._threshold,
            passed=passed,
            explanation=explanation,
            category=MetricCategory.RETRIEVAL_BASED,
        )
