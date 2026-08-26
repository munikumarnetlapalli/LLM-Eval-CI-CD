"""LLM metrics — deterministic and model-based."""
from __future__ import annotations

import re
from typing import Optional

from evaluator.providers.base import LLMProvider, Message
from evaluator.metrics.result import MetricResult, MetricCategory


# ─── Deterministic metrics ────────────────────────────────────────────────────

class LatencyMetric:
    """Deterministic: measures P50 and P95 response latency."""

    def evaluate_p50(self, latencies_ms: list[float], threshold_ms: float = 500) -> MetricResult:
        if not latencies_ms:
            return MetricResult.failing("p50_latency", 0.0, threshold_ms,
                                        "No latency data provided")
        sorted_latencies = sorted(latencies_ms)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        passed = p50 <= threshold_ms
        return MetricResult(
            metric="p50_latency",
            score=p50,
            threshold=threshold_ms,
            passed=passed,
            explanation=f"P50 latency: {p50:.1f}ms (threshold: {threshold_ms}ms)",
            category=MetricCategory.DETERMINISTIC,
        )

    def evaluate_p95(self, latencies_ms: list[float], threshold_ms: float = 1000) -> MetricResult:
        if not latencies_ms:
            return MetricResult.failing("p95_latency", 0.0, threshold_ms,
                                        "No latency data provided")
        sorted_latencies = sorted(latencies_ms)
        idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
        p95 = sorted_latencies[idx]
        passed = p95 <= threshold_ms
        return MetricResult(
            metric="p95_latency",
            score=p95,
            threshold=threshold_ms,
            passed=passed,
            explanation=f"P95 latency: {p95:.1f}ms (threshold: {threshold_ms}ms)",
            category=MetricCategory.DETERMINISTIC,
        )


class TokenUsageMetric:
    """Deterministic: measures token consumption."""

    def evaluate(
        self,
        input_tokens: int,
        output_tokens: int,
        max_total_tokens: int = 4096,
    ) -> MetricResult:
        total = input_tokens + output_tokens
        passed = total <= max_total_tokens
        return MetricResult(
            metric="token_usage",
            score=float(total),
            threshold=float(max_total_tokens),
            passed=passed,
            explanation=(
                f"Total tokens: {total} "
                f"(input={input_tokens}, output={output_tokens}, "
                f"threshold={max_total_tokens})"
            ),
            category=MetricCategory.DETERMINISTIC,
        )


class CostMetric:
    """Deterministic: estimates cost per query in USD."""

    def evaluate(
        self,
        input_tokens: int,
        output_tokens: int,
        input_per_1k: float,
        output_per_1k: float,
        max_cost_usd: float = 0.02,
    ) -> MetricResult:
        cost = (input_tokens / 1000) * input_per_1k + (output_tokens / 1000) * output_per_1k
        cost = round(cost, 6)
        passed = cost <= max_cost_usd
        return MetricResult(
            metric="cost_per_query",
            score=cost,
            threshold=max_cost_usd,
            passed=passed,
            explanation=(
                f"Estimated cost: ${cost:.6f}/query "
                f"(input={input_tokens} tok × ${input_per_1k}/1k, "
                f"output={output_tokens} tok × ${output_per_1k}/1k)"
            ),
            category=MetricCategory.DETERMINISTIC,
        )


# ─── Model-based metrics ──────────────────────────────────────────────────────

class AnswerRelevanceMetric:
    """Model-based: does the answer address the question?"""

    PROMPT_TEMPLATE = """You are an evaluation judge. Rate how relevant the following answer is to the question on a scale from 0.0 to 1.0.

Question: {question}
Answer: {answer}

Scoring rubric:
- 1.0: Answer directly and completely addresses the question
- 0.7-0.9: Answer mostly addresses the question with minor gaps
- 0.4-0.6: Answer partially addresses the question
- 0.1-0.3: Answer barely addresses the question
- 0.0: Answer is completely irrelevant or empty

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    def __init__(self, provider: LLMProvider, threshold: float = 0.80):
        self._provider = provider
        self._threshold = threshold

    def evaluate(self, question: str, answer: str) -> MetricResult:
        if not answer.strip():
            return MetricResult.failing(
                "answer_relevance", 0.0, self._threshold,
                "Empty answer — score is 0",
                MetricCategory.MODEL_BASED,
            )

        prompt = self.PROMPT_TEMPLATE.format(question=question, answer=answer[:2000])
        response = self._provider.complete([Message(role="user", content=prompt)])
        score, explanation = self._parse_response(response.content)

        return MetricResult(
            metric="answer_relevance",
            score=score,
            threshold=self._threshold,
            passed=score >= self._threshold,
            explanation=explanation,
            category=MetricCategory.MODEL_BASED,
        )

    def _parse_response(self, content: str) -> tuple[float, str]:
        try:
            import json
            # Extract JSON from response
            match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = float(data.get("score", 0.5))
                score = max(0.0, min(1.0, score))
                explanation = data.get("explanation", "Model evaluation complete")
                return score, explanation
        except Exception:
            pass
        # Fallback: extract first float in content
        nums = re.findall(r'\b(0\.\d+|1\.0)\b', content)
        if nums:
            return float(nums[0]), "Score extracted from model response"
        return 0.5, "Could not parse model response — defaulting to 0.5"


class HallucinationMetric:
    """Model-based: does the answer contain unsupported claims?"""

    PROMPT_TEMPLATE = """You are an evaluation judge assessing hallucination in an AI answer.

Question: {question}
Retrieved Context: {context}
AI Answer: {answer}

Task: Determine what fraction of claims in the answer are NOT supported by the retrieved context.
A claim is hallucinated if it introduces facts, numbers, names, or assertions not present in the context.

Scoring:
- 0.0: No hallucination — all claims are supported
- 0.1-0.3: Minor hallucination — small details not in context
- 0.4-0.6: Moderate hallucination — several unsupported claims
- 0.7-1.0: Severe hallucination — most content fabricated

Respond with ONLY a JSON object: {{"hallucination_rate": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    def __init__(self, provider: LLMProvider, threshold: float = 0.10):
        self._provider = provider
        self._threshold = threshold  # max acceptable hallucination rate

    def evaluate(self, question: str, answer: str, context: list[str]) -> MetricResult:
        context_str = "\n".join(context[:5]) if context else "No context retrieved."
        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            context=context_str[:3000],
            answer=answer[:2000],
        )
        response = self._provider.complete([Message(role="user", content=prompt)])
        rate, explanation = self._parse_response(response.content)

        # Lower hallucination = better; passes if rate <= threshold
        passed = rate <= self._threshold
        return MetricResult(
            metric="hallucination_rate",
            score=rate,
            threshold=self._threshold,
            passed=passed,
            explanation=explanation,
            category=MetricCategory.MODEL_BASED,
        )

    def _parse_response(self, content: str) -> tuple[float, str]:
        try:
            import json
            match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                rate = float(data.get("hallucination_rate", 0.1))
                rate = max(0.0, min(1.0, rate))
                return rate, data.get("explanation", "Hallucination evaluation complete")
        except Exception:
            pass
        nums = re.findall(r'\b(0\.\d+|1\.0)\b', content)
        if nums:
            return float(nums[0]), "Rate extracted from model response"
        return 0.1, "Could not parse model response — defaulting to 0.1"


class SemanticCorrectnessMetric:
    """Model-based: semantic similarity to expected answer."""

    PROMPT_TEMPLATE = """You are an evaluation judge. Compare the AI answer to the expected answer and rate semantic similarity.

Question: {question}
Expected Answer: {expected}
AI Answer: {actual}

Scoring:
- 1.0: Same meaning, equivalent information
- 0.7-0.9: Mostly correct, minor missing details
- 0.4-0.6: Partially correct
- 0.1-0.3: Mostly wrong or contradictory
- 0.0: Completely wrong or irrelevant

Respond with ONLY: {{"score": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    def __init__(self, provider: LLMProvider, threshold: float = 0.75):
        self._provider = provider
        self._threshold = threshold

    def evaluate(self, question: str, expected: str, actual: str) -> MetricResult:
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, expected=expected[:2000], actual=actual[:2000]
        )
        response = self._provider.complete([Message(role="user", content=prompt)])

        try:
            import json
            match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = max(0.0, min(1.0, float(data.get("score", 0.5))))
                explanation = data.get("explanation", "Semantic evaluation complete")
            else:
                score, explanation = 0.5, "Could not parse response"
        except Exception:
            score, explanation = 0.5, "Parse error — defaulting to 0.5"

        return MetricResult(
            metric="semantic_correctness",
            score=score,
            threshold=self._threshold,
            passed=score >= self._threshold,
            explanation=explanation,
            category=MetricCategory.MODEL_BASED,
        )
