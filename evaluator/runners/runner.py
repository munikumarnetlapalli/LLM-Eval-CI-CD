"""
Evaluation Runner — orchestrates target queries, metric computation, and result collection.
"""
from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from evaluator.datasets.schema import DatasetManifest, EvaluationCase, RAGEvaluationCase
from evaluator.metrics.result import MetricResult, MetricCategory
from evaluator.metrics.llm.metrics import (
    LatencyMetric, TokenUsageMetric, CostMetric,
    AnswerRelevanceMetric, HallucinationMetric, SemanticCorrectnessMetric,
)
from evaluator.metrics.rag.metrics import (
    FaithfulnessMetric, ContextRelevanceMetric, RetrievalQualityMetric,
)
from evaluator.providers.base import LLMProvider
from evaluator.targets.base import EvaluationTarget, TargetResponse


@dataclass
class CaseResult:
    """Result for a single evaluation case."""
    case_id: str
    question: str
    expected_answer: str
    actual_answer: str
    category: str
    difficulty: str
    metrics: list[MetricResult] = field(default_factory=list)
    target_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics) and self.error is None

    @property
    def metric_dict(self) -> dict[str, float]:
        return {m.metric: m.score for m in self.metrics}


@dataclass
class EvaluationRunResult:
    """Complete result of an evaluation run."""
    run_id: str
    experiment_id: str
    model: str
    model_version: str
    prompt_version: str
    rag_version: str
    kb_version: str
    dataset_version: str
    git_sha: str
    timestamp: str
    environment: str

    case_results: list[CaseResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    error_count: int = 0
    quality_gate_result: Optional[str] = None  # "PASSED" | "FAILED" | None

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for c in self.case_results if c.passed)

    @property
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return self.passed_cases / self.total_cases

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model": self.model,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "rag_version": self.rag_version,
            "kb_version": self.kb_version,
            "dataset_version": self.dataset_version,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "pass_rate": round(self.pass_rate, 4),
            "aggregate_metrics": self.aggregate_metrics,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "error_count": self.error_count,
            "quality_gate_result": self.quality_gate_result,
        }


def _get_git_sha() -> str:
    """Get current Git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class EvaluationRunner:
    """Orchestrates the full evaluation loop.

    For each case in the dataset:
    1. Query the evaluation target
    2. Compute all configured metrics
    3. Collect results into CaseResult
    4. Aggregate into EvaluationRunResult
    """

    def __init__(
        self,
        target: EvaluationTarget,
        judge_provider: LLMProvider,
        config: dict[str, Any] | None = None,
    ):
        self._target = target
        self._judge = judge_provider
        self._config = config or {}

        # Thresholds from config, with sensible defaults
        gate_cfg = self._config.get("quality_gate", {})
        self._faithfulness_threshold = gate_cfg.get("faithfulness", {}).get("minimum", 0.90)
        self._relevance_threshold = gate_cfg.get("answer_relevance", {}).get("minimum", 0.80)
        self._hallucination_threshold = gate_cfg.get("hallucination", {}).get("maximum", 0.10)
        self._context_relevance_threshold = gate_cfg.get("context_relevance", {}).get("minimum", 0.80)
        self._retrieval_quality_threshold = gate_cfg.get("retrieval_quality", {}).get("minimum", 0.80)
        self._p95_threshold_ms = gate_cfg.get("p95_latency", {}).get("maximum_ms", 1000)
        self._max_cost_usd = gate_cfg.get("cost_per_query", {}).get("maximum_usd", 0.02)

        # Metric instances
        self._faithfulness = FaithfulnessMetric(judge_provider, self._faithfulness_threshold)
        self._context_relevance = ContextRelevanceMetric(judge_provider, self._context_relevance_threshold)
        self._retrieval_quality = RetrievalQualityMetric(self._retrieval_quality_threshold)
        self._answer_relevance = AnswerRelevanceMetric(judge_provider, self._relevance_threshold)
        self._hallucination = HallucinationMetric(judge_provider, self._hallucination_threshold)
        self._semantic = SemanticCorrectnessMetric(judge_provider, 0.75)
        self._latency_metric = LatencyMetric()
        self._token_metric = TokenUsageMetric()
        self._cost_metric = CostMetric()

    def run(
        self,
        dataset: DatasetManifest,
        experiment_id: Optional[str] = None,
        prompt_version: str = "v1",
        rag_version: str = "v1",
        kb_version: str = "KB-001",
        environment: str = "local",
    ) -> EvaluationRunResult:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        exp_id = experiment_id or f"exp-{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        git_sha = _get_git_sha()

        result = EvaluationRunResult(
            run_id=run_id,
            experiment_id=exp_id,
            model=self._target.target_id,
            model_version="",
            prompt_version=prompt_version,
            rag_version=rag_version,
            kb_version=kb_version,
            dataset_version=dataset.version,
            git_sha=git_sha,
            timestamp=timestamp,
            environment=environment,
        )

        pricing = self._judge.pricing

        for case in dataset.cases:
            case_result = self._evaluate_case(case, pricing)
            result.case_results.append(case_result)
            result.latencies_ms.append(case_result.target_latency_ms)
            result.total_input_tokens += case_result.input_tokens
            result.total_output_tokens += case_result.output_tokens
            if case_result.error:
                result.error_count += 1

        # Aggregate metrics
        result.aggregate_metrics = self._aggregate_metrics(result)
        result.total_cost_usd = round(
            (result.total_input_tokens / 1000) * pricing.input_per_1k
            + (result.total_output_tokens / 1000) * pricing.output_per_1k,
            4,
        )

        return result

    def _evaluate_case(self, case: EvaluationCase | RAGEvaluationCase, pricing) -> CaseResult:
        """Evaluate a single case and return all metric results."""
        target_resp: TargetResponse = self._target.query(case.question)

        required_sources = getattr(case, "required_sources", [])
        metrics: list[MetricResult] = []

        if target_resp.failed:
            return CaseResult(
                case_id=case.id,
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer="",
                category=case.category.value,
                difficulty=case.difficulty.value,
                error=target_resp.error,
                target_latency_ms=target_resp.latency_ms,
            )

        # Always compute answer relevance and hallucination
        metrics.append(self._answer_relevance.evaluate(case.question, target_resp.answer))
        metrics.append(self._hallucination.evaluate(
            case.question, target_resp.answer, target_resp.retrieved_context
        ))
        metrics.append(self._semantic.evaluate(
            case.question, case.expected_answer, target_resp.answer
        ))

        # RAG-specific metrics
        if target_resp.has_context:
            metrics.append(self._faithfulness.evaluate(
                target_resp.answer, target_resp.retrieved_context
            ))
            metrics.append(self._context_relevance.evaluate(
                case.question, target_resp.retrieved_context
            ))
            if required_sources:
                metrics.append(self._retrieval_quality.evaluate(
                    target_resp.retrieved_sources, required_sources
                ))

        # Deterministic cost metric per case
        metrics.append(self._cost_metric.evaluate(
            input_tokens=50,   # estimated per-case judge tokens
            output_tokens=100,
            input_per_1k=pricing.input_per_1k,
            output_per_1k=pricing.output_per_1k,
            max_cost_usd=self._max_cost_usd,
        ))

        return CaseResult(
            case_id=case.id,
            question=case.question,
            expected_answer=case.expected_answer,
            actual_answer=target_resp.answer,
            category=case.category.value,
            difficulty=case.difficulty.value,
            metrics=metrics,
            target_latency_ms=target_resp.latency_ms,
        )

    def _aggregate_metrics(self, result: EvaluationRunResult) -> dict[str, float]:
        """Compute per-metric averages across all cases."""
        from collections import defaultdict
        sums: dict[str, list[float]] = defaultdict(list)

        for case_result in result.case_results:
            for m in case_result.metrics:
                sums[m.metric].append(m.score)

        aggregated = {
            metric: round(sum(scores) / len(scores), 4)
            for metric, scores in sums.items()
            if scores
        }

        # Add latency percentiles
        if result.latencies_ms:
            lat = sorted(result.latencies_ms)
            aggregated["p50_latency_ms"] = lat[int(len(lat) * 0.50)]
            aggregated["p95_latency_ms"] = lat[min(int(len(lat) * 0.95), len(lat) - 1)]

        return aggregated
