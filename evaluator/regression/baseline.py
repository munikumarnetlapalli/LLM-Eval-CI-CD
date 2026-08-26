"""
Baseline Manager + Regression Analyzer + Quality Gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from evaluator.storage.backends import StorageBackend


@dataclass
class BaselineSnapshot:
    """A stored baseline to compare future runs against."""
    baseline_id: str
    run_id: str
    experiment_id: str
    model: str
    prompt_version: str
    rag_version: str
    kb_version: str
    dataset_version: str
    git_sha: str
    timestamp: str
    metrics: dict[str, float]
    environment: str = "local"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "rag_version": self.rag_version,
            "kb_version": self.kb_version,
            "dataset_version": self.dataset_version,
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "environment": self.environment,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaselineSnapshot":
        return cls(**data)


class BaselineManager:
    """Stores and retrieves baseline snapshots."""

    BASELINE_PREFIX = "baselines"

    def __init__(self, storage: StorageBackend):
        self._storage = storage

    def save_baseline(self, snapshot: BaselineSnapshot) -> str:
        key = f"{self.BASELINE_PREFIX}/{snapshot.baseline_id}"
        return self._storage.save(key, snapshot.to_dict())

    def load_baseline(self, baseline_id: str) -> BaselineSnapshot:
        key = f"{self.BASELINE_PREFIX}/{baseline_id}"
        data = self._storage.load(key)
        return BaselineSnapshot.from_dict(data)

    def list_baselines(self) -> list[str]:
        return self._storage.list_keys(prefix=self.BASELINE_PREFIX)

    def set_as_baseline(
        self,
        run_result,
        baseline_id: Optional[str] = None,
        notes: str = "",
    ) -> BaselineSnapshot:
        """Promote an evaluation run result to a baseline."""
        import uuid
        bid = baseline_id or f"baseline-{uuid.uuid4().hex[:8]}"
        snapshot = BaselineSnapshot(
            baseline_id=bid,
            run_id=run_result.run_id,
            experiment_id=run_result.experiment_id,
            model=run_result.model,
            prompt_version=run_result.prompt_version,
            rag_version=run_result.rag_version,
            kb_version=run_result.kb_version,
            dataset_version=run_result.dataset_version,
            git_sha=run_result.git_sha,
            timestamp=run_result.timestamp,
            metrics=run_result.aggregate_metrics,
            environment=run_result.environment,
            notes=notes,
        )
        self.save_baseline(snapshot)
        return snapshot


@dataclass
class RegressionDetail:
    metric: str
    baseline_score: float
    candidate_score: float
    delta: float
    delta_pct: float
    direction: str  # "better" | "worse" | "neutral"
    is_regression: bool
    explanation: str


@dataclass
class RegressionReport:
    baseline_id: str
    candidate_run_id: str
    overall_delta_pct: float
    is_regression: bool
    regressions: list[RegressionDetail] = field(default_factory=list)
    improvements: list[RegressionDetail] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_run_id": self.candidate_run_id,
            "overall_delta_pct": round(self.overall_delta_pct, 2),
            "is_regression": self.is_regression,
            "regressions": [
                {
                    "metric": r.metric,
                    "baseline_score": r.baseline_score,
                    "candidate_score": r.candidate_score,
                    "delta_pct": round(r.delta_pct, 2),
                    "explanation": r.explanation,
                }
                for r in self.regressions
            ],
            "improvements": [
                {
                    "metric": r.metric,
                    "delta_pct": round(r.delta_pct, 2),
                }
                for r in self.improvements
            ],
            "summary": self.summary,
        }


# Metrics where LOWER is better
LOWER_IS_BETTER = {"hallucination_rate", "p50_latency_ms", "p95_latency_ms", "cost_per_query", "token_usage"}

# Regression threshold (5% degradation triggers a regression warning)
REGRESSION_THRESHOLD_PCT = 5.0


class RegressionAnalyzer:
    """Detects regressions between a baseline and a candidate run.

    Uses language like 'potential regression area' — never claims causality.
    """

    def analyze(
        self,
        baseline: BaselineSnapshot,
        candidate_metrics: dict[str, float],
        candidate_run_id: str,
        regression_threshold_pct: float = REGRESSION_THRESHOLD_PCT,
    ) -> RegressionReport:
        regressions: list[RegressionDetail] = []
        improvements: list[RegressionDetail] = []
        deltas: list[float] = []

        all_metrics = set(baseline.metrics.keys()) | set(candidate_metrics.keys())

        for metric in all_metrics:
            if metric not in baseline.metrics or metric not in candidate_metrics:
                continue

            b_score = baseline.metrics[metric]
            c_score = candidate_metrics[metric]

            if b_score == 0:
                continue

            delta = c_score - b_score
            delta_pct = (delta / abs(b_score)) * 100

            lower_better = metric in LOWER_IS_BETTER
            is_worse = (delta_pct < -regression_threshold_pct and not lower_better) or \
                       (delta_pct > regression_threshold_pct and lower_better)
            is_better = (delta_pct > regression_threshold_pct and not lower_better) or \
                        (delta_pct < -regression_threshold_pct and lower_better)

            detail = RegressionDetail(
                metric=metric,
                baseline_score=b_score,
                candidate_score=c_score,
                delta=delta,
                delta_pct=delta_pct,
                direction="worse" if is_worse else ("better" if is_better else "neutral"),
                is_regression=is_worse,
                explanation=(
                    f"Potential regression area: {metric} — "
                    f"changed from {b_score:.3f} to {c_score:.3f} "
                    f"({delta_pct:+.1f}%)"
                    if is_worse else
                    f"{metric} improved from {b_score:.3f} to {c_score:.3f} ({delta_pct:+.1f}%)"
                ),
            )
            deltas.append(delta_pct if not lower_better else -delta_pct)

            if is_worse:
                regressions.append(detail)
            elif is_better:
                improvements.append(detail)

        overall_delta_pct = sum(deltas) / len(deltas) if deltas else 0.0
        is_regression = len(regressions) > 0

        summary_parts = []
        if regressions:
            reg_names = [r.metric for r in regressions]
            summary_parts.append(
                f"Potential regression areas: {', '.join(reg_names)}. "
                "Recommend human review before deployment."
            )
        if improvements:
            imp_names = [r.metric for r in improvements]
            summary_parts.append(f"Improvements detected in: {', '.join(imp_names)}.")

        return RegressionReport(
            baseline_id=baseline.baseline_id,
            candidate_run_id=candidate_run_id,
            overall_delta_pct=overall_delta_pct,
            is_regression=is_regression,
            regressions=regressions,
            improvements=improvements,
            summary=" ".join(summary_parts) or "No significant changes detected.",
        )
