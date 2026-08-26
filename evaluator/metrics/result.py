"""
Metric result schema — every metric returns this shape.
Dashboards, quality gates, and comparisons all depend on this structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricCategory(str, Enum):
    DETERMINISTIC = "deterministic"
    MODEL_BASED = "model_based"
    RETRIEVAL_BASED = "retrieval_based"


@dataclass
class MetricResult:
    """Structured output from any metric evaluation.

    This is the universal shape for all metrics — deterministic,
    model-based, and retrieval-based. Never return a bare float.
    """
    metric: str
    score: float
    threshold: float
    passed: bool
    explanation: str
    category: MetricCategory = MetricCategory.DETERMINISTIC

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "explanation": self.explanation,
            "category": self.category.value,
        }

    @classmethod
    def passing(cls, metric: str, score: float, threshold: float,
                explanation: str, category: MetricCategory = MetricCategory.DETERMINISTIC) -> "MetricResult":
        return cls(metric=metric, score=score, threshold=threshold,
                  passed=True, explanation=explanation, category=category)

    @classmethod
    def failing(cls, metric: str, score: float, threshold: float,
                explanation: str, category: MetricCategory = MetricCategory.DETERMINISTIC) -> "MetricResult":
        return cls(metric=metric, score=score, threshold=threshold,
                  passed=False, explanation=explanation, category=category)
