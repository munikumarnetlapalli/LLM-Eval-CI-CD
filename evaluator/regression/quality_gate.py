"""Quality Gate — evaluates configurable pass/fail rules against run metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateRuleResult:
    metric: str
    rule_type: str  # "minimum" | "maximum"
    threshold: float
    actual_score: float
    passed: bool
    reason: str


@dataclass
class QualityGateResult:
    run_id: str
    passed: bool
    rule_results: list[GateRuleResult] = field(default_factory=list)
    failing_rules: list[GateRuleResult] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "PASSED" if self.passed else "FAILED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "verdict": self.verdict,
            "passed": self.passed,
            "failing_rules": [
                {
                    "metric": r.metric,
                    "rule_type": r.rule_type,
                    "threshold": r.threshold,
                    "actual_score": r.actual_score,
                    "reason": r.reason,
                }
                for r in self.failing_rules
            ],
            "total_rules": len(self.rule_results),
            "failing_count": len(self.failing_rules),
        }

    def format_report(self) -> str:
        """Produce a CI-friendly text report."""
        lines = [f"\n{'='*50}", f"QUALITY GATE: {self.verdict}", f"Run: {self.run_id}", ""]
        if self.failing_rules:
            lines.append("Failing rules:")
            for r in self.failing_rules:
                if r.rule_type == "minimum":
                    lines.append(
                        f"  ✗ {r.metric}: {r.actual_score:.4f} < {r.threshold} minimum"
                    )
                else:
                    lines.append(
                        f"  ✗ {r.metric}: {r.actual_score:.4f} > {r.threshold} maximum"
                    )
        else:
            lines.append("  All rules passed ✓")
        lines.append("=" * 50)
        return "\n".join(lines)


class QualityGate:
    """Evaluates configurable quality rules against evaluation run metrics.

    Rules are read from YAML config — never hard-coded.
    Output names the specific failing rules, not just an overall pass/fail.
    """

    def __init__(self, gate_config: dict[str, Any]):
        """
        gate_config format:
            faithfulness:
              minimum: 0.90
            hallucination_rate:
              maximum: 0.05
            p95_latency_ms:
              maximum_ms: 1000
        """
        self._config = gate_config

    def evaluate(self, run_id: str, metrics: dict[str, float]) -> QualityGateResult:
        rule_results: list[GateRuleResult] = []

        for metric_name, rules in self._config.items():
            actual = metrics.get(metric_name)
            if actual is None:
                continue  # metric not computed in this run — skip

            if "minimum" in rules:
                threshold = float(rules["minimum"])
                passed = actual >= threshold
                rule_results.append(GateRuleResult(
                    metric=metric_name,
                    rule_type="minimum",
                    threshold=threshold,
                    actual_score=actual,
                    passed=passed,
                    reason=(
                        f"{metric_name} is {actual:.4f}, "
                        f"meets minimum {threshold}"
                        if passed else
                        f"{metric_name} is {actual:.4f}, "
                        f"below minimum {threshold}"
                    ),
                ))

            if "maximum" in rules or "maximum_ms" in rules or "maximum_usd" in rules:
                threshold = float(
                    rules.get("maximum") or rules.get("maximum_ms") or rules.get("maximum_usd")
                )
                passed = actual <= threshold
                rule_results.append(GateRuleResult(
                    metric=metric_name,
                    rule_type="maximum",
                    threshold=threshold,
                    actual_score=actual,
                    passed=passed,
                    reason=(
                        f"{metric_name} is {actual:.4f}, "
                        f"within maximum {threshold}"
                        if passed else
                        f"{metric_name} is {actual:.4f}, "
                        f"exceeds maximum {threshold}"
                    ),
                ))

        failing = [r for r in rule_results if not r.passed]
        return QualityGateResult(
            run_id=run_id,
            passed=len(failing) == 0,
            rule_results=rule_results,
            failing_rules=failing,
        )
