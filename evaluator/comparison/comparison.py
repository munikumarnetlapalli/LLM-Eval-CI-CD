"""Model comparison, Model×RAG grid, and cost-quality analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelComparisonRow:
    model: str
    quality_score: float
    faithfulness: float
    answer_relevance: float
    hallucination_rate: float
    p95_latency_ms: float
    cost_per_query_usd: float
    weighted_score: float = 0.0


@dataclass
class ModelComparisonResult:
    rows: list[ModelComparisonRow]
    best_quality: str
    best_latency: str
    lowest_cost: str
    best_balanced: str

    def to_table(self) -> str:
        header = f"{'Model':<20} {'Quality':>8} {'Faithful':>10} {'Relevance':>10} {'P95':>8} {'Cost/Q':>10}"
        sep = "-" * 72
        lines = [header, sep]
        for r in sorted(self.rows, key=lambda x: x.quality_score, reverse=True):
            lines.append(
                f"{r.model:<20} {r.quality_score:>7.1%} {r.faithfulness:>9.1%} "
                f"{r.answer_relevance:>9.1%} {r.p95_latency_ms:>7.0f}ms "
                f"${r.cost_per_query_usd:>8.4f}"
            )
        lines += [sep, f"Best quality: {self.best_quality}",
                  f"Best latency: {self.best_latency}",
                  f"Lowest cost: {self.lowest_cost}",
                  f"Best balanced: {self.best_balanced}"]
        return "\n".join(lines)


class ModelComparison:
    """Compare multiple model configurations on the same golden dataset."""

    DEFAULT_WEIGHTS = {
        "quality_score": 0.5,
        "p95_latency_ms": 0.25,
        "cost_per_query_usd": 0.25,
    }

    def compare(
        self,
        run_results: list[Any],  # list of EvaluationRunResult
        weights: dict[str, float] | None = None,
    ) -> ModelComparisonResult:
        weights = weights or self.DEFAULT_WEIGHTS
        rows: list[ModelComparisonRow] = []

        for run in run_results:
            m = run.aggregate_metrics
            quality = (
                m.get("faithfulness", 0) * 0.4
                + m.get("answer_relevance", 0) * 0.3
                + (1 - m.get("hallucination_rate", 0)) * 0.3
            )
            cost_usd = run.total_cost_usd / max(run.total_cases, 1)
            row = ModelComparisonRow(
                model=run.model,
                quality_score=quality,
                faithfulness=m.get("faithfulness", 0),
                answer_relevance=m.get("answer_relevance", 0),
                hallucination_rate=m.get("hallucination_rate", 0),
                p95_latency_ms=m.get("p95_latency_ms", 0),
                cost_per_query_usd=cost_usd,
            )
            rows.append(row)

        if not rows:
            return ModelComparisonResult(rows=[], best_quality="N/A",
                                         best_latency="N/A", lowest_cost="N/A", best_balanced="N/A")

        # Compute weighted scores (normalize each dimension)
        max_quality = max(r.quality_score for r in rows) or 1
        max_latency = max(r.p95_latency_ms for r in rows) or 1
        max_cost = max(r.cost_per_query_usd for r in rows) or 1

        for row in rows:
            row.weighted_score = (
                (row.quality_score / max_quality) * weights.get("quality_score", 0.5)
                + (1 - row.p95_latency_ms / max_latency) * weights.get("p95_latency_ms", 0.25)
                + (1 - row.cost_per_query_usd / max_cost) * weights.get("cost_per_query_usd", 0.25)
            )

        return ModelComparisonResult(
            rows=rows,
            best_quality=max(rows, key=lambda r: r.quality_score).model,
            best_latency=min(rows, key=lambda r: r.p95_latency_ms).model,
            lowest_cost=min(rows, key=lambda r: r.cost_per_query_usd).model,
            best_balanced=max(rows, key=lambda r: r.weighted_score).model,
        )


@dataclass
class RAGGridCell:
    model: str
    rag_version: str
    quality_score: float
    faithfulness: float
    cost_usd: float
    weighted_score: float = 0.0


class RAGComparison:
    """Model × RAG configuration grid comparison."""

    def compare_grid(self, run_results: list[Any]) -> dict[str, Any]:
        """Build a model×RAG grid from multiple run results."""
        grid: dict[str, dict[str, RAGGridCell]] = {}

        for run in run_results:
            m = run.aggregate_metrics
            quality = (
                m.get("faithfulness", 0) * 0.4
                + m.get("answer_relevance", 0) * 0.3
                + (1 - m.get("hallucination_rate", 0)) * 0.3
            )
            cost = run.total_cost_usd / max(run.total_cases, 1)
            cell = RAGGridCell(
                model=run.model,
                rag_version=run.rag_version,
                quality_score=quality,
                faithfulness=m.get("faithfulness", 0),
                cost_usd=cost,
            )
            grid.setdefault(run.model, {})[run.rag_version] = cell

        return {"grid": grid, "best": self._find_best(grid)}

    def _find_best(self, grid: dict) -> dict[str, str]:
        best_quality = ("", "", 0.0)
        best_cost = ("", "", float("inf"))
        for model, rag_versions in grid.items():
            for rag, cell in rag_versions.items():
                if cell.quality_score > best_quality[2]:
                    best_quality = (model, rag, cell.quality_score)
                if cell.cost_usd < best_cost[2]:
                    best_cost = (model, rag, cell.cost_usd)
        return {
            "best_quality": f"{best_quality[0]}×{best_quality[1]}",
            "lowest_cost": f"{best_cost[0]}×{best_cost[1]}",
        }


class ChangeImpactAnalyzer:
    """Breaks down metric changes by dataset category."""

    def analyze(self, baseline_results: list[Any], candidate_results: list[Any]) -> dict[str, Any]:
        """Compare per-category performance between two run result sets."""
        from collections import defaultdict

        def group_by_category(results: list[Any]) -> dict[str, list[float]]:
            cat_scores: dict[str, list[float]] = defaultdict(list)
            for case in results:
                key = getattr(case, "category", "unknown")
                score = sum(m.score for m in case.metrics) / max(len(case.metrics), 1)
                cat_scores[key].append(score)
            return cat_scores

        base_by_cat = group_by_category(baseline_results)
        cand_by_cat = group_by_category(candidate_results)

        all_cats = set(base_by_cat) | set(cand_by_cat)
        impact = {}
        for cat in sorted(all_cats):
            b_scores = base_by_cat.get(cat, [0.0])
            c_scores = cand_by_cat.get(cat, [0.0])
            b_avg = sum(b_scores) / len(b_scores)
            c_avg = sum(c_scores) / len(c_scores)
            delta_pct = ((c_avg - b_avg) / b_avg * 100) if b_avg else 0.0
            impact[cat] = {
                "baseline": round(b_avg, 3),
                "candidate": round(c_avg, 3),
                "delta_pct": round(delta_pct, 1),
            }
        return impact
