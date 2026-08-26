"""Comparison service."""
from __future__ import annotations
from typing import Optional
from evaluator.storage.backends import build_storage_backend
from evaluator.comparison.comparison import ModelComparison
from backend.app.schemas.schemas import CompareResponseSchema


class ComparisonService:
    def __init__(self):
        self._storage = build_storage_backend()

    def compare(self, run_ids: list[str], weights: Optional[dict] = None) -> CompareResponseSchema:
        if len(run_ids) < 2:
            raise ValueError("At least 2 run IDs are required for comparison")

        runs_data = []
        for run_id in run_ids:
            data = self._storage.load(f"runs/{run_id}")
            runs_data.append(data)

        # Build comparison rows from stored data
        from dataclasses import dataclass
        @dataclass
        class _FakeRun:
            model: str
            rag_version: str
            aggregate_metrics: dict
            total_cost_usd: float
            total_cases: int

        fake_runs = [
            _FakeRun(
                model=d.get("model", run_ids[i]),
                rag_version=d.get("rag_version", "v1"),
                aggregate_metrics=d.get("aggregate_metrics", {}),
                total_cost_usd=d.get("total_cost_usd", 0.0),
                total_cases=d.get("total_cases", 1),
            )
            for i, d in enumerate(runs_data)
        ]

        mc = ModelComparison()
        result = mc.compare(fake_runs, weights)

        return CompareResponseSchema(
            best_quality=result.best_quality,
            best_latency=result.best_latency,
            lowest_cost=result.lowest_cost,
            best_balanced=result.best_balanced,
            rows=[
                {
                    "model": r.model,
                    "quality_score": round(r.quality_score, 4),
                    "faithfulness": round(r.faithfulness, 4),
                    "answer_relevance": round(r.answer_relevance, 4),
                    "hallucination_rate": round(r.hallucination_rate, 4),
                    "p95_latency_ms": round(r.p95_latency_ms, 1),
                    "cost_per_query_usd": round(r.cost_per_query_usd, 6),
                    "weighted_score": round(r.weighted_score, 4),
                }
                for r in result.rows
            ],
        )
