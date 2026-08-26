"""Quality gate service."""
from __future__ import annotations
from typing import Optional
from evaluator.storage.backends import build_storage_backend
from evaluator.regression.quality_gate import QualityGate
from backend.app.schemas.schemas import QualityGateResponseSchema

DEFAULT_GATE_CONFIG = {
    "faithfulness": {"minimum": 0.90},
    "answer_relevance": {"minimum": 0.80},
    "hallucination_rate": {"maximum": 0.05},
    "context_relevance": {"minimum": 0.80},
    "p95_latency_ms": {"maximum_ms": 1000},
    "cost_per_query": {"maximum_usd": 0.02},
}


class GateService:
    def __init__(self):
        self._storage = build_storage_backend()

    def evaluate(self, run_id: str, gate_config: Optional[dict] = None) -> QualityGateResponseSchema:
        data = self._storage.load(f"runs/{run_id}")
        metrics = data.get("aggregate_metrics", {})
        cfg = gate_config or DEFAULT_GATE_CONFIG
        qg = QualityGate(cfg)
        result = qg.evaluate(run_id, metrics)
        return QualityGateResponseSchema(**result.to_dict())
