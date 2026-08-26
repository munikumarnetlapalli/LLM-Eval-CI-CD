"""Pydantic schemas for API request/response validation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Evaluation Run Schemas ───────────────────────────────────────────────────

class RunMetricSchema(BaseModel):
    metric: str
    score: float
    threshold: float
    passed: bool
    explanation: str
    category: str = "deterministic"


class EvaluationRunCreateSchema(BaseModel):
    config_path: str = Field(default="configs/evaluation.yaml")
    experiment_id: Optional[str] = None
    environment: str = Field(default="local")
    set_baseline: bool = False


class EvaluationRunResponseSchema(BaseModel):
    run_id: str
    experiment_id: str
    model: str
    model_version: str = ""
    prompt_version: str = "v1"
    rag_version: str = "v1"
    kb_version: str = "KB-001"
    dataset_version: str = "v1"
    git_sha: str = "unknown"
    timestamp: datetime | str
    environment: str = "local"
    total_cases: int
    passed_cases: int
    error_count: int
    pass_rate: float
    total_cost_usd: float
    quality_gate_result: Optional[str] = None
    aggregate_metrics: dict[str, float] = {}

    class Config:
        from_attributes = True


class EvaluationRunListSchema(BaseModel):
    runs: list[EvaluationRunResponseSchema]
    total: int
    page: int = 1
    page_size: int = 20


# ─── Experiment Schemas ───────────────────────────────────────────────────────

class ExperimentCreateSchema(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    tags: list[str] = []


class ExperimentResponseSchema(BaseModel):
    experiment_id: str
    name: str
    description: str
    created_at: datetime | str
    tags: list[str] = []

    class Config:
        from_attributes = True


# ─── Comparison Schemas ───────────────────────────────────────────────────────

class CompareRequestSchema(BaseModel):
    run_ids: list[str] = Field(..., min_length=2, description="List of run IDs to compare")
    weights: Optional[dict[str, float]] = None


class CompareResponseSchema(BaseModel):
    best_quality: str
    best_latency: str
    lowest_cost: str
    best_balanced: str
    rows: list[dict[str, Any]]


# ─── Quality Gate Schemas ─────────────────────────────────────────────────────

class QualityGateRequestSchema(BaseModel):
    run_id: str
    gate_config: Optional[dict[str, Any]] = None


class QualityGateResponseSchema(BaseModel):
    run_id: str
    verdict: str  # "PASSED" | "FAILED"
    passed: bool
    failing_rules: list[dict[str, Any]] = []
    total_rules: int = 0
    failing_count: int = 0


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponseSchema(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    database: str = "connected"
    storage: str = "local"
