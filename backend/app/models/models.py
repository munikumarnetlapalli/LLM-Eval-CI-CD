"""SQLAlchemy ORM models for the evaluation platform."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    Text, ForeignKey, JSON
)
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


def now_utc():
    return datetime.now(timezone.utc)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, unique=True, nullable=False, index=True)
    experiment_id = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    model_version = Column(String, default="")
    prompt_version = Column(String, default="v1")
    rag_version = Column(String, default="v1")
    kb_version = Column(String, default="KB-001")
    dataset_version = Column(String, default="v1")
    git_sha = Column(String, default="unknown")
    timestamp = Column(DateTime(timezone=True), default=now_utc)
    environment = Column(String, default="local")
    total_cases = Column(Integer, default=0)
    passed_cases = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    pass_rate = Column(Float, default=0.0)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    quality_gate_result = Column(String, nullable=True)  # "PASSED" | "FAILED" | None
    aggregate_metrics = Column(JSON, default=dict)

    metric_results = relationship("MetricResult", back_populates="run", cascade="all, delete-orphan")


class MetricResult(Base):
    __tablename__ = "metric_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, ForeignKey("evaluation_runs.run_id"), nullable=False, index=True)
    metric = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    threshold = Column(Float, default=0.0)
    passed = Column(Boolean, default=True)
    explanation = Column(Text, default="")
    category = Column(String, default="deterministic")  # deterministic | model_based | retrieval_based

    run = relationship("EvaluationRun", back_populates="metric_results")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=now_utc)
    tags = Column(JSON, default=list)


class BaselineRecord(Base):
    __tablename__ = "baseline_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    baseline_id = Column(String, unique=True, nullable=False, index=True)
    run_id = Column(String, nullable=False)
    model = Column(String, default="")
    rag_version = Column(String, default="v1")
    dataset_version = Column(String, default="v1")
    timestamp = Column(DateTime(timezone=True), default=now_utc)
    metrics = Column(JSON, default=dict)
    notes = Column(Text, default="")
    is_active = Column(Boolean, default=True)
