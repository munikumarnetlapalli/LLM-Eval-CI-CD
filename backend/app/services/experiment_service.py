"""Experiment service."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from backend.app.schemas.schemas import ExperimentCreateSchema, ExperimentResponseSchema
from evaluator.storage.backends import build_storage_backend


class ExperimentService:
    def __init__(self):
        self._storage = build_storage_backend()

    def create(self, body: ExperimentCreateSchema) -> ExperimentResponseSchema:
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        ts = datetime.now(timezone.utc).isoformat()
        data = {
            "experiment_id": exp_id,
            "name": body.name,
            "description": body.description,
            "created_at": ts,
            "tags": body.tags,
        }
        self._storage.save(f"experiments/{exp_id}", data)
        return ExperimentResponseSchema(**data)

    def list_all(self) -> list[ExperimentResponseSchema]:
        keys = self._storage.list_keys(prefix="experiments/")
        results = []
        for key in keys:
            try:
                data = self._storage.load(key)
                results.append(ExperimentResponseSchema(**data))
            except Exception:
                continue
        return results
