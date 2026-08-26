"""Evaluation service — business logic for running and querying evaluations."""
from __future__ import annotations

import asyncio
from typing import Optional

from backend.app.schemas.schemas import (
    EvaluationRunCreateSchema,
    EvaluationRunResponseSchema,
    EvaluationRunListSchema,
)
from evaluator.storage.backends import build_storage_backend


class EvaluationService:
    def __init__(self):
        self._storage = build_storage_backend()

    async def run_evaluation(self, body: EvaluationRunCreateSchema) -> EvaluationRunResponseSchema:
        """Run a full evaluation and persist results."""
        from evaluator.providers.factory import build_provider
        from evaluator.targets.mock import MockTarget
        from evaluator.datasets.loader import DatasetLoader
        from evaluator.runners.runner import EvaluationRunner
        import yaml
        from pathlib import Path

        # Load config
        config_path = Path(body.config_path)
        if not config_path.exists():
            raise ValueError(f"Config not found: {body.config_path}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Build provider + target
        provider = build_provider(cfg.get("model", {"provider": "mock"}))
        target = MockTarget()  # Defaults to mock; HTTP target set by config

        # Load dataset
        loader = DatasetLoader()
        dataset_path = cfg.get("dataset", {}).get("path", "datasets/v1/golden.json")
        dataset = loader.load(dataset_path)

        runner = EvaluationRunner(target=target, judge_provider=provider, config=cfg)

        # Run in thread to avoid blocking the event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: runner.run(
                dataset=dataset,
                experiment_id=body.experiment_id,
                environment=body.environment,
            )
        )

        # Persist
        self._storage.save(f"runs/{result.run_id}", result.to_dict())

        return EvaluationRunResponseSchema(
            run_id=result.run_id,
            experiment_id=result.experiment_id,
            model=result.model,
            model_version=result.model_version,
            prompt_version=result.prompt_version,
            rag_version=result.rag_version,
            kb_version=result.kb_version,
            dataset_version=result.dataset_version,
            git_sha=result.git_sha,
            timestamp=result.timestamp,
            environment=result.environment,
            total_cases=result.total_cases,
            passed_cases=result.passed_cases,
            error_count=result.error_count,
            pass_rate=result.pass_rate,
            total_cost_usd=result.total_cost_usd,
            quality_gate_result=result.quality_gate_result,
            aggregate_metrics=result.aggregate_metrics,
        )

    def list_runs(
        self,
        page: int = 1,
        page_size: int = 20,
        experiment_id: Optional[str] = None,
    ) -> EvaluationRunListSchema:
        all_keys = self._storage.list_keys(prefix="runs/")
        runs = []
        for key in all_keys:
            try:
                data = self._storage.load(key)
                if experiment_id and data.get("experiment_id") != experiment_id:
                    continue
                runs.append(EvaluationRunResponseSchema(**data))
            except Exception:
                continue

        total = len(runs)
        start = (page - 1) * page_size
        end = start + page_size
        return EvaluationRunListSchema(
            runs=runs[start:end],
            total=total,
            page=page,
            page_size=page_size,
        )

    def get_run(self, run_id: str) -> Optional[EvaluationRunResponseSchema]:
        try:
            data = self._storage.load(f"runs/{run_id}")
            return EvaluationRunResponseSchema(**data)
        except FileNotFoundError:
            return None

    def get_run_metrics(self, run_id: str) -> Optional[list]:
        try:
            data = self._storage.load(f"runs/{run_id}")
            return data.get("aggregate_metrics", {})
        except FileNotFoundError:
            return None
