"""Evaluations API endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.schemas import (
    EvaluationRunCreateSchema,
    EvaluationRunResponseSchema,
    EvaluationRunListSchema,
)
from backend.app.services.evaluation_service import EvaluationService

router = APIRouter()
_service = EvaluationService()


@router.post("/run", response_model=EvaluationRunResponseSchema, status_code=202)
async def trigger_evaluation(body: EvaluationRunCreateSchema):
    """Trigger a new evaluation run.

    Returns the run result. For long runs, consider using a background task.
    """
    try:
        result = await _service.run_evaluation(body)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.get("", response_model=EvaluationRunListSchema)
def list_evaluations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    experiment_id: Optional[str] = None,
):
    """List all evaluation runs with pagination."""
    try:
        return _service.list_runs(page=page, page_size=page_size, experiment_id=experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list runs")


@router.get("/{run_id}", response_model=EvaluationRunResponseSchema)
def get_evaluation(run_id: str):
    """Get details for a specific evaluation run."""
    try:
        run = _service.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return run
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch run")


@router.get("/{run_id}/metrics")
def get_run_metrics(run_id: str):
    """Get all metric results for a specific run."""
    try:
        metrics = _service.get_run_metrics(run_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return {"run_id": run_id, "metrics": metrics}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")
