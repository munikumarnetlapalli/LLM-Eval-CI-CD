"""Experiments API endpoints."""
from fastapi import APIRouter, HTTPException
from backend.app.schemas.schemas import ExperimentCreateSchema, ExperimentResponseSchema
from backend.app.services.experiment_service import ExperimentService

router = APIRouter()
_service = ExperimentService()


@router.post("", response_model=ExperimentResponseSchema, status_code=201)
def create_experiment(body: ExperimentCreateSchema):
    """Create a new experiment to group evaluation runs."""
    try:
        return _service.create(body)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to create experiment")


@router.get("", response_model=list[ExperimentResponseSchema])
def list_experiments():
    """List all experiments."""
    try:
        return _service.list_all()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to list experiments")
