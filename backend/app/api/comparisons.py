"""Comparisons and quality gate API endpoints."""
from fastapi import APIRouter, HTTPException
from backend.app.schemas.schemas import (
    CompareRequestSchema, CompareResponseSchema,
    QualityGateRequestSchema, QualityGateResponseSchema,
)
from backend.app.services.comparison_service import ComparisonService
from backend.app.services.gate_service import GateService

router = APIRouter()
_compare_svc = ComparisonService()
_gate_svc = GateService()


@router.post("/compare", response_model=CompareResponseSchema)
def compare_runs(body: CompareRequestSchema):
    """Compare multiple evaluation runs side by side."""
    try:
        return _compare_svc.compare(body.run_ids, body.weights)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Comparison failed")


@router.post("/quality-gate", response_model=QualityGateResponseSchema)
def evaluate_gate(body: QualityGateRequestSchema):
    """Evaluate quality gate rules for a run."""
    try:
        return _gate_svc.evaluate(body.run_id, body.gate_config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Quality gate evaluation failed")


@router.get("/models")
def list_models():
    """List supported model providers."""
    return {
        "providers": [
            {"id": "mock", "name": "Mock (testing)", "type": "mock"},
            {"id": "openai", "name": "OpenAI", "type": "api"},
            {"id": "azure_openai", "name": "Azure OpenAI", "type": "api"},
            {"id": "gemini", "name": "Google Gemini", "type": "api"},
        ]
    }
