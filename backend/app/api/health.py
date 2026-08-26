"""Health endpoint."""
from fastapi import APIRouter
from backend.app.schemas.schemas import HealthResponseSchema
from backend.app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponseSchema)
def health_check():
    """Health check — used by Azure Container Apps and CI/CD to verify deployment."""
    return HealthResponseSchema(
        status="ok",
        version="1.0.0",
        database="connected",
        storage=settings.storage_backend,
    )
