"""Health check endpoints."""

from fastapi import APIRouter, HTTPException, status
from pymongo.errors import PyMongoError

from app.models.schemas import DatabaseHealthResponse, HealthResponse
from app.services.database import check_database_connection, get_database_name

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report that the API process is responding."""
    return HealthResponse(status="ok")


@router.get("/db", response_model=DatabaseHealthResponse)
def database_health_check() -> DatabaseHealthResponse:
    """Check that MongoDB responds to a ping command."""
    try:
        check_database_connection()
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB is unavailable or not configured.",
        ) from exc

    return DatabaseHealthResponse(status="ok", database=get_database_name())
