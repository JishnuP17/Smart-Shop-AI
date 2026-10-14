"""Demo profile, interaction, and personalized recommendation endpoints."""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pymongo.errors import PyMongoError

from app.models.schemas import (
    DemoUsersResponse,
    InteractionCreate,
    InteractionHistoryResponse,
    InteractionResponse,
    PersonalizedRecommendationsResponse,
)
from app.services.database import (
    get_demo_interactions,
    get_demo_user,
    get_demo_users,
    get_product_by_id,
    save_demo_interaction,
)
from app.services.personalization import DEMO_PROFILES, DEMO_USER_IDS, get_personalized_recommendations
from app.services.vector_search import VectorIndexStale, VectorIndexUnavailable

router = APIRouter(prefix="/api/demo/users", tags=["demo personalization"])


@router.get("", response_model=DemoUsersResponse)
def list_demo_users() -> DemoUsersResponse:
    """List the three active demo profiles."""
    try:
        return DemoUsersResponse(items=get_demo_users(DEMO_PROFILES))
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Demo profiles are unavailable.") from exc


@router.get("/{user_id}/recommendations", response_model=PersonalizedRecommendationsResponse)
def personalized_recommendations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=50),
) -> PersonalizedRecommendationsResponse:
    """Return preference-based recommendations for one demo profile."""
    try:
        result = get_personalized_recommendations(user_id, limit)
        return PersonalizedRecommendationsResponse(**result)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (VectorIndexStale, VectorIndexUnavailable) as exc:
        raise HTTPException(status_code=503, detail="Personalized recommendations are unavailable.") from exc
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Personalized recommendations are unavailable.") from exc


@router.post("/{user_id}/interactions", response_model=InteractionResponse, status_code=201)
def record_demo_interaction(user_id: str, interaction: InteractionCreate) -> InteractionResponse:
    """Record a demo profile's view, click, or search event idempotently."""
    try:
        if user_id not in DEMO_USER_IDS:
            raise HTTPException(status_code=404, detail="Demo profile was not found.")
        if get_demo_user(user_id) is None:
            raise HTTPException(status_code=404, detail="Demo profile was not found.")
        if interaction.product_id is not None and get_product_by_id(interaction.product_id) is None:
            raise HTTPException(status_code=404, detail="Product was not found.")
        saved, inserted = save_demo_interaction(
            {
                "user_id": user_id,
                **interaction.model_dump(exclude_none=True),
                "timestamp": datetime.now(timezone.utc),
            }
        )
        return InteractionResponse(
            event_id=saved["event_id"],
            user_id=saved["user_id"],
            event_type=saved["event_type"],
            product_id=saved.get("product_id"),
            query=saved.get("query"),
            timestamp=saved["timestamp"],
            recorded=inserted,
        )
    except HTTPException:
        raise
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="The interaction could not be recorded.") from exc


@router.get("/{user_id}/interactions", response_model=InteractionHistoryResponse)
def demo_interaction_history(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> InteractionHistoryResponse:
    """Read recent interaction history scoped to one demo profile."""
    try:
        if user_id not in DEMO_USER_IDS:
            raise HTTPException(status_code=404, detail="Demo profile was not found.")
        if get_demo_user(user_id) is None:
            raise HTTPException(status_code=404, detail="Demo profile was not found.")
        items: list[dict[str, Any]] = []
        for event in get_demo_interactions(user_id, limit):
            items.append(
                {
                    "event_id": event["event_id"],
                    "user_id": event["user_id"],
                    "event_type": event["event_type"],
                    "product_id": event.get("product_id"),
                    "query": event.get("query"),
                    "timestamp": event["timestamp"],
                    "recorded": True,
                }
            )
        return InteractionHistoryResponse(user_id=user_id, items=items)
    except HTTPException:
        raise
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Interaction history is unavailable.") from exc
