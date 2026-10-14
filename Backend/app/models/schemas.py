"""Response schemas for the API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HealthResponse(BaseModel):
    status: str


class DatabaseHealthResponse(BaseModel):
    status: str
    database: str


class ProductsResponse(BaseModel):
    items: list[dict[str, Any]]
    skip: int
    limit: int
    count: int


class ProductCategory(BaseModel):
    name: str
    count: int


class ProductCategoriesResponse(BaseModel):
    items: list[ProductCategory]


class ProductCreate(BaseModel):
    """Fields accepted when adding a product to the catalog."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, description="Product name.")
    price: float = Field(ge=0, description="Product price; must be zero or greater.")
    description: str | None = None
    category: str | None = None
    brand: str | None = None
    image_url: str | None = None


class ProductSearchHit(BaseModel):
    product: dict[str, Any]
    score: float


class ProductSearchResponse(BaseModel):
    query: str
    results: list[ProductSearchHit]
    indexed_products: int


class ProductRecommendationHit(BaseModel):
    product: dict[str, Any]
    score: float


class ProductRecommendationsResponse(BaseModel):
    product_id: str
    recommendations: list[ProductRecommendationHit]
    indexed_products: int


class ProductIndexResponse(BaseModel):
    indexed: int
    skipped: int
    model: str


class DemoUser(BaseModel):
    id: str
    name: str
    description: str
    interests: list[str]


class DemoUsersResponse(BaseModel):
    items: list[DemoUser]


class InteractionCreate(BaseModel):
    event_id: str = Field(min_length=1, max_length=80)
    event_type: Literal["view", "click", "search"]
    product_id: str | None = Field(default=None, min_length=1, max_length=64)
    query: str | None = Field(default=None, max_length=500)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_event_fields(self) -> "InteractionCreate":
        if self.event_type == "search":
            if not self.query or not self.query.strip() or self.product_id is not None:
                raise ValueError("Search events require a non-empty query and no product ID.")
            self.query = self.query.strip()
        elif self.product_id is None or self.query is not None:
            raise ValueError("View and click events require a product ID and no query.")
        return self


class InteractionResponse(BaseModel):
    event_id: str
    user_id: str
    event_type: Literal["view", "click", "search"]
    product_id: str | None = None
    query: str | None = None
    timestamp: datetime
    recorded: bool


class InteractionHistoryResponse(BaseModel):
    user_id: str
    items: list[InteractionResponse]


class PersonalizedRecommendationHit(BaseModel):
    product: dict[str, Any]
    score: float


class PersonalizedRecommendationsResponse(BaseModel):
    user_id: str
    recommendations: list[PersonalizedRecommendationHit]
    indexed_products: int
    based_on: Literal["initial_interests", "interests_and_interactions"]
