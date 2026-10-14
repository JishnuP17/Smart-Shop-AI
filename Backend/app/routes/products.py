"""Product catalog endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pymongo.errors import PyMongoError

from app.models.schemas import (
    ProductCreate,
    ProductCategoriesResponse,
    ProductIndexResponse,
    ProductRecommendationsResponse,
    ProductSearchResponse,
    ProductsResponse,
)
from app.config import settings
from app.services.database import create_product, get_product_by_id, get_product_categories, get_products
from app.services.embeddings import embed_product, product_content_hash
from app.services.semantic_search import index_existing_products, search_products
from app.services.vector_search import (
    ProductNotFound,
    ProductNotIndexed,
    VectorIndexStale,
    VectorIndexUnavailable,
    mark_catalog_stale,
    recommend_products as recommend_indexed_products,
)

router = APIRouter(prefix="/api/products", tags=["products"])


@router.post("", status_code=status.HTTP_201_CREATED)
def add_product(product: ProductCreate) -> dict[str, Any]:
    """Save a product document in MongoDB and return the saved document."""
    try:
        product_data = product.model_dump()
        product_data["_embedding"] = embed_product(product_data)
        product_data["_embedding_model"] = settings.embedding_model
        product_data["_source_content_hash"] = product_content_hash(product_data)
        created_product = create_product(product_data)
        mark_catalog_stale()
        return created_product
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The product database is unavailable or not configured.",
        ) from exc


@router.get("/categories", response_model=ProductCategoriesResponse)
def list_product_categories() -> ProductCategoriesResponse:
    """Return available product categories and counts from MongoDB."""
    try:
        return ProductCategoriesResponse(items=get_product_categories())
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The product database is unavailable or not configured.",
        ) from exc


@router.get("", response_model=ProductsResponse)
def list_products(
    skip: int = Query(default=0, ge=0, description="Number of products to skip."),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum products to return."),
    category: str | None = Query(default=None, min_length=1, max_length=200),
) -> ProductsResponse:
    """Return a page of products from MongoDB, optionally filtered by category."""
    try:
        products = get_products(skip=skip, limit=limit, category=category)
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The product database is unavailable or not configured.",
        ) from exc

    return ProductsResponse(items=products, skip=skip, limit=limit, count=len(products))


@router.get("/search", response_model=ProductSearchResponse)
def semantic_product_search(
    q: str = Query(min_length=2, max_length=500, description="Natural-language product query."),
    limit: int = Query(default=10, ge=1, le=100),
) -> ProductSearchResponse:
    """Find products by semantic similarity to the query text."""
    try:
        query = q.strip()
        if len(query) < 2:
            raise HTTPException(status_code=422, detail="Query must contain at least 2 non-space characters.")
        return ProductSearchResponse(**search_products(query, limit))
    except HTTPException:
        raise
    except (VectorIndexStale, VectorIndexUnavailable) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search is unavailable. Check MongoDB and the embedding model.",
        ) from exc


@router.get("/{product_id}")
def read_product(product_id: str) -> dict[str, Any]:
    """Return one existing product for direct product-page navigation."""
    try:
        product = get_product_by_id(product_id)
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The product database is unavailable or not configured.",
        ) from exc
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product was not found.")
    return product


@router.get("/{product_id}/recommendations", response_model=ProductRecommendationsResponse)
def recommend_products(
    product_id: str,
    limit: int = Query(default=5, ge=1, le=100, description="Maximum recommendations to return."),
) -> ProductRecommendationsResponse:
    """Recommend similar products using the selected product's FAISS vector."""
    try:
        return ProductRecommendationsResponse(**recommend_indexed_products(product_id, limit))
    except ProductNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ProductNotIndexed as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (VectorIndexStale, VectorIndexUnavailable) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Product recommendations are unavailable. Check MongoDB and the FAISS index.",
        ) from exc
@router.post("/index-embeddings", response_model=ProductIndexResponse)
def index_product_embeddings() -> ProductIndexResponse:
    """Build the persistent FAISS index from current products."""
    try:
        return ProductIndexResponse(**index_existing_products())
    except (VectorIndexStale, VectorIndexUnavailable) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (PyMongoError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Product embedding indexing failed. Check MongoDB and the model download.",
        ) from exc
