"""Compatibility façade for the shared embedding and FAISS search services."""

from app.services.embeddings import (
    EMBEDDING_DIMENSIONS,
    embed_product,
    embed_text,
    embed_texts,
    get_embedding_model,
    product_content_hash,
    product_text,
)
from app.services.vector_search import rebuild_faiss_index, search_products


def index_existing_products() -> dict[str, int | str]:
    """Rebuild the persistent vector index for the full current catalog."""
    return rebuild_faiss_index()


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "embed_product",
    "embed_text",
    "embed_texts",
    "get_embedding_model",
    "index_existing_products",
    "product_content_hash",
    "product_text",
    "rebuild_faiss_index",
    "search_products",
]
