"""Shared local Sentence Transformer model and normalized product embeddings."""

from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

EMBEDDING_DIMENSIONS = 384
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Load and reuse the configured pretrained model in this process."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model = SentenceTransformer(settings.embedding_model)
                actual_dimensions = model.get_sentence_embedding_dimension()
                if actual_dimensions != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Configured embedding model has {actual_dimensions} dimensions; expected {EMBEDDING_DIMENSIONS}."
                    )
                _model = model
    return _model


def product_text(product: dict[str, Any]) -> str:
    """Build embedding text from available product text and genuine features."""
    labels = ("name", "description", "category", "brand", "features")
    parts: list[str] = []
    for label in labels:
        value = product.get(label)
        if isinstance(value, (list, tuple)):
            text = " ".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip() if value is not None else ""
        if text:
            parts.append(f"{label}: {text}")
    return " ".join(parts)


def product_content_hash(product: dict[str, Any]) -> str:
    """Hash the exact semantic text to detect whether an embedding is reusable."""
    return hashlib.sha256(product_text(product).encode("utf-8")).hexdigest()


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode and L2-normalize texts, verifying shape and finite vector values."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
    vectors = np.asarray(
        get_embedding_model().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.shape != (len(texts), EMBEDDING_DIMENSIONS):
        raise ValueError(f"Embedding model returned unexpected shape {vectors.shape}.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned a non-finite vector.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector.")
    vectors /= norms[:, None]
    return vectors


def embed_text(text: str) -> list[float]:
    """Encode one query as a normalized 384-dimensional vector."""
    return embed_texts([text], batch_size=1)[0].tolist()


def embed_product(product: dict[str, Any]) -> list[float]:
    """Create a normalized embedding from a product's available search fields."""
    text = product_text(product)
    if not text:
        raise ValueError("Product needs at least one of name, description, category, or brand.")
    return embed_text(text)
