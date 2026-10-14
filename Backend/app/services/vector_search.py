"""Persistent, ID-mapped FAISS cosine index for MongoDB products."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from bson import ObjectId
from filelock import FileLock

from app.config import settings
from app.services.database import (
    _serialize_document,
    finalize_vector_migration,
    get_products_by_id_refs,
    iter_catalog_products,
)
from app.services.embeddings import (
    EMBEDDING_DIMENSIONS,
    embed_text,
    embed_texts,
    product_content_hash,
    product_text,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FAISS_ROOT = BACKEND_ROOT / "data" / "faiss"
GENERATIONS_ROOT = FAISS_ROOT / "generations"
CURRENT_FILE = FAISS_ROOT / "CURRENT"
LOCK_FILE = FAISS_ROOT / ".write.lock"
INDEX_FILENAME = "products.faiss"
MAPPING_FILENAME = "mapping.json"
MANIFEST_FILENAME = "manifest.json"
_GENERATION_PATTERN = re.compile(r"^generation-[0-9a-f]{32}$")
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


class VectorIndexUnavailable(RuntimeError):
    """The persistent index is missing or cannot be validated."""


class VectorIndexStale(RuntimeError):
    """The product catalogue changed since its vectors were indexed."""


@dataclass
class IndexBundle:
    generation: str
    index: Any
    mapping: list[dict[str, Any]]
    manifest: dict[str, Any]
    id_keys: set[str]


_loaded_bundle: IndexBundle | None = None
_index_lock = threading.Lock()
_catalog_validation_lock = threading.Lock()
_catalog_validated = False
_catalog_stale = False


def _mongo_id_ref(product_id: Any) -> dict[str, str]:
    if isinstance(product_id, ObjectId):
        return {"product_id": str(product_id), "id_type": "objectid"}
    if isinstance(product_id, str):
        return {"product_id": product_id, "id_type": "string"}
    raise ValueError(f"Unsupported MongoDB product ID type: {type(product_id).__name__}.")


def _ref_key(id_ref: dict[str, str]) -> str:
    return f"{id_ref['id_type']}:{id_ref['product_id']}"


def _catalog_digest_update(digest: Any, product: dict[str, Any]) -> str:
    text = product_text(product)
    payload = json.dumps(
        [str(product["_id"]), text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest.update(payload.encode("utf-8"))
    digest.update(b"\n")
    return text


def catalog_signature() -> str:
    """Hash product identities and semantic text to detect stale vectors."""
    digest = hashlib.sha256()
    for product in iter_catalog_products(batch_size=512, include_vectors=False):
        _catalog_digest_update(digest, product)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
        output.flush()
        os.fsync(output.fileno())


def _publish_generation(index: Any, mapping: list[dict[str, Any]], signature: str) -> IndexBundle:
    FAISS_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    generation = f"generation-{uuid.uuid4().hex}"
    staging = GENERATIONS_ROOT / f".tmp-{uuid.uuid4().hex}"
    final_directory = GENERATIONS_ROOT / generation
    staging.mkdir()

    manifest = {
        "generation": generation,
        "model": settings.embedding_model,
        "dimensions": EMBEDDING_DIMENSIONS,
        "vector_count": int(index.ntotal),
        "catalog_signature": signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        faiss.write_index(index, str(staging / INDEX_FILENAME))
        _write_json(staging / MAPPING_FILENAME, mapping)
        _write_json(staging / MANIFEST_FILENAME, manifest)
        os.replace(staging, final_directory)

        pointer_temp = FAISS_ROOT / f".CURRENT-{uuid.uuid4().hex}.tmp"
        with pointer_temp.open("w", encoding="utf-8") as pointer:
            pointer.write(generation)
            pointer.flush()
            os.fsync(pointer.fileno())
        os.replace(pointer_temp, CURRENT_FILE)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return IndexBundle(
        generation,
        index,
        mapping,
        manifest,
        {_ref_key(entry["mongo_id"]) for entry in mapping},
    )


def _load_generation(generation: str) -> IndexBundle:
    directory = GENERATIONS_ROOT / generation
    try:
        with (directory / MANIFEST_FILENAME).open(encoding="utf-8") as source:
            manifest = json.load(source)
        with (directory / MAPPING_FILENAME).open(encoding="utf-8") as source:
            mapping = json.load(source)
        index = faiss.read_index(str(directory / INDEX_FILENAME))
    except Exception as exc:
        raise VectorIndexUnavailable(
            "The saved FAISS index files are unreadable. Rebuild with `python -m app.services.vector_search --rebuild`."
        ) from exc

    if manifest.get("generation") != generation:
        raise VectorIndexUnavailable("FAISS manifest and generation do not match; rebuild the index.")
    if manifest.get("dimensions") != EMBEDDING_DIMENSIONS or index.d != EMBEDDING_DIMENSIONS:
        raise VectorIndexUnavailable("FAISS index dimension is incompatible with the configured embedding model.")
    if manifest.get("model") != settings.embedding_model:
        raise VectorIndexStale("Embedding model changed; rebuild the FAISS index before searching.")
    if index.ntotal != len(mapping) or index.ntotal != manifest.get("vector_count"):
        raise VectorIndexUnavailable("FAISS index and ID mapping have different sizes; rebuild the index.")

    seen_positions: set[int] = set()
    seen_ids: set[str] = set()
    for entry in mapping:
        position = entry.get("vector_position")
        id_ref = entry.get("mongo_id")
        if (
            not isinstance(position, int)
            or position < 0
            or position >= len(mapping)
            or position in seen_positions
            or not isinstance(id_ref, dict)
            or id_ref.get("id_type") not in {"objectid", "string"}
            or not isinstance(id_ref.get("product_id"), str)
        ):
            raise VectorIndexUnavailable("FAISS mapping contains an invalid vector-position entry.")
        key = _ref_key(id_ref)
        if key in seen_ids:
            raise VectorIndexUnavailable("FAISS mapping contains duplicate MongoDB product IDs.")
        seen_positions.add(position)
        seen_ids.add(key)
    if seen_positions != set(range(len(mapping))):
        raise VectorIndexUnavailable("FAISS mapping has missing vector positions; rebuild the index.")

    return IndexBundle(
        generation,
        index,
        mapping,
        manifest,
        {_ref_key(entry["mongo_id"]) for entry in mapping},
    )


def load_current_index(force_reload: bool = False) -> IndexBundle:
    """Load and validate the immutable generation named by CURRENT."""
    global _loaded_bundle
    if not CURRENT_FILE.exists():
        raise VectorIndexUnavailable(
            "FAISS index is missing. Build it with `python -m app.services.vector_search --rebuild`."
        )
    generation = CURRENT_FILE.read_text(encoding="utf-8").strip()
    if not _GENERATION_PATTERN.fullmatch(generation):
        raise VectorIndexUnavailable("FAISS CURRENT pointer is invalid; rebuild the index.")
    if not force_reload and _loaded_bundle is not None and _loaded_bundle.generation == generation:
        return _loaded_bundle
    with _index_lock:
        if not force_reload and _loaded_bundle is not None and _loaded_bundle.generation == generation:
            return _loaded_bundle
        _loaded_bundle = _load_generation(generation)
    return _loaded_bundle


def validate_catalog_against_index() -> bool:
    """Validate the current catalog outside request handling and cache the result."""
    global _catalog_validated, _catalog_stale
    bundle = load_current_index()
    signature = catalog_signature()
    with _catalog_validation_lock:
        _catalog_validated = True
        _catalog_stale = signature != bundle.manifest.get("catalog_signature")
        return not _catalog_stale


def mark_catalog_stale() -> None:
    """Invalidate search after an API write that is not in the active FAISS map."""
    global _catalog_validated, _catalog_stale
    with _catalog_validation_lock:
        _catalog_validated = True
        _catalog_stale = True


def _ensure_catalog_validated() -> None:
    """Perform one lazy check for callers that did not start through FastAPI."""
    global _catalog_validated, _catalog_stale
    with _catalog_validation_lock:
        if _catalog_validated:
            if _catalog_stale:
                raise VectorIndexStale(
                    "The product catalogue changed after indexing. Rebuild with `python -m app.services.vector_search --rebuild`."
                )
            return
        bundle = load_current_index()
        signature = catalog_signature()
        _catalog_validated = True
        _catalog_stale = signature != bundle.manifest.get("catalog_signature")
        if _catalog_stale:
            raise VectorIndexStale(
                "The product catalogue changed after indexing. Rebuild with `python -m app.services.vector_search --rebuild`."
            )


def has_product_vector(product_id: str) -> bool:
    """Return whether a persisted mapping already contains this ObjectId."""
    try:
        bundle = load_current_index()
    except VectorIndexUnavailable:
        return False
    return f"objectid:{product_id}" in bundle.id_keys


def _validate_or_normalize(vector: Any) -> np.ndarray | None:
    try:
        result = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(result).all():
        return None
    norm = float(np.linalg.norm(result))
    if norm == 0:
        return None
    return result / norm


def rebuild_faiss_index(
    batch_size: int = 32,
    cleanup_mongo_embeddings: bool = True,
) -> dict[str, int | str]:
    """Build, validate, and atomically publish a FAISS inner-product snapshot.

    Set ``cleanup_mongo_embeddings`` to False to keep MongoDB documents unchanged.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    mapping: list[dict[str, Any]] = []
    prepared: list[tuple[dict[str, Any], str, np.ndarray | None]] = []
    digest = hashlib.sha256()
    indexed = 0
    skipped = 0

    for product in iter_catalog_products(batch_size=max(64, batch_size), include_vectors=True):
        text = _catalog_digest_update(digest, product)
        content_hash = product_content_hash(product)
        if not text:
            skipped += 1
            continue

        vector = None
        if (
            product.get("_source_content_hash") == content_hash
            and product.get("_embedding_model") == settings.embedding_model
        ):
            vector = _validate_or_normalize(product.get("_embedding"))

        # Batch documents for model encoding while retaining catalog order.
        prepared.append((product, content_hash, vector))

        if len(prepared) >= batch_size:
            indexed += _add_prepared(prepared, index, mapping, batch_size)
            prepared.clear()

    if prepared:
        indexed += _add_prepared(prepared, index, mapping, batch_size)
        prepared.clear()

    signature = digest.hexdigest()
    if catalog_signature() != signature:
        raise VectorIndexStale("The catalogue changed while the FAISS index was being built; rerun the rebuild command.")

    FAISS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(LOCK_FILE), timeout=60):
            bundle = _publish_generation(index, mapping, signature)
    except Exception as exc:
        if isinstance(exc, (VectorIndexUnavailable, VectorIndexStale)):
            raise
        raise VectorIndexUnavailable(f"Could not persist the FAISS index ({type(exc).__name__}).") from exc

    global _loaded_bundle
    _loaded_bundle = bundle
    global _catalog_validated, _catalog_stale
    with _catalog_validation_lock:
        _catalog_validated = True
        _catalog_stale = False

    if cleanup_mongo_embeddings:
        # Remove redundant Mongo vector arrays only after the durable index is published.
        cleanup_batch: list[dict[str, Any]] = []
        for product in iter_catalog_products(batch_size=256, include_vectors=False):
            cleanup_batch.append(
                {
                    "_id": product["_id"],
                    "_source_content_hash": product_content_hash(product),
                    **{field: product[field] for field in ("name", "description", "category", "brand") if field in product},
                }
            )
            if len(cleanup_batch) >= 256:
                finalize_vector_migration(cleanup_batch)
                cleanup_batch.clear()
        if cleanup_batch:
            finalize_vector_migration(cleanup_batch)

    return {"indexed": indexed, "skipped": skipped, "model": settings.embedding_model}


def _add_prepared(
    prepared: list[tuple[dict[str, Any], str, np.ndarray | None]],
    index: Any,
    mapping: list[dict[str, Any]],
    batch_size: int,
) -> int:
    texts = [product_text(product) for product, _, vector in prepared if vector is None]
    new_vectors = iter(embed_texts(texts, batch_size=batch_size))
    vectors: list[np.ndarray] = []
    for product, _, vector in prepared:
        current_vector = vector if vector is not None else next(new_vectors)
        current_vector = _validate_or_normalize(current_vector)
        if current_vector is None:
            raise ValueError(f"Could not create a valid embedding for MongoDB product {product['_id']}.")
        vectors.append(current_vector)

    matrix = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    start_position = index.ntotal
    index.add(matrix)
    for offset, (product, _, _) in enumerate(prepared):
        id_ref = _mongo_id_ref(product["_id"])
        mapping.append(
            {
                "vector_position": start_position + offset,
                "mongo_id": id_ref,
                "source_dataset": product.get("source_dataset"),
                "source_product_id": product.get("source_product_id"),
            }
        )
    return len(prepared)


def search_products(query: str, limit: int) -> dict[str, Any]:
    """Search the persistent FAISS index and fetch ranked current MongoDB records."""
    request_started = time.perf_counter()
    stage: dict[str, float] = {}
    mark = time.perf_counter()
    bundle = load_current_index()
    stage["index_load_s"] = time.perf_counter() - mark
    mark = time.perf_counter()
    _ensure_catalog_validated()
    stage["catalog_validation_s"] = time.perf_counter() - mark
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if bundle.index.ntotal == 0:
        return {"query": query, "results": [], "indexed_products": 0}

    mark = time.perf_counter()
    query_vector = np.asarray(embed_text(query), dtype=np.float32).reshape(1, -1)
    stage["query_embedding_s"] = time.perf_counter() - mark
    if query_vector.shape != (1, EMBEDDING_DIMENSIONS) or not np.isfinite(query_vector).all():
        raise ValueError("Query embedding is invalid or has the wrong dimensions.")
    faiss.normalize_L2(query_vector)
    # Ask for extra semantic candidates so explicit gender conflicts can be
    # filtered without returning a short page when there are enough matches.
    candidate_limit = min(max(limit * 10, 50), bundle.index.ntotal)
    mark = time.perf_counter()
    scores, positions = bundle.index.search(query_vector, candidate_limit)
    stage["faiss_search_s"] = time.perf_counter() - mark

    ranked: list[tuple[dict[str, Any], float, dict[str, Any]]] = []
    refs: list[dict[str, str]] = []
    for score, position in zip(scores[0], positions[0], strict=True):
        if position < 0:
            continue
        entry = bundle.mapping[int(position)]
        id_ref = entry["mongo_id"]
        refs.append(id_ref)
        ranked.append((id_ref, float(max(-1.0, min(1.0, score))), entry))

    mark = time.perf_counter()
    products = get_products_by_id_refs(refs)
    stage["mongodb_retrieval_s"] = time.perf_counter() - mark
    mark = time.perf_counter()
    results: list[dict[str, Any]] = []
    requested_gender = _explicit_gender(query)
    for id_ref, score, mapping_entry in ranked:
        product = products.get(_ref_key(id_ref))
        if product is None:
            raise VectorIndexStale("A mapped product is missing from MongoDB; rebuild the FAISS index.")
        if (
            product.get("source_dataset") != mapping_entry.get("source_dataset")
            or product.get("source_product_id") != mapping_entry.get("source_product_id")
        ):
            raise VectorIndexStale("A FAISS mapping no longer matches its MongoDB product; rebuild the index.")
        if _has_apparel_gender_conflict(product, requested_gender, query):
            continue
        results.append({"product": _serialize_document(product), "score": score})
        if len(results) >= limit:
            break

    stage["mapping_filter_serialize_s"] = time.perf_counter() - mark
    stage["service_total_s"] = time.perf_counter() - request_started
    logger.info("Semantic search timings: %s", {key: round(value, 4) for key, value in stage.items()})
    return {"query": query, "results": results, "indexed_products": bundle.index.ntotal}


_MEN_LABEL = re.compile(r"\b(?:men|men's|mens|male|man)\b", re.IGNORECASE)
_WOMEN_LABEL = re.compile(r"\b(?:women|women's|womens|female|woman)\b", re.IGNORECASE)
_UNISEX_LABEL = re.compile(r"\b(?:unisex|all[- ]gender)\b", re.IGNORECASE)
_APPAREL_QUERY = re.compile(
    r"\b(?:dress(?:es)?|shirts?|tops?|clothing|apparel|pants|trousers|jeans|skirts?|jackets?|coats?|blouses?|hoodies|sweaters|suits?)\b",
    re.IGNORECASE,
)


def _explicit_gender(query: str) -> str | None:
    """Return a gender only when the shopper explicitly asks for one."""
    men = bool(_MEN_LABEL.search(query))
    women = bool(_WOMEN_LABEL.search(query))
    if men == women:
        return None
    return "men" if men else "women"


def _has_gender_conflict(product: dict[str, Any], requested_gender: str | None) -> bool:
    """Exclude conflicting audience labels while leaving general queries untouched."""
    if requested_gender is None:
        return False
    title = str(product.get("name") or "")
    if requested_gender == "men":
        return bool(_WOMEN_LABEL.search(title))
    return bool(_MEN_LABEL.search(title))


def _has_apparel_gender_conflict(product: dict[str, Any], requested_gender: str | None, query: str) -> bool:
    """For gender-specific apparel searches, require matching or unisex title data."""
    if _has_gender_conflict(product, requested_gender):
        return True
    if requested_gender is None or not _APPAREL_QUERY.search(query):
        return False
    title = str(product.get("name") or "")
    explicit_requested = _MEN_LABEL.search(title) if requested_gender == "men" else _WOMEN_LABEL.search(title)
    return not explicit_requested and not _UNISEX_LABEL.search(title)


def recommend_products(product_id: str, limit: int) -> dict[str, Any]:
    """Return nearest indexed products for a MongoDB product ID.

    Scores are cosine similarities because this FAISS index stores normalized
    vectors and uses inner-product search.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    bundle = load_current_index()
    _ensure_catalog_validated()

    mapping_by_id = {_ref_key(entry["mongo_id"]): entry for entry in bundle.mapping}
    objectid_ref = {"product_id": product_id, "id_type": "objectid"}
    string_ref = {"product_id": product_id, "id_type": "string"}
    objectid_key = _ref_key(objectid_ref)
    string_key = _ref_key(string_ref)
    requested_key = objectid_key if objectid_key in mapping_by_id else string_key
    requested_ref = mapping_by_id.get(requested_key, {}).get("mongo_id", string_ref)
    selected_mapping = mapping_by_id.get(requested_key)

    lookup_refs = [objectid_ref, string_ref] if len(product_id) == 24 else [string_ref]
    selected_products = get_products_by_id_refs(lookup_refs)
    selected_product = selected_products.get(objectid_key) or selected_products.get(string_key)
    if selected_product is not None:
        requested_key = _ref_key(_mongo_id_ref(selected_product["_id"]))
        requested_ref = _mongo_id_ref(selected_product["_id"])
        selected_mapping = mapping_by_id.get(requested_key)
    if selected_product is None:
        raise ProductNotFound("Product was not found in MongoDB.")
    if selected_mapping is None:
        raise ProductNotIndexed("This product is not present in the current FAISS index.")
    if (
        selected_product.get("source_dataset") != selected_mapping.get("source_dataset")
        or selected_product.get("source_product_id") != selected_mapping.get("source_product_id")
    ):
        raise VectorIndexStale("The FAISS mapping no longer matches this MongoDB product; rebuild the index.")

    selected_position = selected_mapping["vector_position"]
    selected_vector = np.asarray(bundle.index.reconstruct(selected_position), dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(selected_vector)
    candidate_count = bundle.index.ntotal
    scores, positions = bundle.index.search(selected_vector, candidate_count)

    ranked_refs: list[dict[str, str]] = []
    ranked: list[tuple[str, float, dict[str, Any]]] = []
    seen: set[str] = {requested_key}
    ranked_positions = sorted(
        zip(scores[0], positions[0], strict=True),
        key=lambda result: float(result[0]),
        reverse=True,
    )
    for score, position in ranked_positions:
        if position < 0:
            continue
        entry = bundle.mapping[int(position)]
        ref = entry["mongo_id"]
        key = _ref_key(ref)
        if key in seen:
            continue
        seen.add(key)
        ranked_refs.append(ref)
        ranked.append((key, float(max(-1.0, min(1.0, score))), entry))

    products = get_products_by_id_refs(ranked_refs)
    results: list[dict[str, Any]] = []
    for key, score, mapping_entry in ranked:
        product = products.get(key)
        if product is None:
            raise VectorIndexStale("A mapped product is missing from MongoDB; rebuild the FAISS index.")
        if (
            product.get("source_dataset") != mapping_entry.get("source_dataset")
            or product.get("source_product_id") != mapping_entry.get("source_product_id")
        ):
            raise VectorIndexStale("A FAISS mapping no longer matches its MongoDB product; rebuild the index.")
        results.append({"product": _serialize_document(product), "score": score})
        if len(results) >= limit:
            break

    return {
        "product_id": product_id,
        "recommendations": results,
        "indexed_products": bundle.index.ntotal,
    }


class ProductNotFound(LookupError):
    """The requested source product does not exist in MongoDB."""


class ProductNotIndexed(LookupError):
    """The requested MongoDB product has no vector in the active FAISS index."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist the current MongoDB product FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from all current MongoDB products.")
    parser.add_argument("--batch-size", type=int, default=32, help="Model embedding batch size (default: 32).")
    parser.add_argument(
        "--preserve-mongodb-embeddings",
        action="store_true",
        help="Publish the FAISS index without removing cached embeddings from MongoDB.",
    )
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("Pass --rebuild to write a new persistent FAISS snapshot.")
    try:
        result = rebuild_faiss_index(
            batch_size=args.batch_size,
            cleanup_mongo_embeddings=not args.preserve_mongodb_embeddings,
        )
    except Exception as exc:
        # Driver errors may contain connection configuration; keep CLI output safe.
        parser.error(f"Index build failed ({type(exc).__name__}).")
    print(json.dumps({**result, "path": str(FAISS_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
