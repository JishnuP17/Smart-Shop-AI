"""Transparent, user-based recommendations for the three active demo profiles."""

from __future__ import annotations

import threading
import re
from datetime import datetime, timezone
from typing import Any

import faiss
import numpy as np

from app.services.database import get_demo_interactions, get_demo_user
from app.services.embeddings import EMBEDDING_DIMENSIONS, embed_texts
from app.services.vector_search import (
    VectorIndexStale,
    _ensure_catalog_validated,
    _ref_key,
    _serialize_document,
    load_current_index,
)

DEMO_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "_id": "demo_alex",
        "name": "Alex",
        "description": "Audio for listening, creating, and recording.",
        "interests": ["wireless headphones", "Bluetooth speakers", "studio microphones", "audio interfaces and recording gear"],
    },
    {
        "_id": "demo_sam",
        "name": "Sam",
        "description": "A mix of apparel, footwear, and everyday fashion accessories.",
        "interests": ["men's shirts and casualwear", "women's dresses and tops", "running shoes and sneakers", "sportswear and activewear", "fashion bags and accessories"],
    },
    {
        "_id": "demo_taylor",
        "name": "Taylor",
        "description": "Practical kitchen, storage, and small-appliance finds.",
        "interests": ["kitchen appliances", "kitchen storage and organizers", "refrigerator organizers", "small home appliances", "home organization"],
    },
)
DEMO_USER_IDS = tuple(profile["_id"] for profile in DEMO_PROFILES)

_interest_cache: dict[tuple[str, ...], np.ndarray] = {}
_interest_vectors_cache: dict[tuple[str, ...], np.ndarray] = {}
_interest_cache_lock = threading.Lock()
_EVENT_WEIGHTS = {"click": 3.0, "view": 0.5, "search": 1.5}
_HISTORY_BLEND = 0.25
_RECENCY_HALF_LIFE_DAYS = 30.0
_SEMANTIC_WEIGHT = 0.88
_METADATA_WEIGHT = 0.12
_TOPIC_DIVERSITY_PENALTY = 0.025
_WORD_RE = re.compile(r"[a-z0-9]+")
_INTEREST_STOPWORDS = {"and", "or", "for", "with", "the", "a", "an", "and", "gear", "products", "product"}


def _normalized(vector: Any) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32).reshape(-1)
    if result.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(result).all():
        raise ValueError("The personalized recommendation vector is invalid.")
    faiss.normalize_L2(result.reshape(1, -1))
    if not np.any(result):
        raise ValueError("The personalized recommendation vector is empty.")
    return result


def _initial_preference(interests: list[str]) -> np.ndarray:
    cache_key = tuple(interests)
    with _interest_cache_lock:
        cached = _interest_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

    vectors = _interest_vectors(interests)
    vector = _normalized(np.mean(vectors, axis=0))
    with _interest_cache_lock:
        _interest_cache[cache_key] = vector.copy()
    return vector


def _interest_vectors(interests: list[str]) -> np.ndarray:
    cache_key = tuple(interests)
    with _interest_cache_lock:
        cached = _interest_vectors_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
    vectors = np.asarray(embed_texts(interests), dtype=np.float32)
    for vector in vectors:
        faiss.normalize_L2(vector.reshape(1, -1))
    with _interest_cache_lock:
        _interest_vectors_cache[cache_key] = vectors.copy()
    return vectors


def _interaction_preference(
    interactions: list[dict[str, Any]], bundle: Any
) -> tuple[np.ndarray | None, set[str]]:
    mapping_by_id = {_ref_key(entry["mongo_id"]): entry for entry in bundle.mapping}
    contributions: dict[str, tuple[np.ndarray | None, float]] = {}
    seen_product_ids: set[str] = set()
    search_events: dict[str, tuple[str, float]] = {}

    def recency(event: dict[str, Any]) -> float:
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, datetime):
            return 1.0
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds() / 86400)
        return 0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS)

    for event in interactions:
        event_type = event.get("event_type")
        weight = _EVENT_WEIGHTS.get(event_type)
        if weight is None:
            continue
        if event_type == "search":
            query = str(event.get("query") or "").strip()
            if query:
                key = "query:" + " ".join(query.lower().split())
                candidate = (query, weight * recency(event))
                if key not in search_events or candidate[1] > search_events[key][1]:
                    search_events[key] = candidate
            continue
        product_id = event.get("product_id")
        if not product_id:
            continue
        seen_product_ids.add(product_id)
        ref_key = f"objectid:{product_id}" if len(product_id) == 24 else f"string:{product_id}"
        mapping = mapping_by_id.get(ref_key)
        if mapping is None:
            continue
        position = mapping.get("vector_position")
        key = "product:" + product_id
        candidate = (_normalized(bundle.index.reconstruct(position)), weight * recency(event))
        # Keep the strongest real signal for a product; repeated clicks with fresh
        # random event IDs cannot overwhelm a profile's stated interests.
        if key not in contributions or candidate[1] > contributions[key][1]:
            contributions[key] = candidate

    if search_events:
        query_vectors = embed_texts([event[0] for event in search_events.values()])
        for (query, weight), vector in zip(search_events.values(), query_vectors, strict=True):
            contributions["query:" + " ".join(query.lower().split())] = (_normalized(vector), weight)

    if not contributions:
        return None, seen_product_ids
    history = np.average(
        np.stack([vector for vector, _weight in contributions.values() if vector is not None]),
        axis=0,
        weights=[weight for vector, weight in contributions.values() if vector is not None],
    )
    return _normalized(history), seen_product_ids


def _tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in _WORD_RE.findall(str(value or "").lower()):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("sses"):
            token = token[:-2]
        elif token.endswith(("ches", "shes", "xes", "zes")):
            token = token[:-2]
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            token = token[:-1]
        if token == "mens":
            token = "men"
        elif token == "womens":
            token = "women"
        if token not in _INTEREST_STOPWORDS and len(token) > 1:
            tokens.add(token)
    return tokens


def _metadata_relevance(product: dict[str, Any], interests: list[str]) -> float:
    """Return a small, explainable lexical relevance signal from real product fields."""
    title = " ".join(str(product.get(key) or "") for key in ("name", "title")).lower()
    category = str(product.get("category") or "").lower()
    description = " ".join(str(product.get(key) or "") for key in ("description", "features")).lower()
    best = 0.0
    for interest in interests:
        terms = _tokens(interest)
        if not terms:
            continue
        matched = 0.0
        for term in terms:
            if term in _tokens(title):
                matched += 1.0
            elif term in _tokens(category):
                matched += 0.8
            elif term in _tokens(description):
                matched += 0.35
        coverage = matched / len(terms)
        phrase_tokens = " ".join(_WORD_RE.findall(interest.lower()))
        if phrase_tokens and phrase_tokens in " ".join(_WORD_RE.findall(title)):
            coverage = max(coverage, 1.0)
        best = max(best, coverage)
    return min(1.0, best)


def _title_key(product: dict[str, Any]) -> str:
    return " ".join(_WORD_RE.findall(str(product.get("name") or product.get("title") or "").lower()))


def get_personalized_recommendations(user_id: str, limit: int = 10) -> dict[str, Any]:
    """Rank products using initial interests plus recent real user interactions."""
    if user_id not in DEMO_USER_IDS:
        raise LookupError("Demo profile was not found.")
    user = get_demo_user(user_id)
    if user is None:
        raise LookupError("Demo profile was not found.")
    profile = next(profile for profile in DEMO_PROFILES if profile["_id"] == user_id)
    if limit < 1:
        raise ValueError("Recommendation limit must be at least one.")

    bundle = load_current_index()
    _ensure_catalog_validated()
    if bundle.index.ntotal == 0:
        return {
            "user_id": user_id,
            "recommendations": [],
            "indexed_products": 0,
            "based_on": "initial_interests",
        }

    base = _initial_preference(profile["interests"])
    interactions = get_demo_interactions(user_id, limit=200)
    history, seen_product_ids = _interaction_preference(interactions, bundle)
    if history is None:
        preference = base
        based_on = "initial_interests"
    else:
        # Recent actions adjust, but do not erase, each profile's stated interests.
        preference = _normalized((1.0 - _HISTORY_BLEND) * base + _HISTORY_BLEND * history)
        based_on = "interests_and_interactions"

    interest_vectors = _interest_vectors(profile["interests"])
    candidate_count = min(bundle.index.ntotal, max(80, limit * 10))
    from app.services.database import get_products_by_id_refs

    while True:
        query_vectors = [preference, *interest_vectors]
        candidates: dict[int, int] = {}
        for interest_index, query_vector in enumerate(query_vectors):
            _scores, positions = bundle.index.search(query_vector.reshape(1, -1), candidate_count)
            for raw_position in positions[0]:
                position = int(raw_position)
                if position >= 0:
                    candidates.setdefault(position, interest_index)

        ranked: list[tuple[dict[str, str], float, dict[str, Any], int]] = []
        refs: list[dict[str, str]] = []
        for position in candidates:
            mapping = bundle.mapping[position]
            ref = mapping["mongo_id"]
            refs.append(ref)
            product_vector = _normalized(bundle.index.reconstruct(position))
            topic_scores = interest_vectors @ product_vector
            topic_index = int(np.argmax(topic_scores))
            profile_score = float(topic_scores[topic_index])
            if history is not None:
                score = (1.0 - _HISTORY_BLEND) * profile_score + _HISTORY_BLEND * float(np.dot(history, product_vector))
            else:
                score = profile_score
            ranked.append((ref, float(max(-1.0, min(1.0, score))), mapping, topic_index))

        products = get_products_by_id_refs(refs)
        fresh: list[tuple[dict[str, Any], str, int]] = []
        familiar: list[tuple[dict[str, Any], str, int]] = []
        seen_keys: set[str] = set()
        for ref, score, mapping, topic_index in ranked:
            key = _ref_key(ref)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            product = products.get(key)
            if product is None:
                raise VectorIndexStale("A mapped product is missing from MongoDB; rebuild the FAISS index.")
            if (
                product.get("source_dataset") != mapping.get("source_dataset")
                or product.get("source_product_id") != mapping.get("source_product_id")
            ):
                raise VectorIndexStale("A FAISS mapping no longer matches its MongoDB product; rebuild the index.")
            serialized = _serialize_document(product)
            metadata_score = _metadata_relevance(product, profile["interests"])
            final_score = _SEMANTIC_WEIGHT * score + _METADATA_WEIGHT * metadata_score
            hit = {"product": serialized, "score": final_score}
            product_id = str(product["_id"])
            title_key = _title_key(product)
            if not title_key:
                title_key = key
            (familiar if product_id in seen_product_ids else fresh).append((hit, title_key, topic_index))
        fresh.sort(key=lambda item: item[0]["score"], reverse=True)
        familiar.sort(key=lambda item: item[0]["score"], reverse=True)
        unique_fresh_titles = {title_key for _hit, title_key, _topic in fresh}
        if len(unique_fresh_titles) >= limit or candidate_count == bundle.index.ntotal:
            break
        candidate_count = min(bundle.index.ntotal, candidate_count * 2)

    chosen_hits: list[dict[str, Any]] = []
    chosen_titles: set[str] = set()
    # Exact/near-exact duplicate catalog titles add little value to a recommendation list.
    remaining = fresh + familiar
    topic_counts: dict[int, int] = {}
    while remaining and len(chosen_hits) < limit:
        best_index = -1
        best_adjusted_score = float("-inf")
        for index, (hit, title_key, topic_index) in enumerate(remaining):
            if title_key in chosen_titles:
                continue
            adjusted = hit["score"] - _TOPIC_DIVERSITY_PENALTY * topic_counts.get(topic_index, 0)
            if adjusted > best_adjusted_score:
                best_index = index
                best_adjusted_score = adjusted
        if best_index < 0:
            break
        hit, title_key, topic_index = remaining.pop(best_index)
        chosen_titles.add(title_key)
        topic_counts[topic_index] = topic_counts.get(topic_index, 0) + 1
        chosen_hits.append(hit)
    return {
        "user_id": user_id,
        "recommendations": chosen_hits,
        "indexed_products": bundle.index.ntotal,
        "based_on": based_on,
    }
