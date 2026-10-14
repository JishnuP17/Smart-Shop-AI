"""MongoDB client and product catalog operations."""

from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import MongoClient, UpdateOne
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.config import settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Create the shared MongoDB client on first use and reuse it thereafter."""
    global _client

    if not settings.mongodb_uri.strip():
        raise ValueError("MONGODB_URI is not configured.")
    if not settings.mongodb_database.strip():
        raise ValueError("MONGODB_DATABASE is not configured.")

    if _client is None:
        _client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )
    return _client


def get_database_name() -> str:
    """Return the configured database name."""
    if not settings.mongodb_database.strip():
        raise ValueError("MONGODB_DATABASE is not configured.")
    return settings.mongodb_database


def initialize_demo_user_storage(profiles: Sequence[dict[str, Any]]) -> None:
    """Create indexes and insert missing demo profiles without resetting existing data."""
    database = get_client()[get_database_name()]
    users = database["demo_users"]
    interactions = database["user_interactions"]
    interactions.create_index([("user_id", 1), ("timestamp", -1)], name="user_recent_interactions")
    interactions.create_index([("user_id", 1), ("event_id", 1)], unique=True, name="user_event_id_unique")
    for profile in profiles:
        users.update_one(
            {"_id": profile["_id"]},
            {"$setOnInsert": {**profile, "created_at": datetime.now(timezone.utc)}},
            upsert=True,
        )


def get_demo_users(profiles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return configured profile descriptions only for profiles seeded in MongoDB."""
    if not profiles:
        return []
    user_ids = [profile["_id"] for profile in profiles]
    found = {
        document["_id"]: document
        for document in get_client()[get_database_name()]["demo_users"].find(
            {"_id": {"$in": user_ids}}, {"_id": 1}
        )
    }
    return [
        {
            "id": profile["_id"],
            "name": profile["name"],
            "description": profile["description"],
            "interests": profile["interests"],
        }
        for profile in profiles
        if profile["_id"] in found
    ]


def get_demo_user(user_id: str) -> dict[str, Any] | None:
    """Fetch one stable demo profile by its ID."""
    document = get_client()[get_database_name()]["demo_users"].find_one(
        {"_id": user_id}, {"name": 1, "description": 1, "interests": 1}
    )
    if document is None:
        return None
    return {
        "id": user_id,
        "name": document["name"],
        "description": document.get("description", "Explore products selected for this demo profile."),
        "interests": document["interests"],
    }


def save_demo_interaction(interaction: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Persist an interaction once per demo user and event ID."""
    collection = get_client()[get_database_name()]["user_interactions"]
    key = {"user_id": interaction["user_id"], "event_id": interaction["event_id"]}
    try:
        result = collection.update_one(key, {"$setOnInsert": interaction}, upsert=True)
    except DuplicateKeyError:
        result = None
    saved = collection.find_one(key)
    if saved is None:
        if result is not None:
            raise PyMongoError("The interaction could not be read after it was saved.")
        raise PyMongoError("The interaction could not be saved.")
    return saved, bool(result and result.upserted_id is not None)


def get_demo_interactions(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Read only one demo user's recent interaction history."""
    collection = get_client()[get_database_name()]["user_interactions"]
    cursor = collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    return list(cursor)


def check_database_connection() -> None:
    """Ping MongoDB, raising an error if it cannot be reached."""
    client = get_client()
    client.admin.command("ping")


def _serialize_document(document: dict[str, Any]) -> dict[str, Any]:
    """Convert ObjectIds to strings and hide internal embedding metadata."""
    serialized: dict[str, Any] = {}
    for key, value in document.items():
        if key in {"_embedding", "_embedding_model", "_source_content_hash"}:
            continue
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, dict):
            serialized[key] = _serialize_document(value)
        elif isinstance(value, list):
            serialized[key] = [
                _serialize_document(item) if isinstance(item, dict)
                else str(item) if isinstance(item, ObjectId)
                else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def get_products(skip: int, limit: int, category: str | None = None) -> list[dict[str, Any]]:
    """Read a page from the products collection without changing records."""
    client = get_client()
    cursor = (
        client[get_database_name()]["products"]
        .find({"category": category} if category else {})
        .skip(skip)
        .limit(limit)
    )
    return [_serialize_document(product) for product in cursor]


def get_product_by_id(product_id: str) -> dict[str, Any] | None:
    """Fetch one real catalog product by its MongoDB ObjectId or string ID."""
    collection = get_client()[get_database_name()]["products"]
    candidates: list[Any] = [product_id]
    if ObjectId.is_valid(product_id):
        candidates.insert(0, ObjectId(product_id))
    document = collection.find_one({"_id": {"$in": candidates}})
    return _serialize_document(document) if document else None


def get_product_categories() -> list[dict[str, Any]]:
    """Return category names and product counts derived from MongoDB records."""
    collection = get_client()[get_database_name()]["products"]
    pipeline = [
        {"$match": {"category": {"$type": "string", "$ne": ""}}},
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1, "_id": 1}},
    ]
    return [
        {"name": row["_id"], "count": row["count"]}
        for row in collection.aggregate(pipeline)
    ]


def create_product(product_data: dict[str, Any]) -> dict[str, Any]:
    """Insert a product and return the saved document with a string ID."""
    client = get_client()
    collection = client[get_database_name()]["products"]
    result = collection.insert_one(product_data)
    created_product = collection.find_one({"_id": result.inserted_id})
    if created_product is None:
        raise PyMongoError("The inserted product could not be read back.")
    return _serialize_document(created_product)


def get_products_for_embedding() -> list[dict[str, Any]]:
    """Read catalog documents for the one-time embedding backfill."""
    collection = get_client()[get_database_name()]["products"]
    return list(collection.find({}, {"name": 1, "description": 1, "category": 1, "brand": 1}))


def save_product_embedding(product_id: ObjectId, vector: list[float], model_name: str) -> None:
    """Store or replace only the internal embedding fields on one product."""
    collection = get_client()[get_database_name()]["products"]
    collection.update_one(
        {"_id": product_id},
        {"$set": {"_embedding": vector, "_embedding_model": model_name}},
    )


def get_products_with_embeddings(model_name: str) -> list[dict[str, Any]]:
    """Return catalog records whose vectors were made by the active model."""
    collection = get_client()[get_database_name()]["products"]
    return list(collection.find({"_embedding_model": model_name, "_embedding": {"$exists": True}}))


def iter_catalog_products(batch_size: int = 256, include_vectors: bool = True) -> Iterator[dict[str, Any]]:
    """Stream product fields needed to build and validate the vector index."""
    collection = get_client()[get_database_name()]["products"]
    projection = {
        "name": 1,
        "description": 1,
        "category": 1,
        "brand": 1,
        "features": 1,
        "source_dataset": 1,
        "source_product_id": 1,
        "_embedding": 1,
        "_embedding_model": 1,
        "_source_content_hash": 1,
    }
    if not include_vectors:
        projection.pop("_embedding")
        projection.pop("_embedding_model")
    cursor = collection.find({}, projection).sort("_id", 1).batch_size(batch_size)
    yield from cursor


def get_products_by_id_refs(id_refs: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Fetch products by the explicit ID types persisted in the FAISS mapping."""
    if not id_refs:
        return {}

    ids: list[Any] = []
    for ref in id_refs:
        if ref.get("id_type") == "objectid":
            ids.append(ObjectId(ref["product_id"]))
        elif ref.get("id_type") == "string":
            ids.append(ref["product_id"])
        else:
            raise ValueError("FAISS mapping contains an unsupported MongoDB ID type.")

    collection = get_client()[get_database_name()]["products"]
    documents = collection.find({"_id": {"$in": ids}})
    result: dict[str, dict[str, Any]] = {}
    for document in documents:
        id_type = "objectid" if isinstance(document["_id"], ObjectId) else "string"
        result[f"{id_type}:{document['_id']}"] = document
    return result


def finalize_vector_migration(indexed_records: list[dict[str, Any]]) -> None:
    """After the FAISS snapshot is durable, keep hashes and remove Mongo vectors."""
    if not indexed_records:
        return
    operations: list[UpdateOne] = []
    semantic_fields = ("name", "description", "category", "brand", "features")
    for record in indexed_records:
        filter_document: dict[str, Any] = {"_id": record["_id"]}
        for field in semantic_fields:
            if field in record:
                filter_document[field] = record[field]
            else:
                filter_document[field] = {"$exists": False}
        operations.append(
            UpdateOne(
                filter_document,
                {
                    "$set": {"_source_content_hash": record["_source_content_hash"]},
                    "$unset": {"_embedding": "", "_embedding_model": ""},
                },
            )
        )
    if operations:
        get_client()[get_database_name()]["products"].bulk_write(operations, ordered=False)
