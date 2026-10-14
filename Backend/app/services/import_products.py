"""Import a bounded subset of Amazon product metadata into MongoDB."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as pq
from pymongo import ASCENDING, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, PyMongoError

from app.config import settings
from app.services.database import check_database_connection, get_client, get_database_name
from app.services.embeddings import EMBEDDING_DIMENSIONS, get_embedding_model, product_content_hash, product_text

DATASET_ID = "iguzelofficial/AMAZON-Products-2023"
DATASET_API = "https://datasets-server.huggingface.co/parquet"
SOURCE_DATASET = DATASET_ID
DATASET_COLUMNS = (
    "parent_asin",
    "title",
    "description",
    "main_category",
    "categories",
    "store",
    "price",
    "image",
    "features",
)
RANGE_BLOCK_SIZE = 1024 * 1024
RANGE_CACHE_BLOCKS = 4


class HTTPRangeFile(io.RawIOBase):
    """Seekable, small-memory HTTP reader that fetches only byte ranges."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.file_size = size
        self.position = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.file_size + offset
        else:
            raise ValueError("Unsupported seek mode.")
        if position < 0:
            raise ValueError("Cannot seek before the start of the file.")
        self.position = min(position, self.file_size)
        return self.position

    def _get_block(self, block_number: int) -> bytes:
        cached = self._blocks.get(block_number)
        if cached is not None:
            self._blocks.move_to_end(block_number)
            return cached

        start = block_number * RANGE_BLOCK_SIZE
        end = min(start + RANGE_BLOCK_SIZE, self.file_size) - 1
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        try:
            response = urlopen(request, timeout=60)
        except (HTTPError, URLError) as exc:
            raise OSError(f"Could not read a dataset byte range ({type(exc).__name__}).") from exc
        with response:
            if response.status != 206:
                raise OSError("Dataset host did not honor a partial-content request.")
            data = response.read(RANGE_BLOCK_SIZE)

        self._blocks[block_number] = data
        self._blocks.move_to_end(block_number)
        while len(self._blocks) > RANGE_CACHE_BLOCKS:
            self._blocks.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed dataset reader.")
        if size is None or size < 0:
            size = self.file_size - self.position
        size = min(size, self.file_size - self.position)
        if size <= 0:
            return b""

        chunks: list[bytes] = []
        remaining = size
        while remaining:
            block_number = self.position // RANGE_BLOCK_SIZE
            offset = self.position % RANGE_BLOCK_SIZE
            block = self._get_block(block_number)
            chunk = block[offset : offset + remaining]
            if not chunk:
                break
            chunks.append(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def readinto(self, target: bytearray) -> int:
        data = self.read(len(target))
        target[: len(data)] = data
        return len(data)


@dataclass
class ImportStats:
    scanned: int = 0
    imported: int = 0
    updated: int = 0
    skipped: int = 0
    duplicate_rows: int = 0
    rejected: int = 0
    failed: int = 0
    invalid_price: int = 0
    would_import: int = 0
    would_update: int = 0


def _clean_text(value: Any) -> str | None:
    """Return a compact display string for a dataset text or list field."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [_clean_text(item) for item in value]
        text = " ".join(part for part in parts if part)
    elif isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
    else:
        text = str(value).strip()
    return text or None


def _parse_price(value: Any) -> tuple[float | None, str]:
    """Return a nonnegative finite price and whether it was valid or missing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    try:
        amount = Decimal(str(value).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, ValueError):
        return None, "invalid"
    if not amount.is_finite() or amount < 0 or not math.isfinite(float(amount)):
        return None, "invalid"
    return float(amount), "valid"


def map_dataset_product(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str, str | None]:
    """Map known source fields, returning product, price status, and rejection reason."""
    source_id = _clean_text(row.get("parent_asin"))
    name = _clean_text(row.get("title"))
    if not source_id:
        return None, "missing", "missing_parent_asin"
    if not name:
        return None, "missing", "missing_title"

    category = _clean_text(row.get("main_category"))
    if not category:
        category = _clean_text(row.get("categories"))

    product: dict[str, Any] = {
        "name": name,
        "source_dataset": SOURCE_DATASET,
        "source_product_id": source_id,
    }
    description = _clean_text(row.get("description"))
    brand = _clean_text(row.get("store"))
    image_url = _clean_text(row.get("image"))
    if description:
        product["description"] = description
    if category:
        product["category"] = category
    if brand:
        product["brand"] = brand
    if image_url:
        product["image_url"] = image_url
    features = row.get("features")
    if isinstance(features, (list, tuple)):
        cleaned_features = [text for item in features if (text := _clean_text(item))]
    elif (feature_text := _clean_text(features)):
        cleaned_features = [feature_text]
    else:
        cleaned_features = []
    if cleaned_features:
        product["features"] = cleaned_features

    price, price_status = _parse_price(row.get("price"))
    if price is not None:
        product["price"] = price
    return product, price_status, None


def ensure_source_unique_index(collection: Collection) -> None:
    """Check source-scoped records before creating their partial unique index."""
    scoped = collection.find(
        {"source_dataset": SOURCE_DATASET},
        {"source_product_id": 1},
    )
    source_ids: list[str] = []
    for document in scoped:
        source_id = document.get("source_product_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise RuntimeError(
                "Cannot create the import index: an existing dataset-scoped product has no source_product_id."
            )
        source_ids.append(source_id)
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError(
            "Cannot create the import index: duplicate source_product_id values already exist for this dataset."
        )

    collection.create_index(
        [("source_dataset", ASCENDING), ("source_product_id", ASCENDING)],
        unique=True,
        partialFilterExpression={"source_dataset": SOURCE_DATASET},
        name="uniq_amazon_products_2023_source_id",
    )


def process_import_batch(
    rows: list[dict[str, Any]],
    stats: ImportStats,
    seen_ids: set[str],
    collection: Collection | None,
    batch_size: int,
    dry_run: bool = False,
    insert_only: bool = False,
) -> None:
    """Map and embed one bounded batch, upserting products without replacing documents."""
    mapped: list[dict[str, Any]] = []
    for row in rows:
        stats.scanned += 1
        product, price_status, _rejection = map_dataset_product(row)
        if price_status == "invalid":
            stats.invalid_price += 1
        if product is None:
            stats.rejected += 1
            continue

        source_id = product["source_product_id"]
        if source_id in seen_ids:
            stats.skipped += 1
            stats.duplicate_rows += 1
            continue
        seen_ids.add(source_id)
        product["_source_content_hash"] = product_content_hash(product)
        mapped.append(product)

    existing: dict[str, dict[str, Any]] = {}
    if collection is not None and mapped:
        source_ids = [product["source_product_id"] for product in mapped]
        for document in collection.find(
            {"source_dataset": SOURCE_DATASET, "source_product_id": {"$in": source_ids}},
            {"_id": 1, "source_product_id": 1, "_source_content_hash": 1, "_embedding_model": 1, "_embedding": 1},
        ):
            existing[document["source_product_id"]] = document

    pending: list[tuple[dict[str, Any], bool]] = []
    for product in mapped:
        source_id = product["source_product_id"]
        old = existing.get(source_id)
        if old:
            if insert_only:
                stats.skipped += 1
                continue
            old_vector = old.get("_embedding")
            embedding_valid = (
                isinstance(old_vector, list)
                and len(old_vector) == EMBEDDING_DIMENSIONS
                and old.get("_embedding_model") == settings.embedding_model
            )
            indexed_in_faiss = False
            if old.get("_source_content_hash") == product["_source_content_hash"] and not embedding_valid:
                from app.services.vector_search import VectorIndexStale, VectorIndexUnavailable, has_product_vector

                try:
                    indexed_in_faiss = has_product_vector(str(old["_id"]))
                except (VectorIndexStale, VectorIndexUnavailable):
                    indexed_in_faiss = False
            if old.get("_source_content_hash") == product["_source_content_hash"] and (
                embedding_valid or indexed_in_faiss
            ):
                stats.skipped += 1
                continue
            pending.append((product, True))
        else:
            pending.append((product, False))

    if not pending:
        return

    model = get_embedding_model()
    vectors = np.asarray(
        model.encode(
            [product_text(product) for product, _ in pending],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.shape != (len(pending), EMBEDDING_DIMENSIONS):
        raise ValueError(f"Embedding model returned unexpected shape {vectors.shape}.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned a non-finite vector.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector.")
    vectors /= norms[:, None]

    operations: list[UpdateOne] = []
    operation_is_update: list[bool] = []
    for (product, was_existing), vector in zip(pending, vectors, strict=True):
        values = {
            **product,
            "_embedding": vector.tolist(),
            "_embedding_model": settings.embedding_model,
        }
        fields_to_unset = {
            field: ""
            for field in ("description", "category", "brand", "image_url", "price", "features")
            if field not in product
        }
        update = (
            {"$setOnInsert": values}
            if insert_only
            else {"$set": values, **({"$unset": fields_to_unset} if fields_to_unset else {})}
        )
        operations.append(
            UpdateOne(
                {"source_dataset": SOURCE_DATASET, "source_product_id": product["source_product_id"]},
                update,
                upsert=True,
            )
        )
        operation_is_update.append(was_existing)
        if dry_run:
            if was_existing:
                stats.would_update += 1
            else:
                stats.would_import += 1
        elif was_existing:
            stats.updated += 1
        elif not insert_only:
            stats.imported += 1

    if not dry_run and collection is not None:
        try:
            result = collection.bulk_write(operations, ordered=False)
            if insert_only:
                stats.imported += result.upserted_count
                stats.skipped += len(operations) - result.upserted_count
        except BulkWriteError as exc:
            details = exc.details or {}
            failed_indexes = {item.get("index") for item in details.get("writeErrors", [])}
            if details.get("writeConcernErrors"):
                failed_indexes.update(range(len(operation_is_update)))
            stats.failed += len(failed_indexes)
            if insert_only:
                stats.imported += int(details.get("nUpserted", 0))
            else:
                for index in failed_indexes:
                    if index is not None and 0 <= index < len(operation_is_update):
                        if operation_is_update[index]:
                            stats.updated -= 1
                        else:
                            stats.imported -= 1
        except PyMongoError:
            stats.failed += len(operation_is_update)
            if not insert_only:
                stats.imported -= sum(not was_update for was_update in operation_is_update)
            stats.updated -= sum(operation_is_update)
            raise


def get_parquet_shards() -> list[dict[str, Any]]:
    """Ask the Hugging Face dataset service for the current train Parquet shards."""
    url = DATASET_API + "?" + urlencode({"dataset": DATASET_ID})
    try:
        with urlopen(url, timeout=45) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not retrieve dataset shard metadata ({type(exc).__name__}).") from exc

    shards = [
        shard
        for shard in payload.get("parquet_files", [])
        if shard.get("split") == "train" and shard.get("url") and shard.get("size")
    ]
    if not shards:
        raise RuntimeError("Hugging Face returned no train Parquet shards for the dataset.")
    return sorted(shards, key=lambda shard: shard.get("filename", ""))


def iter_dataset_batches(limit: int, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Read only selected columns from remote Parquet byte ranges, stopping at limit."""
    remaining = limit
    for shard in get_parquet_shards():
        if remaining <= 0:
            break
        with HTTPRangeFile(shard["url"], int(shard["size"])) as remote_file:
            parquet = pq.ParquetFile(remote_file)
            available = set(parquet.schema_arrow.names)
            columns = [name for name in DATASET_COLUMNS if name in available]
            if "parent_asin" not in columns or "title" not in columns:
                raise RuntimeError("The dataset shard is missing parent_asin or title.")

            for record_batch in parquet.iter_batches(batch_size=min(batch_size, remaining), columns=columns):
                rows = record_batch.to_pylist()[:remaining]
                if not rows:
                    continue
                yield rows
                remaining -= len(rows)
                if remaining <= 0:
                    break


def run_import(
    limit: int = 100,
    batch_size: int = 32,
    dry_run: bool = False,
    insert_only: bool = False,
) -> ImportStats:
    """Import up to limit source rows, generating fresh MiniLM embeddings."""
    if limit < 1:
        raise ValueError("--limit must be at least 1.")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    collection: Collection | None = None
    if not dry_run:
        check_database_connection()
        collection = get_client()[get_database_name()]["products"]
        ensure_source_unique_index(collection)

    stats = ImportStats()
    seen_ids: set[str] = set()
    try:
        for rows in iter_dataset_batches(limit=limit, batch_size=batch_size):
            process_import_batch(
                rows,
                stats,
                seen_ids,
                collection,
                batch_size,
                dry_run=dry_run,
                insert_only=insert_only,
            )
    except (PyMongoError, OSError, RuntimeError, ValueError):
        print(json.dumps({**asdict(stats), "status": "partial_failure"}), file=sys.stderr)
        raise
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="Maximum source rows to scan (default: 100).")
    parser.add_argument("--batch-size", type=int, default=32, help="Rows and embeddings per batch (default: 32).")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and embed products but do not connect to or write MongoDB.")
    parser.add_argument(
        "--insert-only",
        action="store_true",
        help="Skip all existing source IDs and never update existing product documents.",
    )
    args = parser.parse_args()

    try:
        stats = run_import(args.limit, args.batch_size, args.dry_run, args.insert_only)
    except (PyMongoError, OSError, RuntimeError, ValueError) as exc:
        # Error strings from database drivers can contain connection details.
        parser.error(f"Import failed: {type(exc).__name__}.")
    print(json.dumps({**asdict(stats), "dataset": DATASET_ID, "model": settings.embedding_model, "dry_run": args.dry_run, "insert_only": args.insert_only}, indent=2))


if __name__ == "__main__":
    main()
