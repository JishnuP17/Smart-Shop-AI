"""Tests for Amazon dataset field mapping, deduplication, and batch embedding."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

from app.services.import_products import (
    ImportStats,
    iter_dataset_batches,
    map_dataset_product,
    process_import_batch,
    run_import,
)
from app.services.embeddings import EMBEDDING_DIMENSIONS, product_content_hash, product_text
from app.config import settings


class FakeCollection:
    def __init__(self, existing=None):
        self.existing = existing or []
        self.operations = []

    def find(self, query, projection):
        requested = set(query["source_product_id"]["$in"])
        return [doc for doc in self.existing if doc["source_product_id"] in requested]

    def bulk_write(self, operations, ordered=False):
        self.operations.extend(operations)
        return type("BulkResult", (), {"upserted_count": len(operations)})()


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


class FakeRecordBatch:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return self.rows


class FakeParquetFile:
    schema_arrow = type("FakeSchema", (), {"names": ["parent_asin", "title"]})()

    def __init__(self, _remote_file):
        pass

    def iter_batches(self, batch_size, columns):
        self.columns = columns
        rows = [{"parent_asin": f"id-{number}", "title": f"Product {number}"} for number in range(64)]
        yield FakeRecordBatch(rows[:32])
        yield FakeRecordBatch(rows[32:])


class ImportProductTests(unittest.TestCase):
    def test_maps_source_fields_without_retaining_raw_embedding(self):
        product, price_status, rejection = map_dataset_product(
            {
                "parent_asin": " B000000001 ",
                "title": "  Garden   Hose ",
                "description": ["Flexible", "25 feet"],
                "main_category": "Garden",
                "store": "Example Brand",
                "price": 19.95,
                "image": "https://example.com/hose.jpg",
                "features": ["25-foot reach", "Kink-resistant"],
                "embeddings": [0.1, 0.2],
            }
        )

        self.assertIsNone(rejection)
        self.assertEqual(price_status, "valid")
        self.assertEqual(product["name"], "Garden Hose")
        self.assertEqual(product["description"], "Flexible 25 feet")
        self.assertEqual(product["brand"], "Example Brand")
        self.assertEqual(product["category"], "Garden")
        self.assertEqual(product["price"], 19.95)
        self.assertEqual(product["image_url"], "https://example.com/hose.jpg")
        self.assertEqual(product["features"], ["25-foot reach", "Kink-resistant"])
        self.assertNotIn("embeddings", product)

    def test_missing_optional_data_and_price_are_not_fabricated(self):
        product, price_status, rejection = map_dataset_product(
            {"parent_asin": "B000000002", "title": "Unnamed brand product", "price": None}
        )
        self.assertIsNone(rejection)
        self.assertEqual(price_status, "missing")
        self.assertNotIn("price", product)
        self.assertNotIn("brand", product)
        self.assertEqual(product_text(product), "name: Unnamed brand product")

    def test_invalid_prices_are_omitted_and_malformed_records_rejected(self):
        product, price_status, rejection = map_dataset_product(
            {"parent_asin": "B000000003", "title": "Bad price", "price": -1}
        )
        self.assertIsNone(rejection)
        self.assertEqual(price_status, "invalid")
        self.assertNotIn("price", product)
        malformed, _, reason = map_dataset_product({"title": "No stable id"})
        self.assertIsNone(malformed)
        self.assertEqual(reason, "missing_parent_asin")

    def test_batch_deduplicates_ids_and_generates_vectors_in_batch(self):
        collection = FakeCollection()
        model = FakeModel()
        stats = ImportStats()
        seen_ids = set()
        rows = [
            {"parent_asin": "B000000004", "title": "Blue running shoe", "price": 29.99},
            {"parent_asin": "B000000004", "title": "Duplicate row", "price": 29.99},
            {"parent_asin": "B000000005", "title": "Light hiking backpack", "price": 45.00},
        ]

        with patch("app.services.import_products.get_embedding_model", return_value=model):
            process_import_batch(rows, stats, seen_ids, collection, batch_size=2)

        self.assertEqual(stats.scanned, 3)
        self.assertEqual(stats.imported, 2)
        self.assertEqual(stats.skipped, 1)
        self.assertEqual(stats.duplicate_rows, 1)
        self.assertEqual(len(collection.operations), 2)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][1]["batch_size"], 2)

    def test_streaming_reader_never_exceeds_requested_limit(self):
        with (
            patch("app.services.import_products.get_parquet_shards", return_value=[{"url": "unused", "size": 1}]),
            patch("app.services.import_products.HTTPRangeFile", return_value=nullcontext(object())),
            patch("app.services.import_products.pq.ParquetFile", FakeParquetFile),
        ):
            batches = list(iter_dataset_batches(limit=35, batch_size=32))

        self.assertEqual([len(batch) for batch in batches], [32, 3])
        self.assertEqual(sum(map(len, batches)), 35)

    def test_dry_run_generates_vectors_without_connecting_or_writing(self):
        model = FakeModel()
        rows = [{"parent_asin": "B000000006", "title": "Travel charger", "price": 12.0}]
        with (
            patch("app.services.import_products.iter_dataset_batches", return_value=iter([rows])),
            patch("app.services.import_products.get_embedding_model", return_value=model),
            patch("app.services.import_products.get_client", side_effect=AssertionError("dry run connected to MongoDB")),
        ):
            stats = run_import(limit=1, batch_size=1, dry_run=True)

        self.assertEqual(stats.would_import, 1)
        self.assertEqual(stats.imported, 0)
        self.assertEqual(len(model.calls), 1)

    def test_rerun_skips_product_already_mapped_in_faiss(self):
        product, _, _ = map_dataset_product(
            {"parent_asin": "B000000007", "title": "Travel mug", "price": 8.0}
        )
        product_hash = product_content_hash(product)
        object_id = "000000000000000000000007"
        collection = FakeCollection(
            [
                {
                    "_id": object_id,
                    "source_product_id": product["source_product_id"],
                    "_source_content_hash": product_hash,
                    "_embedding_model": settings.embedding_model,
                }
            ]
        )
        stats = ImportStats()

        with (
            patch("app.services.vector_search.has_product_vector", return_value=True),
            patch("app.services.import_products.get_embedding_model", side_effect=AssertionError("embedding was regenerated")),
        ):
            process_import_batch(
                [{"parent_asin": "B000000007", "title": "Travel mug", "price": 8.0}],
                stats,
                set(),
                collection,
                batch_size=1,
            )

        self.assertEqual(stats.skipped, 1)
        self.assertEqual(stats.imported, 0)
        self.assertEqual(collection.operations, [])

    def test_insert_only_never_updates_existing_source_records(self):
        collection = FakeCollection([{"source_product_id": "B000000008"}])
        model = FakeModel()
        stats = ImportStats()
        with patch("app.services.import_products.get_embedding_model", return_value=model):
            process_import_batch(
                [
                    {"parent_asin": "B000000008", "title": "Existing item"},
                    {"parent_asin": "B000000009", "title": "New item"},
                ],
                stats,
                set(),
                collection,
                batch_size=2,
                insert_only=True,
            )

        self.assertEqual(stats.skipped, 1)
        self.assertEqual(stats.imported, 1)
        self.assertEqual(len(collection.operations), 1)
        update_doc = collection.operations[0]._doc
        self.assertIn("$setOnInsert", update_doc)
        self.assertNotIn("$set", update_doc)


if __name__ == "__main__":
    unittest.main()
