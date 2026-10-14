"""Tests for the persistent FAISS index and explicit MongoDB ID mapping."""

import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from bson import ObjectId

from app.config import settings
from app.services import vector_search
from app.services.embeddings import EMBEDDING_DIMENSIONS, product_content_hash


def make_catalogue():
    first = {
        "_id": ObjectId("000000000000000000000001"),
        "name": "Wireless travel headphones",
        "description": "Noise cancelling audio",
        "category": "Electronics",
        "brand": "Sound Co",
        "source_dataset": "amazon-test",
        "source_product_id": "A1",
    }
    second = {
        "_id": ObjectId("000000000000000000000002"),
        "name": "Cotton garden gloves",
        "category": "Garden",
        "brand": "Outdoor Co",
        "source_dataset": "amazon-test",
        "source_product_id": "A2",
    }
    for product, coordinate in ((first, 0), (second, 1)):
        vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        vector[coordinate] = 1
        product["_embedding"] = vector.tolist()
        product["_embedding_model"] = settings.embedding_model
        product["_source_content_hash"] = product_content_hash(product)
    return [first, second]


def signature_for(products):
    digest = hashlib.sha256()
    for product in products:
        vector_search._catalog_digest_update(digest, product)
    return digest.hexdigest()


class VectorSearchTests(unittest.TestCase):
    def test_concurrent_searches_load_index_generation_once(self):
        generation = "generation-" + "a" * 32
        loaded = SimpleNamespace(generation=generation)
        with tempfile.TemporaryDirectory() as directory:
            pointer = Path(directory) / "CURRENT"
            pointer.write_text(generation, encoding="utf-8")
            with (
                patch.object(vector_search, "CURRENT_FILE", pointer),
                patch.object(vector_search, "_loaded_bundle", None),
                patch.object(vector_search, "_load_generation", return_value=loaded) as loader,
                ThreadPoolExecutor(max_workers=8) as pool,
            ):
                bundles = list(pool.map(lambda _index: vector_search.load_current_index(), range(16)))

        self.assertEqual(loader.call_count, 1)
        self.assertTrue(all(bundle is loaded for bundle in bundles))

    def test_persistent_index_reloads_and_returns_correct_mongo_product(self):
        products = make_catalogue()
        signature = signature_for(products)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "faiss"
            with (
                patch.object(vector_search, "FAISS_ROOT", root),
                patch.object(vector_search, "GENERATIONS_ROOT", root / "generations"),
                patch.object(vector_search, "CURRENT_FILE", root / "CURRENT"),
                patch.object(vector_search, "LOCK_FILE", root / ".write.lock"),
                patch.object(vector_search, "iter_catalog_products", side_effect=lambda **_kwargs: iter(products)),
                patch.object(vector_search, "catalog_signature", return_value=signature),
                patch.object(vector_search, "finalize_vector_migration"),
            ):
                built = vector_search.rebuild_faiss_index(batch_size=2, cleanup_mongo_embeddings=False)
                vector_search.finalize_vector_migration.assert_not_called()
                vector_search._loaded_bundle = None
                reloaded = vector_search.load_current_index(force_reload=True)

                result_products = {
                    f"objectid:{product['_id']}": product for product in products
                }
                query_vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
                query_vector[0] = 1
                with (
                    patch.object(vector_search, "embed_text", return_value=query_vector.tolist()),
                    patch.object(vector_search, "get_products_by_id_refs", return_value=result_products),
                ):
                    results = vector_search.search_products("wireless audio", limit=2)

            self.assertEqual(built["indexed"], 2)
            self.assertEqual(reloaded.index.d, EMBEDDING_DIMENSIONS)
            self.assertEqual(reloaded.index.ntotal, 2)
            self.assertEqual(len(reloaded.mapping), 2)
            self.assertEqual(results["results"][0]["product"]["name"], "Wireless travel headphones")
            self.assertAlmostEqual(results["results"][0]["score"], 1.0)

    def test_missing_index_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(vector_search, "CURRENT_FILE", Path(directory) / "missing"),
                patch.object(vector_search, "_loaded_bundle", None),
            ):
                with self.assertRaisesRegex(vector_search.VectorIndexUnavailable, "FAISS index is missing"):
                    vector_search.load_current_index(force_reload=True)

    def test_stale_catalogue_fails_instead_of_using_old_vectors(self):
        products = make_catalogue()
        signature = signature_for(products)
        bundle = type(
            "Bundle",
            (),
            {
                "manifest": {"catalog_signature": "old-signature"},
                "index": type("Index", (), {"ntotal": 2})(),
                "mapping": [],
            },
        )()
        with (
            patch.object(vector_search, "load_current_index", return_value=bundle),
            patch.object(vector_search, "catalog_signature", return_value=signature),
            patch.object(vector_search, "_catalog_validated", False),
            patch.object(vector_search, "_catalog_stale", False),
        ):
            with self.assertRaises(vector_search.VectorIndexStale):
                vector_search.search_products("headphones", limit=5)

    def test_empty_index_returns_empty_results(self):
        bundle = type(
            "Bundle",
            (),
            {"manifest": {"catalog_signature": "empty"}, "index": type("Index", (), {"ntotal": 0})()},
        )()
        with (
            patch.object(vector_search, "load_current_index", return_value=bundle),
            patch.object(vector_search, "catalog_signature", return_value="empty"),
            patch.object(vector_search, "embed_text", side_effect=AssertionError("empty index should not embed")),
        ):
            result = vector_search.search_products("anything", limit=10)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["indexed_products"], 0)

    def test_missing_mongo_product_from_mapping_is_reported_as_stale(self):
        product = make_catalogue()[0]
        index = type(
            "Index",
            (),
            {
                "ntotal": 1,
                "search": lambda _self, _query, _limit: (
                    np.asarray([[0.5]], dtype=np.float32),
                    np.asarray([[0]], dtype=np.int64),
                ),
            },
        )()
        bundle = type(
            "Bundle",
            (),
            {
                "manifest": {"catalog_signature": "same"},
                "index": index,
                "mapping": [
                    {
                        "vector_position": 0,
                        "mongo_id": {"id_type": "objectid", "product_id": str(product["_id"])},
                        "source_dataset": product["source_dataset"],
                        "source_product_id": product["source_product_id"],
                    }
                ],
            },
        )()
        query_vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        query_vector[0] = 1
        with (
            patch.object(vector_search, "load_current_index", return_value=bundle),
            patch.object(vector_search, "catalog_signature", return_value="same"),
            patch.object(vector_search, "embed_text", return_value=query_vector.tolist()),
            patch.object(vector_search, "get_products_by_id_refs", return_value={}),
        ):
            with self.assertRaisesRegex(vector_search.VectorIndexStale, "missing from MongoDB"):
                vector_search.search_products("headphones", limit=1)

    def test_recommendations_use_mapping_exclude_source_and_deduplicate(self):
        source, first = make_catalogue()
        second = {
            "_id": ObjectId("000000000000000000000003"),
            "name": "Bluetooth travel speaker",
            "source_dataset": "amazon-test",
            "source_product_id": "A3",
        }
        third = {
            "_id": ObjectId("000000000000000000000004"),
            "name": "Wireless earbud set",
            "source_dataset": "amazon-test",
            "source_product_id": "A4",
        }
        # FAISS returns nearest neighbors in descending similarity order.
        index = type(
            "Index",
            (),
            {
                "ntotal": 4,
                "reconstruct": lambda _self, position: np.eye(EMBEDDING_DIMENSIONS, dtype=np.float32)[position],
                "search": lambda _self, _query, _limit: (
                    np.asarray([[1.0, 0.5, 0.0, -1.0]], dtype=np.float32),
                    np.asarray([[0, 2, 3, 1]], dtype=np.int64),
                ),
            },
        )()
        mapping = [
            {
                "vector_position": position,
                "mongo_id": {"id_type": "objectid", "product_id": str(product["_id"])},
                "source_dataset": product["source_dataset"],
                "source_product_id": product["source_product_id"],
            }
            for position, product in enumerate((source, first, second, third))
        ]
        bundle = type(
            "Bundle",
            (),
            {
                "manifest": {"catalog_signature": "same"},
                "index": index,
                "mapping": mapping,
            },
        )()
        products = {f"objectid:{product['_id']}": product for product in (source, first, second, third)}
        with (
            patch.object(vector_search, "load_current_index", return_value=bundle),
            patch.object(vector_search, "catalog_signature", return_value="same"),
            patch.object(
                vector_search,
                "get_products_by_id_refs",
                side_effect=lambda refs: {
                    vector_search._ref_key(ref): products[vector_search._ref_key(ref)]
                    for ref in refs
                    if vector_search._ref_key(ref) in products
                },
            ),
        ):
            result = vector_search.recommend_products(str(source["_id"]), limit=5)

        self.assertEqual(
            [hit["product"]["name"] for hit in result["recommendations"]],
            ["Bluetooth travel speaker", "Wireless earbud set", "Cotton garden gloves"],
        )
        self.assertNotIn(str(source["_id"]), [hit["product"]["_id"] for hit in result["recommendations"]])

    def test_recommendations_report_missing_and_unindexed_products(self):
        bundle = type(
            "Bundle",
            (),
            {
                "manifest": {"catalog_signature": "same"},
                "index": type("Index", (), {"ntotal": 0})(),
                "mapping": [],
            },
        )()
        with (
            patch.object(vector_search, "load_current_index", return_value=bundle),
            patch.object(vector_search, "catalog_signature", return_value="same"),
            patch.object(
                vector_search,
                "get_products_by_id_refs",
                return_value={},
            ),
        ):
            with self.assertRaises(vector_search.ProductNotFound):
                vector_search.recommend_products("000000000000000000000099", limit=5)

        product = make_catalogue()[0]
        bundle = type(
            "Bundle",
            (),
            {
                "manifest": {"catalog_signature": "same"},
                "index": type("Index", (), {"ntotal": 1})(),
                "mapping": [],
            },
        )()
        def lookup_product(refs):
            return (
                {f"objectid:{product['_id']}": product}
                if any(vector_search._ref_key(ref) == f"objectid:{product['_id']}" for ref in refs)
                else {}
            )

        with (
            patch.object(vector_search, "load_current_index", return_value=bundle),
            patch.object(vector_search, "catalog_signature", return_value="same"),
            patch.object(vector_search, "get_products_by_id_refs", side_effect=lookup_product),
        ):
            with self.assertRaises(vector_search.ProductNotIndexed):
                vector_search.recommend_products(str(product["_id"]), limit=5)


if __name__ == "__main__":
    unittest.main()
