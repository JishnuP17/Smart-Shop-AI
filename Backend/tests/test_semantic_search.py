"""Unit tests for semantic text preparation and cosine ranking."""

import unittest
from unittest.mock import patch

import numpy as np

from app.services import semantic_search


class SemanticSearchTests(unittest.TestCase):
    def test_explicit_gender_filter_only_excludes_conflicting_title_labels(self) -> None:
        from app.services.vector_search import _explicit_gender, _has_apparel_gender_conflict, _has_gender_conflict

        requested = _explicit_gender("men dress")
        self.assertEqual(requested, "men")
        self.assertTrue(_has_gender_conflict({"name": "Women's Maxi Dress"}, requested))
        self.assertFalse(_has_gender_conflict({"name": "Men's Dress Shirt"}, requested))
        self.assertFalse(_has_gender_conflict({"name": "Unisex Casual Shirt"}, requested))
        self.assertFalse(_has_gender_conflict({"name": "Women's Maxi Dress"}, None))
        self.assertTrue(_has_apparel_gender_conflict({"name": "Prom Dress Lace Gown"}, requested, "men dress"))
        self.assertFalse(_has_apparel_gender_conflict({"name": "Unisex Shirt"}, requested, "men shirt"))
        self.assertFalse(_has_apparel_gender_conflict({"name": "Wireless Headphones"}, requested, "men headphones"))

    def test_product_text_uses_available_catalog_fields(self) -> None:
        text = semantic_search.product_text(
            {"name": "Trail shoe", "category": "Footwear", "brand": "Northwind"}
        )
        self.assertEqual(text, "name: Trail shoe category: Footwear brand: Northwind")

    def test_search_ranks_by_cosine_similarity(self) -> None:
        query_vector = np.zeros(semantic_search.EMBEDDING_DIMENSIONS, dtype=np.float32)
        query_vector[0] = 1
        close_vector = query_vector.tolist()
        distant_vector = np.zeros(semantic_search.EMBEDDING_DIMENSIONS, dtype=np.float32).tolist()
        distant_vector[1] = 1
        with (
            patch("app.services.vector_search.load_current_index") as load_index,
            patch("app.services.vector_search.catalog_signature", return_value="signature"),
            patch("app.services.vector_search.embed_text", return_value=query_vector.tolist()),
            patch("app.services.vector_search.get_products_by_id_refs") as get_products,
        ):
            from types import SimpleNamespace

            load_index.return_value = SimpleNamespace(
                manifest={"catalog_signature": "signature"},
                index=SimpleNamespace(
                    ntotal=2,
                    search=lambda _query, _limit: (
                        np.asarray([[1.0, 0.0]], dtype=np.float32),
                        np.asarray([[0, 1]], dtype=np.int64),
                    ),
                ),
                mapping=[
                    {
                        "vector_position": 0,
                        "mongo_id": {"id_type": "string", "product_id": "close"},
                        "source_dataset": None,
                        "source_product_id": None,
                    },
                    {
                        "vector_position": 1,
                        "mongo_id": {"id_type": "string", "product_id": "distant"},
                        "source_dataset": None,
                        "source_product_id": None,
                    },
                ],
            )
            get_products.return_value = {
                "string:close": {"_id": "close", "name": "Match"},
                "string:distant": {"_id": "distant", "name": "Other"},
            }
            result = semantic_search.search_products("trail footwear", limit=5)

        self.assertEqual(result["results"][0]["product"]["name"], "Match")
        self.assertAlmostEqual(result["results"][0]["score"], 1.0)
        self.assertEqual(result["results"][1]["score"], 0.0)

if __name__ == "__main__":
    unittest.main()
