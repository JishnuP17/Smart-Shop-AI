"""FastAPI response validation tests for semantic product search."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.vector_search import (
    ProductNotFound,
    ProductNotIndexed,
    VectorIndexStale,
    VectorIndexUnavailable,
)


class ProductRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_search_response_preserves_score_and_product_fields(self):
        search_result = {
            "query": "wireless audio for travel",
            "indexed_products": 1,
            "results": [
                {
                    "product": {
                        "_id": "abc123",
                        "name": "Travel headphones",
                        "price": 49.99,
                        "category": "Electronics",
                        "image_url": "https://example.test/image.jpg",
                    },
                    "score": 0.812,
                }
            ],
        }
        with patch("app.routes.products.search_products", return_value=search_result):
            response = self.client.get(
                "/api/products/search",
                params={"q": "wireless audio for travel", "limit": 10},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["results"][0]["product"]["_id"], "abc123")
        self.assertEqual(body["results"][0]["product"]["name"], "Travel headphones")
        self.assertAlmostEqual(body["results"][0]["score"], 0.812)

    def test_empty_query_is_rejected(self):
        response = self.client.get("/api/products/search", params={"q": "  "})
        self.assertEqual(response.status_code, 422)

    def test_stale_or_missing_index_returns_clear_service_error(self):
        for error in (
            VectorIndexStale("Catalogue changed; rebuild FAISS."),
            VectorIndexUnavailable("FAISS index is missing."),
        ):
            with patch("app.routes.products.search_products", side_effect=error):
                response = self.client.get("/api/products/search", params={"q": "headphones"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("FAISS", response.json()["detail"])

    def test_recommendations_return_schema_and_requested_limit(self):
        result = {
            "product_id": "abc123",
            "indexed_products": 3,
            "recommendations": [
                {"product": {"_id": "def456", "name": "Similar product", "price": 10.0}, "score": 0.93}
            ],
        }
        with patch("app.routes.products.recommend_indexed_products", return_value=result) as recommend:
            response = self.client.get("/api/products/abc123/recommendations?limit=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["product_id"], "abc123")
        self.assertEqual(len(response.json()["recommendations"]), 1)
        self.assertEqual(response.json()["recommendations"][0]["product"]["name"], "Similar product")
        recommend.assert_called_once_with("abc123", 1)

    def test_recommendation_limit_validation_and_missing_cases(self):
        self.assertEqual(
            self.client.get("/api/products/abc/recommendations?limit=0").status_code,
            422,
        )
        for error, expected_status in ((ProductNotFound("missing"), 404), (ProductNotIndexed("unindexed"), 404)):
            with patch("app.routes.products.recommend_indexed_products", side_effect=error):
                response = self.client.get("/api/products/abc/recommendations")
            self.assertEqual(response.status_code, expected_status)

    def test_categories_are_returned_from_database_service(self):
        with patch(
            "app.routes.products.get_product_categories",
            return_value=[{"name": "Electronics", "count": 12}, {"name": "Books", "count": 4}],
        ):
            response = self.client.get("/api/products/categories")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0], {"name": "Electronics", "count": 12})

    def test_catalog_category_filter_is_forwarded_to_database(self):
        with patch("app.routes.products.get_products", return_value=[{"_id": "one", "name": "Test"}]) as get_page:
            response = self.client.get("/api/products", params={"category": "Electronics", "limit": 5})
        self.assertEqual(response.status_code, 200)
        get_page.assert_called_once_with(skip=0, limit=5, category="Electronics")

    def test_product_detail_route_returns_the_requested_catalog_product(self):
        product = {"_id": "abc123", "name": "Real catalog product", "category": "Electronics"}
        with patch("app.routes.products.get_product_by_id", return_value=product) as get_product:
            response = self.client.get("/api/products/abc123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), product)
        get_product.assert_called_once_with("abc123")

    def test_product_detail_route_returns_404_for_missing_product(self):
        with patch("app.routes.products.get_product_by_id", return_value=None):
            response = self.client.get("/api/products/missing")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
