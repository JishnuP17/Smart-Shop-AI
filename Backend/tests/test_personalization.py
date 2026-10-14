"""Unit tests for active demo profiles and user-based recommendations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services import database, personalization
from app.services.embeddings import EMBEDDING_DIMENSIONS


class FakeCollection:
    def __init__(self):
        self.documents = {}
        self.indexes = []

    def create_index(self, keys, **kwargs):
        if not any(existing[1].get("name") == kwargs.get("name") for existing in self.indexes):
            self.indexes.append((keys, kwargs))

    def update_one(self, query, update, upsert=False):
        key = query["_id"] if "_id" in query else (query["user_id"], query["event_id"])
        if key in self.documents:
            return SimpleNamespace(upserted_id=None)
        if not upsert:
            return SimpleNamespace(upserted_id=None)
        if "_id" in query:
            document = {**update["$setOnInsert"], "_id": key}
        else:
            document = dict(update["$setOnInsert"])
        self.documents[key] = document
        return SimpleNamespace(upserted_id=key)

    def find_one(self, query):
        key = query["_id"] if "_id" in query else (query.get("user_id"), query.get("event_id"))
        return self.documents.get(key)

    def find(self, query, projection=None):
        if "user_id" in query:
            docs = [doc for doc in self.documents.values() if doc.get("user_id") == query["user_id"]]
        elif "_id" in query and "$in" in query["_id"]:
            ids = set(query["_id"]["$in"])
            docs = [doc for key, doc in self.documents.items() if key in ids or doc.get("_id") in ids]
        else:
            docs = list(self.documents.values())
        return FakeCursor(docs)


class FakeCursor(list):
    def sort(self, *_args):
        return self

    def limit(self, limit):
        return FakeCursor(self[:limit])


class FakeDatabase(dict):
    def __getitem__(self, name):
        return self.setdefault(name, FakeCollection())


class FakeClient(dict):
    def __getitem__(self, name):
        return self.setdefault(name, FakeDatabase())


class SimilarityIndex:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal = len(vectors)

    def reconstruct(self, position):
        return self.vectors[position].copy()

    def search(self, query, count):
        scores = (query @ self.vectors.T)[0]
        positions = np.argsort(-scores)[:count]
        return scores[positions].reshape(1, -1), positions.astype(np.int64).reshape(1, -1)


def make_recommendation_fixture():
    vectors = np.zeros((3, EMBEDDING_DIMENSIONS), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 0], vectors[1, 1] = 0.9, 0.43589
    vectors[2, 1] = 1
    ids = ["baseline", "history_match", "clicked"]
    products = {}
    mapping = []
    for position, product_id in enumerate(ids):
        products[f"string:{product_id}"] = {
            "_id": product_id,
            "name": product_id,
            "source_dataset": None,
            "source_product_id": None,
        }
        mapping.append({
            "vector_position": position,
            "mongo_id": {"id_type": "string", "product_id": product_id},
            "source_dataset": None,
            "source_product_id": None,
        })
    bundle = SimpleNamespace(index=SimilarityIndex(vectors), mapping=mapping)
    return bundle, products


class PersonalizationTests(unittest.TestCase):
    def test_exactly_three_active_profiles_are_seeded_without_resetting_existing_or_history_data(self):
        client = FakeClient()
        database_name = "demo_test"
        users = client[database_name]["demo_users"]
        users.documents["demo_jordan"] = {
            "_id": "demo_jordan", "name": "Jordan", "interests": ["legacy preferences"],
        }
        interaction = {
            "event_id": "jordan-search-1", "user_id": "demo_jordan", "event_type": "search",
            "query": "Mobile", "timestamp": "historic",
        }
        interactions = client[database_name]["user_interactions"]
        interactions.documents[("demo_jordan", interaction["event_id"])] = interaction
        with (
            patch.object(database, "get_client", return_value=client),
            patch.object(database, "get_database_name", return_value=database_name),
        ):
            database.initialize_demo_user_storage(personalization.DEMO_PROFILES)
            self.assertEqual(len([profile for profile in personalization.DEMO_PROFILES if profile["_id"] in users.documents]), 3)
            users.documents["demo_alex"]["interests"] = ["preserved existing preference"]
            database.initialize_demo_user_storage(personalization.DEMO_PROFILES)

        self.assertEqual(personalization.DEMO_USER_IDS, ("demo_alex", "demo_sam", "demo_taylor"))
        self.assertEqual(users.documents["demo_alex"]["interests"], ["preserved existing preference"])
        self.assertIn("demo_jordan", users.documents)
        self.assertEqual(interactions.documents[("demo_jordan", "jordan-search-1")], interaction)
        self.assertEqual(len(client[database_name]["user_interactions"].indexes), 2)

    def test_interactions_are_idempotent_and_scoped_to_the_selected_user(self):
        client = FakeClient()
        with (
            patch.object(database, "get_client", return_value=client),
            patch.object(database, "get_database_name", return_value="demo_test"),
        ):
            event = {
                "event_id": "view-1", "user_id": "demo_alex", "event_type": "view",
                "product_id": "abc", "timestamp": "now",
            }
            first, inserted = database.save_demo_interaction(event)
            second, inserted_again = database.save_demo_interaction(event)
            alex_events = database.get_demo_interactions("demo_alex")
            sam_events = database.get_demo_interactions("demo_sam")

        self.assertTrue(inserted)
        self.assertFalse(inserted_again)
        self.assertEqual(first, second)
        self.assertEqual(len(alex_events), 1)
        self.assertEqual(sam_events, [])

    def test_initial_interests_give_profiles_different_ranked_products(self):
        bundle, products = make_recommendation_fixture()
        axis_x = np.eye(EMBEDDING_DIMENSIONS, dtype=np.float32)[0]
        axis_y = np.eye(EMBEDDING_DIMENSIONS, dtype=np.float32)[1]
        with (
            patch.object(personalization, "get_demo_user", side_effect=[
                {"id": "demo_alex", "name": "Alex", "interests": ["audio"]},
                {"id": "demo_taylor", "name": "Taylor", "interests": ["home"]},
            ]),
            patch.object(personalization, "get_demo_interactions", return_value=[]),
            patch.object(personalization, "_initial_preference", side_effect=[axis_x, axis_y]),
            patch.object(personalization, "_interest_vectors", side_effect=[axis_x.reshape(1, -1), axis_y.reshape(1, -1)]),
            patch.object(personalization, "load_current_index", return_value=bundle),
            patch.object(personalization, "_ensure_catalog_validated"),
            patch("app.services.database.get_products_by_id_refs", return_value=products),
        ):
            alex = personalization.get_personalized_recommendations("demo_alex", 1)
            taylor = personalization.get_personalized_recommendations("demo_taylor", 1)

        self.assertEqual(alex["recommendations"][0]["product"]["_id"], "baseline")
        self.assertEqual(taylor["recommendations"][0]["product"]["_id"], "clicked")
        self.assertEqual(alex["based_on"], "initial_interests")

    def test_click_history_changes_ranking_and_clicked_items_are_deprioritized(self):
        bundle, products = make_recommendation_fixture()
        base = np.eye(EMBEDDING_DIMENSIONS, dtype=np.float32)[0]
        events = [{"event_type": "click", "product_id": "clicked"}]
        with (
            patch.object(personalization, "get_demo_user", return_value={
                "id": "demo_alex", "name": "Alex", "interests": ["audio"],
            }),
            patch.object(personalization, "get_demo_interactions", return_value=events),
            patch.object(personalization, "_initial_preference", return_value=base),
            patch.object(personalization, "_interest_vectors", return_value=base.reshape(1, -1)),
            patch.object(personalization, "load_current_index", return_value=bundle),
            patch.object(personalization, "_ensure_catalog_validated"),
            patch("app.services.database.get_products_by_id_refs", return_value=products),
        ):
            result = personalization.get_personalized_recommendations("demo_alex", 2)

        names = [hit["product"]["_id"] for hit in result["recommendations"]]
        self.assertEqual(names[0], "history_match")
        self.assertNotEqual(names[0], "clicked")
        self.assertEqual(result["based_on"], "interests_and_interactions")

    def test_clicks_outweigh_views_and_search_queries_contribute_to_history(self):
        bundle, _products = make_recommendation_fixture()
        click_and_view = [
            {"event_type": "view", "product_id": "baseline"},
            {"event_type": "click", "product_id": "clicked"},
        ]
        history, _seen = personalization._interaction_preference(click_and_view, bundle)
        self.assertGreater(float(history[1]), float(history[0]))

        search_vector = np.eye(EMBEDDING_DIMENSIONS, dtype=np.float32)[0]
        with patch.object(personalization, "embed_texts", return_value=search_vector.reshape(1, -1)):
            search_history, _seen = personalization._interaction_preference(
                [{"event_type": "search", "query": "studio microphones"}], bundle
            )
        np.testing.assert_allclose(search_history, search_vector)

    def test_repeated_clicks_for_the_same_product_are_capped(self):
        bundle, _products = make_recommendation_fixture()
        event = {"event_type": "click", "product_id": "clicked"}
        single, _ = personalization._interaction_preference([event], bundle)
        repeated, _ = personalization._interaction_preference([event, event, event, event], bundle)
        np.testing.assert_allclose(repeated, single)

    def test_empty_index_returns_an_empty_recommendation_list(self):
        empty_bundle = SimpleNamespace(
            index=SimilarityIndex(np.zeros((0, EMBEDDING_DIMENSIONS), dtype=np.float32)),
            mapping=[],
        )
        base = np.eye(EMBEDDING_DIMENSIONS, dtype=np.float32)[0]
        with (
            patch.object(personalization, "get_demo_user", return_value={"id": "demo_alex"}),
            patch.object(personalization, "get_demo_interactions", return_value=[]),
            patch.object(personalization, "_initial_preference", return_value=base),
            patch.object(personalization, "load_current_index", return_value=empty_bundle),
            patch.object(personalization, "_ensure_catalog_validated"),
        ):
            result = personalization.get_personalized_recommendations("demo_alex", 5)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["indexed_products"], 0)

    def test_inactive_fourth_profile_is_not_listed_or_accepted(self):
        self.assertEqual(len(personalization.DEMO_PROFILES), 3)
        self.assertNotIn("demo_jordan", personalization.DEMO_USER_IDS)
        client = TestClient(app)
        active_profiles = [
            {
                "id": profile["_id"],
                "name": profile["name"],
                "description": profile["description"],
                "interests": profile["interests"],
            }
            for profile in personalization.DEMO_PROFILES
        ]
        with patch("app.routes.personalization.get_demo_users", return_value=active_profiles):
            response = client.get("/api/demo/users")
        self.assertEqual(response.status_code, 200)
        users = response.json()["items"]
        self.assertEqual({profile["id"] for profile in users}, set(personalization.DEMO_USER_IDS))
        self.assertEqual(client.get("/api/demo/users/demo_jordan/recommendations").status_code, 404)
        self.assertEqual(client.get("/api/demo/users/demo_jordan/interactions").status_code, 404)
        self.assertEqual(
            client.post(
                "/api/demo/users/demo_jordan/interactions",
                json={"event_id": "inactive", "event_type": "search", "query": "Mobile"},
            ).status_code,
            404,
        )

    def test_unknown_profile_and_invalid_interaction_references_are_rejected(self):
        client = TestClient(app)
        with patch("app.routes.personalization.get_demo_user", return_value=None):
            self.assertEqual(client.get("/api/demo/users/not-a-demo/recommendations").status_code, 404)
            response = client.post(
                "/api/demo/users/not-a-demo/interactions",
                json={"event_id": "bad", "event_type": "view", "product_id": "abc"},
            )
            self.assertEqual(response.status_code, 404)

        with (
            patch("app.routes.personalization.get_demo_user", return_value={"id": "demo_alex", "name": "Alex", "interests": []}),
            patch("app.routes.personalization.get_product_by_id", return_value=None),
        ):
            self.assertEqual(
                client.post(
                    "/api/demo/users/demo_alex/interactions",
                    json={"event_id": "missing-product", "event_type": "click", "product_id": "missing"},
                ).status_code,
                404,
            )
        self.assertEqual(
            client.post(
                "/api/demo/users/demo_alex/interactions",
                json={"event_id": "invalid", "event_type": "view"},
            ).status_code,
            422,
        )

    def test_valid_product_interaction_is_saved_and_history_is_user_scoped(self):
        client = TestClient(app)
        product = {"_id": "abc123", "name": "Audio product"}
        event = {"event_id": "click-1", "event_type": "click", "product_id": "abc123"}
        with (
            patch("app.routes.personalization.get_demo_user", return_value={"id": "demo_alex", "name": "Alex", "interests": []}),
            patch("app.routes.personalization.get_product_by_id", return_value=product),
            patch("app.routes.personalization.save_demo_interaction", return_value=({
                **event,
                "user_id": "demo_alex",
                "timestamp": "2026-01-01T00:00:00Z",
            }, True)) as save,
        ):
            response = client.post("/api/demo/users/demo_alex/interactions", json=event)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["recorded"])
        self.assertEqual(save.call_args.args[0]["user_id"], "demo_alex")

    def test_search_interaction_accepts_query_only_and_trims_query(self):
        client = TestClient(app)
        event = {"event_id": "search-1", "event_type": "search", "query": "  wireless headphones  "}
        with (
            patch("app.routes.personalization.get_demo_user", return_value={"id": "demo_alex", "name": "Alex", "interests": []}),
            patch("app.routes.personalization.save_demo_interaction", return_value=({
                **event,
                "query": "wireless headphones",
                "user_id": "demo_alex",
                "timestamp": "2026-01-01T00:00:00Z",
            }, True)) as save,
        ):
            response = client.post("/api/demo/users/demo_alex/interactions", json=event)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["query"], "wireless headphones")
        self.assertNotIn("product_id", save.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
