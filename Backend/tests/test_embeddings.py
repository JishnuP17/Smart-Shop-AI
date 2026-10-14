"""Tests for shared MiniLM embedding dimension and normalization checks."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from app.services.embeddings import EMBEDDING_DIMENSIONS, embed_texts


class FakeEmbeddingModel:
    def __init__(self, output):
        self.output = output

    def encode(self, texts, **_kwargs):
        return self.output


class EmbeddingTests(unittest.TestCase):
    def test_concurrent_model_access_loads_model_once(self):
        from app.services import embeddings

        model = Mock()
        model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIMENSIONS
        factory = Mock(return_value=model)
        with (
            patch.object(embeddings, "_model", None),
            patch.object(embeddings, "SentenceTransformer", factory),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            models = list(pool.map(lambda _index: embeddings.get_embedding_model(), range(16)))

        self.assertEqual(factory.call_count, 1)
        self.assertTrue(all(item is model for item in models))

    def test_embedding_vectors_are_384_dimensional_and_normalized(self):
        vectors = np.zeros((2, EMBEDDING_DIMENSIONS), dtype=np.float32)
        vectors[0, 0] = 3
        vectors[1, 1] = 4
        with patch("app.services.embeddings.get_embedding_model", return_value=FakeEmbeddingModel(vectors)):
            result = embed_texts(["one", "two"], batch_size=2)
        self.assertEqual(result.shape, (2, 384))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])

    def test_wrong_embedding_dimensions_are_rejected(self):
        with patch(
            "app.services.embeddings.get_embedding_model",
            return_value=FakeEmbeddingModel(np.zeros((1, 12), dtype=np.float32)),
        ):
            with self.assertRaisesRegex(ValueError, "unexpected shape"):
                embed_texts(["one"])


if __name__ == "__main__":
    unittest.main()
