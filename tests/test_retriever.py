import unittest
import numpy as np

from src.vectorstore.vector_store import VectorStore
from src.retrieval.retriever import Retriever


class MockEmbeddingGenerator:

    class MockModel:

        def encode(self, texts):

            return np.random.rand(
                len(texts),
                384
            ).astype(np.float32)

    def __init__(self):

        self.model = self.MockModel()


class TestRetriever(unittest.TestCase):

    def setUp(self):

        self.embedding_generator = (
            MockEmbeddingGenerator()
        )

        self.store = VectorStore(
            embedding_dim=384
        )

        embeddings = np.random.rand(
            10,
            384
        ).astype(np.float32)

        metadata = [
            {"chunk_id": i}
            for i in range(10)
        ]

        self.store.add(
            embeddings,
            metadata
        )

        self.retriever = Retriever(
            self.embedding_generator,
            self.store
        )

    def test_retrieve(self):

        results = self.retriever.retrieve(
            "self attention",
            top_k=3
        )

        self.assertEqual(
            len(results),
            3
        )

    def test_empty_query(self):

        with self.assertRaises(
            ValueError
        ):
            self.retriever.retrieve("")