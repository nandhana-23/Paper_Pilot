import tempfile
import unittest

import numpy as np

from src.vectorstore.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):

    def setUp(self):

        self.dimension = 384

        self.store = VectorStore(
            embedding_dim=self.dimension
        )

        self.embeddings = np.random.rand(
            3,
            self.dimension
        ).astype(np.float32)

        self.metadata = [
            {"chunk": "chunk1"},
            {"chunk": "chunk2"},
            {"chunk": "chunk3"},
        ]

    def test_add_embeddings(self):

        self.store.add(
            self.embeddings,
            self.metadata
        )

        self.assertEqual(
            self.store.index.ntotal,
            3
        )

    def test_search(self):

        self.store.add(
            self.embeddings,
            self.metadata
        )

        query = self.embeddings[0]

        results = self.store.search(
            query,
            top_k=2
        )

        self.assertEqual(
            len(results),
            2
        )

    def test_save_and_load(self):

        self.store.add(
            self.embeddings,
            self.metadata
        )

        with tempfile.TemporaryDirectory() as temp_dir:

            index_file = (
                f"{temp_dir}/index.faiss"
            )

            metadata_file = (
                f"{temp_dir}/metadata.json"
            )

            self.store.save(
                index_file,
                metadata_file
            )

            loaded_store = (
                VectorStore.load(
                    index_file,
                    metadata_file
                )
            )

            self.assertEqual(
                loaded_store.index.ntotal,
                3
            )

    def test_empty_search_raises(self):

        with self.assertRaises(RuntimeError):
            self.store.search(
                np.random.rand(
                    self.dimension
                )
            )


if __name__ == "__main__":
    unittest.main()