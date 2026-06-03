from __future__ import annotations

from typing import List, Dict, Any

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectorstore.vector_store import VectorStore


class Retriever:
    """
    Retrieval layer for PaperPilot.

    Responsibilities:
    - Convert user query to embedding
    - Search vector store
    - Return top-k results
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore
    ):

        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        query_embedding = (
            self.embedding_generator.model.encode(
                [query]
            )
        )

        results = self.vector_store.search(
            query_embedding[0],
            top_k=top_k
        )

        return results