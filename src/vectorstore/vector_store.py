from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np


class VectorStore:
    """
    FAISS-backed vector store for PaperPilot.

    Responsibilities:
    - Store embeddings
    - Store metadata
    - Similarity search
    - Save/load index
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

        self.index = faiss.IndexFlatL2(embedding_dim)

        self.metadata: List[Dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> None:

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Embeddings and metadata must have same length"
            )

        embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)

        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        if self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty")

        query_embedding = (
            np.asarray(query_embedding)
            .astype(np.float32)
            .reshape(1, -1)
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, idx in zip(
            distances[0],
            indices[0]
        ):
            if idx < 0:
                continue

            results.append(
                {
                    "distance": float(distance),
                    "metadata": self.metadata[idx]
                }
            )

        return results

    def save(
        self,
        index_path: str,
        metadata_path: str
    ) -> None:

        faiss.write_index(
            self.index,
            index_path
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                self.metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

    @classmethod
    def load(
        cls,
        index_path: str,
        metadata_path: str
    ) -> "VectorStore":

        index = faiss.read_index(index_path)

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        store = cls(index.d)

        store.index = index
        store.metadata = metadata

        return store