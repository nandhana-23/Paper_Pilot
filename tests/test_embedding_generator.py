"""
test_embedding_generator.py — Phase 3 unit tests

Because HuggingFace is not reachable in this environment, we test with a
StubModel that implements the exact same contract as SentenceTransformer:
  - encode(sentences, batch_size, ...) → np.ndarray of shape (N, 384)
  - dtype float32
  - normalised rows (unit vectors) when normalize_embeddings=True
  - preserves input order

All production code paths in EmbeddingGenerator are exercised; the only
thing not tested is the SentenceTransformer download itself.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.chunking.text_chunker import Chunk, ChunkMetadata
from src.embeddings.embedding_generator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    EmbeddedChunk,
    EmbeddingGenerator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _make_chunk(text: str, page: int = 1, index: int = 0) -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        page_number=page,
        text=text,
        metadata=ChunkMetadata(
            source_path=Path("/fake/paper.pdf"),
            page_number=page,
            chunk_index=index,
            total_chunks=0,
        ),
    )


class StubSentenceTransformer:
    """Minimal stand-in for sentence_transformers.SentenceTransformer.

    Produces deterministic, unit-normalised vectors. Each row is seeded
    from the hash of the input text, so identical texts get identical
    vectors — which is the real model's behaviour.
    """

    def __init__(self, model_name_or_path=None, device=None, **kwargs):
        self.model_name_or_path = model_name_or_path
        # Mimic the .device attribute the real model exposes
        self.device = device or "cpu"

    def encode(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        convert_to_tensor=False,
        normalize_embeddings=False,
        **kwargs,
    ) -> np.ndarray:
        rows = []
        for s in sentences:
            rng = np.random.default_rng(abs(hash(s)) % (2**32))
            vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
            if normalize_embeddings:
                vec /= np.linalg.norm(vec) + 1e-12
            rows.append(vec)
        return np.array(rows, dtype=np.float32)


def _make_generator(**kwargs) -> EmbeddingGenerator:
    """Build an EmbeddingGenerator with the StubModel injected."""
    with patch(
        "src.embeddings.embedding_generator.SentenceTransformer",
        StubSentenceTransformer,
    ):
        return EmbeddingGenerator(**kwargs)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_params(self):
        gen = _make_generator()
        assert gen.model_name == DEFAULT_MODEL_NAME
        assert gen.batch_size == DEFAULT_BATCH_SIZE
        assert gen.normalize_embeddings is True

    def test_custom_params(self):
        gen = _make_generator(batch_size=16, normalize_embeddings=False)
        assert gen.batch_size == 16
        assert gen.normalize_embeddings is False

    def test_invalid_batch_size_type(self):
        with pytest.raises(TypeError, match="batch_size must be an int"):
            _make_generator(batch_size="32")

    def test_invalid_batch_size_zero(self):
        with pytest.raises(ValueError, match="positive integer"):
            _make_generator(batch_size=0)

    def test_invalid_batch_size_negative(self):
        with pytest.raises(ValueError, match="positive integer"):
            _make_generator(batch_size=-1)


# ---------------------------------------------------------------------------
# generate() happy-path tests
# ---------------------------------------------------------------------------

class TestGenerate:
    def setup_method(self):
        self.gen = _make_generator()

    def test_output_length_matches_input(self):
        chunks = [_make_chunk(f"sentence {i}") for i in range(7)]
        result = self.gen.generate(chunks)
        assert len(result) == 7

    def test_all_items_are_embedded_chunks(self):
        chunks = [_make_chunk("text")]
        result = self.gen.generate(chunks)
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)

    def test_embedding_shape(self):
        chunks = [_make_chunk("hello world")]
        result = self.gen.generate(chunks)
        ec = result[0]
        assert ec.embedding.shape == (EMBEDDING_DIM,)

    def test_embedding_dtype_is_float32(self):
        chunks = [_make_chunk("some text")]
        ec = self.gen.generate(chunks)[0]
        assert ec.embedding.dtype == np.float32

    def test_embedding_is_contiguous(self):
        """FAISS and other ANN libraries require contiguous arrays."""
        chunks = [_make_chunk("contiguous check")]
        ec = self.gen.generate(chunks)[0]
        assert ec.embedding.flags["C_CONTIGUOUS"]

    def test_embedding_dim_field_matches_actual(self):
        chunks = [_make_chunk("dim check")]
        ec = self.gen.generate(chunks)[0]
        assert ec.embedding_dim == len(ec.embedding) == EMBEDDING_DIM

    def test_model_name_is_recorded(self):
        chunks = [_make_chunk("model name check")]
        ec = self.gen.generate(chunks)[0]
        assert ec.model_name == DEFAULT_MODEL_NAME

    def test_order_preserved(self):
        """Output[i] must correspond to input[i]."""
        texts = [f"unique sentence number {i}" for i in range(10)]
        chunks = [_make_chunk(t, index=i) for i, t in enumerate(texts)]
        result = self.gen.generate(chunks)
        for i, (chunk, ec) in enumerate(zip(chunks, result)):
            assert ec.chunk_id == chunk.chunk_id, f"Order mismatch at index {i}"
            assert ec.text == chunk.text

    def test_metadata_is_preserved(self):
        chunk = _make_chunk("metadata test", page=5, index=3)
        ec = self.gen.generate([chunk])[0]
        assert ec.page_number == 5
        assert ec.metadata.chunk_index == 3
        assert ec.metadata.source_path == Path("/fake/paper.pdf")

    def test_normalised_embeddings_are_unit_vectors(self):
        gen = _make_generator(normalize_embeddings=True)
        chunks = [_make_chunk(f"text {i}") for i in range(5)]
        result = gen.generate(chunks)
        for ec in result:
            norm = float(np.linalg.norm(ec.embedding))
            assert abs(norm - 1.0) < 1e-5, f"Norm {norm:.6f} is not unit"

    def test_unnormalised_embeddings_vary_in_magnitude(self):
        gen = _make_generator(normalize_embeddings=False)
        chunks = [_make_chunk(f"text {i}") for i in range(5)]
        result = gen.generate(chunks)
        norms = [float(np.linalg.norm(ec.embedding)) for ec in result]
        # With random vectors, norms should not all be exactly 1.0
        assert not all(abs(n - 1.0) < 1e-5 for n in norms)

    def test_single_chunk(self):
        """Single-item list must still return a 2-D matrix internally."""
        result = self.gen.generate([_make_chunk("solo")])
        assert len(result) == 1
        assert result[0].embedding.shape == (EMBEDDING_DIM,)

    def test_large_batch(self):
        """More chunks than batch_size should still produce correct output."""
        gen = _make_generator(batch_size=3)
        chunks = [_make_chunk(f"chunk {i}", index=i) for i in range(10)]
        result = gen.generate(chunks)
        assert len(result) == 10
        for i, (c, ec) in enumerate(zip(chunks, result)):
            assert ec.chunk_id == c.chunk_id

    def test_identical_texts_produce_identical_vectors(self):
        """Same text → same embedding (determinism check)."""
        text = "The Transformer model uses self-attention."
        chunks = [_make_chunk(text), _make_chunk(text)]
        result = self.gen.generate(chunks)
        np.testing.assert_array_equal(result[0].embedding, result[1].embedding)

    def test_different_texts_produce_different_vectors(self):
        chunks = [_make_chunk("cat"), _make_chunk("quantum entanglement")]
        result = self.gen.generate(chunks)
        assert not np.array_equal(result[0].embedding, result[1].embedding)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def setup_method(self):
        self.gen = _make_generator()

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            self.gen.generate([])

    def test_non_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            self.gen.generate("not a list")

    def test_list_with_non_chunk_raises_type_error(self):
        chunks = [_make_chunk("ok"), "not a chunk", _make_chunk("ok too")]
        with pytest.raises(TypeError, match="index 1"):
            self.gen.generate(chunks)

    def test_list_with_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Chunk instances"):
            self.gen.generate([None])


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_model_failure_raises_runtime_error(self):
        """If the underlying model raises, we re-raise as RuntimeError."""
        gen = _make_generator()

        def exploding_encode(*args, **kwargs):
            raise OSError("CUDA out of memory")

        gen._model.encode = exploding_encode

        with pytest.raises(RuntimeError, match="Model encoding failed"):
            gen.generate([_make_chunk("trigger failure")])


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s [%(levelname)s] %(message)s",
    )

    print("Running EmbeddingGenerator tests with StubModel...\n")

    suites = [
        TestConstruction,
        TestGenerate,
        TestValidation,
        TestErrorHandling,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite_cls) if m.startswith("test_")]
        print(f"  {suite_cls.__name__} ({len(methods)} tests)")
        for method_name in methods:
            if hasattr(suite, "setup_method"):
                suite.setup_method()
            try:
                getattr(suite, method_name)()
                print(f"    ✓ {method_name}")
                passed += 1
            except Exception as exc:
                print(f"    ✗ {method_name}: {exc}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
