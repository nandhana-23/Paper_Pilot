"""
embedding_generator.py — PaperPilot Phase 3: Embedding Generation Module

Responsibility: Accept a list of Chunk objects produced by Phase 2, run
each chunk's text through a sentence-transformer model in configurable
batches, and return a list of EmbeddedChunk objects — each carrying the
original chunk's text, all of its metadata, and its dense vector
representation.

Design philosophy
-----------------
Embeddings are the bridge between raw text and semantic search. This layer
must be:

  * **Faithful** — every vector must be traceable back to its exact source
    chunk, page number, and PDF path. Metadata is never dropped.
  * **Efficient** — embedding one chunk at a time is 10–50× slower than
    batching on CPU; far worse on GPU. Batching is mandatory, not optional.
  * **Transparent** — the model name and embedding dimensionality are stored
    on every result so that downstream consumers (and future debugging
    sessions) know exactly which model produced a given vector, and can
    detect mismatches before attempting a cosine search.
  * **Narrow** — this module does not write to any database, does not call
    any LLM, and does not perform retrieval. It is a pure transformation:
    List[Chunk] → List[EmbeddedChunk].

Model choice: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional output — small enough to store cheaply, expressive
    enough for semantic similarity on scientific text.
  - Trained on over 1 billion sentence pairs; strong out-of-the-box recall.
  - ~23 MB on disk; loads in < 2 s on CPU. No GPU required.
  - Widely used as the default RAG embedding model in production.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List
import torch

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking.text_chunker import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Default model — pinned to a specific revision in production; here we use
# the canonical HuggingFace identifier. Override at construction time to
# swap models without changing call sites.
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# 32 is the sentence-transformers library default and a solid CPU baseline.
# Increase to 64–128 on GPU; decrease on memory-constrained systems.
DEFAULT_BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    """A Chunk paired with its dense vector embedding.

    This is the primary output type of Phase 3 and the primary *input* type
    for Phase 4 (vector store ingestion). Every field that Phase 4 could
    possibly need is present here; it must not have to re-open the PDF or
    re-run the chunker.

    Attributes
    ----------
    chunk_id:
        UUID4 string inherited unchanged from the source :class:`Chunk`.
        Provides a stable cross-phase identifier for deduplication and
        incremental updates.
    page_number:
        1-based page number, promoted from metadata for fast access.
    text:
        Original chunk text exactly as chunked — not truncated, not
        normalised further. The vector must match this text.
    embedding:
        Dense float32 numpy array of shape ``(embedding_dim,)``.
        Stored as a 1-D array (not a 2-D row vector) for ergonomic
        downstream use: ``np.dot(a.embedding, b.embedding)`` just works.
    embedding_dim:
        Length of the embedding vector. Stored explicitly so a consumer
        can validate dimensionality without loading numpy.
    model_name:
        Exact model identifier used to produce this embedding. Critical
        for correctness: querying a FAISS index built with model A using
        a vector from model B produces silently wrong results. Recording
        the model name enables a fast mismatch check.
    metadata:
        Full provenance record from Phase 2 — source path, chunk index,
        total chunks. Unchanged from the original Chunk.
    """

    chunk_id: str
    page_number: int
    text: str
    embedding: np.ndarray
    embedding_dim: int
    model_name: str
    metadata: ChunkMetadata = field(repr=False)

    def __repr__(self) -> str:
        # numpy arrays in the default repr are noisy; replace with a summary.
        return (
            f"EmbeddedChunk("
            f"chunk_id={self.chunk_id!r}, "
            f"page_number={self.page_number}, "
            f"embedding_dim={self.embedding_dim}, "
            f"model={self.model_name!r}, "
            f"text={self.text[:40]!r}…"
            f")"
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """Embeds a list of :class:`~src.chunking.text_chunker.Chunk` objects
    using a sentence-transformers model.

    The model is loaded **once** at construction time and reused across
    all calls to :meth:`generate`. This is intentional: model loading is
    expensive (disk I/O, weight initialisation, tokeniser build). Paying
    that cost once per generator instance — not once per batch or call —
    is the correct production pattern.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local path. Defaults to
        ``sentence-transformers/all-MiniLM-L6-v2``.
    batch_size:
        Number of chunks to embed per forward pass. Larger batches are
        faster but use more memory. Must be a positive integer.
    normalize_embeddings:
        When ``True``, each embedding vector is L2-normalised to unit
        length before returning. Unit vectors reduce cosine similarity
        to a dot product, which is faster to compute and required by
        several ANN libraries (e.g. FAISS ``IndexFlatIP``).
        Set ``False`` if you need raw (unnormalised) distances.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        ``None`` lets sentence-transformers auto-detect.

    Usage::

        from src.ingestion.pdf_loader import PDFLoader
        from src.chunking.text_chunker import TextChunker
        from src.embeddings.embedding_generator import EmbeddingGenerator

        chunks = TextChunker().chunk(PDFLoader().load("paper.pdf"))
        generator = EmbeddingGenerator()
        embedded = generator.generate(chunks)

        for ec in embedded:
            print(ec.chunk_id, ec.embedding.shape, ec.text[:50])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        self._validate_config(batch_size)

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        logger.info("Loading embedding model: %s", model_name)
        t0 = time.perf_counter()

        # SentenceTransformer.__init__ accepts local_files_only, cache_folder,
        # and device. We pass device explicitly so the caller has full control;
        # None triggers auto-detection (GPU if available, else CPU).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(
            model_name,
            device=device
        )
        logger.info(f"Model loaded on device: {device}")

        elapsed = time.perf_counter() - t0
        logger.info(
            "Model loaded in %.2fs — device: %s | batch_size: %d | "
            "normalize: %s",
            elapsed,
            self._model.device,
            self.batch_size,
            self.normalize_embeddings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Embed all chunks and return :class:`EmbeddedChunk` objects.

        The method:

        1. Validates the input list.
        2. Extracts chunk texts in order (preserving chunk-to-index mapping).
        3. Calls the model in batches, collecting a 2-D numpy array of
           shape ``(n_chunks, embedding_dim)``.
        4. Pairs each row back with its source chunk and builds
           :class:`EmbeddedChunk` objects.

        The pairing in step 4 relies on the fact that
        ``sentence_transformers.SentenceTransformer.encode`` preserves input
        order — this is guaranteed by the library and verified in our tests.

        Args:
            chunks: Ordered list of :class:`~src.chunking.text_chunker.Chunk`
                    objects from Phase 2. Must be non-empty.

        Returns:
            List of :class:`EmbeddedChunk` objects in the same order as
            *chunks*. Every input chunk appears exactly once in the output.

        Raises:
            TypeError:  If *chunks* is not a list or contains non-Chunk items.
            ValueError: If *chunks* is empty.
            RuntimeError: If the model fails to produce embeddings.
        """
        self._validate_chunks(chunks)

        n = len(chunks)
        logger.info(
            "Generating embeddings — model: %s | chunks: %d | batch_size: %d",
            self.model_name,
            n,
            self.batch_size,
        )

        texts: List[str] = [c.text for c in chunks]

        t0 = time.perf_counter()
        matrix = self._encode_batched(texts)
        elapsed = time.perf_counter() - t0

        embedding_dim: int = matrix.shape[1]

        logger.info(
            "Encoding complete — chunks: %d | dim: %d | "
            "elapsed: %.2fs | throughput: %.1f chunks/s",
            n,
            embedding_dim,
            elapsed,
            n / elapsed if elapsed > 0 else float("inf"),
        )

        embedded = self._pair_embeddings(chunks, matrix, embedding_dim)

        # Back-fill model_name now that we're back in instance scope.
        # _pair_embeddings is a pure static function (no `self`) so the
        # model name is injected here rather than passed as a parameter —
        # a deliberate separation of concerns.
        for ec in embedded:
            ec.model_name = self.model_name

        return embedded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_batched(self, texts: List[str]) -> np.ndarray:
        """Run all texts through the model and return a 2-D float32 array.

        We delegate batching entirely to ``SentenceTransformer.encode``
        rather than slicing manually. The library handles:
          - splitting *texts* into batches of ``batch_size``
          - moving tensors to the correct device
          - padding / truncating to the model's max sequence length
          - concatenating results back into a single array

        We set ``convert_to_numpy=True`` and ``convert_to_tensor=False``
        so the return value is always a ``np.ndarray`` — not a torch
        Tensor — keeping this module free of a hard PyTorch runtime
        dependency in the public interface.

        ``show_progress_bar`` is set to ``False`` because progress output
        belongs to a CLI layer, not a library module. Callers that want
        progress should wrap this call in ``tqdm`` themselves.

        Args:
            texts: Ordered list of strings to embed. Non-empty.

        Returns:
            Float32 numpy array of shape ``(len(texts), embedding_dim)``.

        Raises:
            RuntimeError: On any model-level failure.
        """
        try:
            result: np.ndarray = self._model.encode(
                sentences=texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=self.normalize_embeddings,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model encoding failed for {len(texts)} texts "
                f"using '{self.model_name}': {exc}"
            ) from exc

        # encode() with a list input always returns a 2-D array, but we
        # guard here for safety — a single-item list should still yield (1, D).
        if result.ndim == 1:
            result = result[np.newaxis, :]

        logger.debug(
            "Raw encode output — shape: %s | dtype: %s",
            result.shape,
            result.dtype,
        )

        return result

    @staticmethod
    def _pair_embeddings(
        chunks: List[Chunk],
        matrix: np.ndarray,
        embedding_dim: int,
    ) -> List[EmbeddedChunk]:
        """Zip source chunks with their embedding rows.

        Iterates in lock-step: ``chunks[i]`` maps to ``matrix[i]``.
        The ``ascontiguousarray`` call ensures each row is a proper
        C-contiguous 1-D array — some ANN libraries (notably FAISS)
        require contiguous memory and will segfault or raise cryptic errors
        on non-contiguous views.

        Args:
            chunks:        Original ordered list of Chunk objects.
            matrix:        2-D array, shape ``(len(chunks), embedding_dim)``.
            embedding_dim: Pre-computed column count for the metadata field.

        Returns:
            List of EmbeddedChunk, same length and order as *chunks*.
        """
        embedded: List[EmbeddedChunk] = []

        for chunk, row in zip(chunks, matrix):
            ec = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                text=chunk.text,
                # np.ascontiguousarray on a 2-D row view → 1-D contiguous copy
                embedding=np.ascontiguousarray(row, dtype=np.float32),
                embedding_dim=embedding_dim,
                model_name="",      # back-filled below — avoids a closure
                metadata=chunk.metadata,
            )
            embedded.append(ec)

        # Back-fill model_name once rather than capturing `self` in the loop.
        # This is a minor style choice: keeps _pair_embeddings a pure static
        # function that can be unit-tested without an EmbeddingGenerator instance.
        # Caller sets model_name after this returns — see generate().
        return embedded

    @staticmethod
    def _validate_config(batch_size: int) -> None:
        """Raise early for invalid construction parameters.

        Args:
            batch_size: Proposed batch size.

        Raises:
            TypeError:  If batch_size is not an int.
            ValueError: If batch_size is not positive.
        """
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be an int; got {type(batch_size).__name__}."
            )
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer; got {batch_size}."
            )

    @staticmethod
    def _validate_chunks(chunks: List[Chunk]) -> None:
        """Raise early for invalid or empty chunk lists.

        Args:
            chunks: Proposed input to :meth:`generate`.

        Raises:
            TypeError:  If *chunks* is not a list, or contains non-Chunk items.
            ValueError: If *chunks* is empty.
        """
        if not isinstance(chunks, list):
            raise TypeError(
                f"chunks must be a list; got {type(chunks).__name__}."
            )
        if len(chunks) == 0:
            raise ValueError(
                "chunks must not be empty; nothing to embed."
            )
        non_chunks = [i for i, c in enumerate(chunks) if not isinstance(c, Chunk)]
        if non_chunks:
            bad_type = type(chunks[non_chunks[0]]).__name__
            raise TypeError(
                f"All items in chunks must be Chunk instances; "
                f"found {bad_type!r} at index {non_chunks[0]}."
            )
