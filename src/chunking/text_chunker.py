"""
text_chunker.py — PaperPilot Phase 2: Text Chunking Module

Responsibility: Accept a PDFDocument produced by Phase 1, split its text
into overlapping, fixed-size chunks, and return those chunks as a typed
list — each carrying enough metadata for a future retrieval layer to cite
its exact origin.

Design philosophy
-----------------
Chunking is the most consequential preprocessing decision in a RAG pipeline.
Chunk too large → the retriever returns irrelevant context that drowns the
answer. Chunk too small → a single idea is fragmented across multiple chunks
and the LLM never sees it whole.

This module deliberately makes chunk_size and overlap *configurable at
construction time* so they can be tuned against retrieval quality metrics
without touching the code. Sensible defaults are provided, but no default
is universally optimal.

The chunker operates at the *character* level rather than the token level.
Character-based splitting is library-free, deterministic, and portable across
tokenisers. The trade-off (character count ≠ token count) is acceptable at
this phase; a token-aware variant can be added in a future pass without
changing the public interface.

This module does NOT:
  - embed chunks
  - write to a vector database
  - call any LLM
  - re-open or re-parse the PDF
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Phase 1 types — imported directly to keep the pipeline's dependency graph
# explicit. If the ingestion package moves, this import line is the only
# thing that needs updating.
from src.ingestion.pdf_loader import PDFDocument, PageContent  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants — tunable defaults
# ---------------------------------------------------------------------------

# 500 characters ≈ 100–125 tokens with a standard BPE tokeniser.
# At this size a chunk typically holds one or two dense paragraphs —
# enough semantic context for retrieval without overwhelming the LLM.
DEFAULT_CHUNK_SIZE: int = 500

# 20 % overlap prevents a sentence that straddles a boundary from being
# invisible to either neighbouring chunk. Overlap trades index size for
# recall; set to 0 to disable.
DEFAULT_OVERLAP: int = 100


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """Provenance information attached to every chunk.

    Keeping metadata in its own dataclass rather than as flat fields on
    :class:`Chunk` means it can be serialised independently (e.g. to a
    metadata store alongside a vector index) and extended without
    breaking the Chunk interface.

    Attributes:
        source_path: Absolute path of the originating PDF file.
        page_number: 1-based page number where this chunk *begins*.
            For chunks that span a page boundary (possible when pages are
            joined before splitting — see design note in TextChunker) the
            value reflects the page that contributed the most characters.
        chunk_index: 0-based position of this chunk within the document.
            Useful for reconstructing reading order and for debugging.
        total_chunks: Total number of chunks produced from this document.
            Populated after all chunks are built; allows a consumer to
            express "chunk 3 of 47" without a second pass.
    """

    source_path: Path
    page_number: int
    chunk_index: int
    total_chunks: int = 0  # back-filled by TextChunker after the full pass


@dataclass
class Chunk:
    """A single unit of text ready for downstream embedding or retrieval.

    Fields
    ------
    chunk_id:
        A UUID4 string. Using a UUID rather than a sequential integer means
        chunks can be merged, deduplicated, or distributed across workers
        without collision. The ID is stable for the lifetime of the object
        but is NOT deterministic across runs — use ``metadata.chunk_index``
        if you need a reproducible identifier.
    page_number:
        Convenience alias for ``metadata.page_number``.  Promoted to the
        top level because retrieval code queries it constantly; requiring
        ``chunk.metadata.page_number`` everywhere would be verbose.
    text:
        The raw chunk text. No post-processing is applied; the embedder
        or LLM integration layer in a later phase owns that responsibility.
    metadata:
        Full provenance record. See :class:`ChunkMetadata`.
    """

    chunk_id: str
    page_number: int
    text: str
    metadata: ChunkMetadata = field(repr=False)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """Splits a :class:`~src.ingestion.pdf_loader.PDFDocument` into
    overlapping text chunks, preserving page-level provenance on each.

    Parameters
    ----------
    chunk_size:
        Target chunk length in *characters*. Each chunk will be at most
        this long; the final chunk of a document may be shorter.
        Must be a positive integer.
    overlap:
        Number of characters from the end of chunk *N* to repeat at the
        start of chunk *N+1*. Must be non-negative and strictly less than
        ``chunk_size`` (otherwise chunks would never advance).
    respect_paragraphs:
        When ``True`` (default), the chunker attempts to break at the
        nearest paragraph boundary *before* the hard ``chunk_size`` limit,
        preventing mid-paragraph splits. Falls back to a hard cut when no
        paragraph boundary is found within the window. Set to ``False`` for
        strictly uniform chunk sizes.

    Usage::

        from src.ingestion.pdf_loader import PDFLoader
        from src.chunking.text_chunker import TextChunker

        doc = PDFLoader().load("paper.pdf")
        chunks = TextChunker(chunk_size=500, overlap=100).chunk(doc)
        for c in chunks:
            print(c.chunk_id, c.page_number, c.text[:60])
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        respect_paragraphs: bool = True,
    ) -> None:
        self._validate_config(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_paragraphs = respect_paragraphs

        logger.debug(
            "TextChunker initialised — chunk_size: %d | overlap: %d | "
            "respect_paragraphs: %s",
            self.chunk_size,
            self.overlap,
            self.respect_paragraphs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, document: PDFDocument) -> List[Chunk]:
        """Chunk all pages of *document* and return an ordered list of
        :class:`Chunk` objects.

        The method operates in three stages:

        1. **Normalise** — strip each page's text, skip blank pages.
        2. **Split** — apply the sliding-window algorithm to the
           normalised text, respecting paragraph boundaries if configured.
        3. **Annotate** — assign UUIDs, page numbers, indices, and
           back-fill ``total_chunks`` on the metadata of every chunk.

        Args:
            document: A :class:`~src.ingestion.pdf_loader.PDFDocument`
                      returned by :class:`~src.ingestion.pdf_loader.PDFLoader`.

        Returns:
            Ordered list of :class:`Chunk` objects. Returns an empty list
            if the document contains no extractable text.

        Raises:
            TypeError: If *document* is not a :class:`PDFDocument`.
        """
        if not isinstance(document, PDFDocument):
            raise TypeError(
                f"Expected a PDFDocument, got {type(document).__name__!r}."
            )

        logger.info(
            "Chunking document — source: %s | pages: %d | "
            "chunk_size: %d | overlap: %d",
            document.source_path,
            document.page_count,
            self.chunk_size,
            self.overlap,
        )

        all_chunks: List[Chunk] = []

        for page in document.pages:
            page_chunks = self._chunk_page(page, document.source_path)
            all_chunks.extend(page_chunks)

        # --- Back-fill sequential index and total count -----------------
        # We can only know total_chunks after processing all pages, so we
        # do a second pass here rather than guessing during construction.
        total = len(all_chunks)
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = idx
            chunk.metadata.total_chunks = total

        logger.info(
            "Chunking complete — source: %s | chunks produced: %d",
            document.source_path.name,
            total,
        )

        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_page(
        self,
        page: PageContent,
        source_path: Path,
    ) -> List[Chunk]:
        """Produce all chunks for a single page.

        Pages are processed independently (rather than joining all pages
        into one long string) so that ``page_number`` is always accurate.
        The trade-off is that a concept which spans a page break will land
        in two adjacent chunks. For academic PDFs this is generally
        acceptable; page boundaries in papers usually coincide with natural
        section breaks.

        Args:
            page: A single page from the document.
            source_path: Path carried forward into each chunk's metadata.

        Returns:
            List of chunks for this page. Empty if the page has no text.
        """
        text = page.text.strip()

        if not text:
            logger.debug("Page %d is empty — skipping.", page.page_number)
            return []

        # Normalise internal whitespace: collapse runs of 3+ newlines to two
        # (preserving paragraph breaks) and replace hard tabs with spaces.
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.replace("\t", " ")

        windows = self._sliding_window(text)

        chunks: List[Chunk] = []
        for raw_text in windows:
            clean = raw_text.strip()
            if not clean:
                continue
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                page_number=page.page_number,
                text=clean,
                metadata=ChunkMetadata(
                    source_path=source_path,
                    page_number=page.page_number,
                    chunk_index=0,       # back-filled in chunk()
                    total_chunks=0,      # back-filled in chunk()
                ),
            )
            chunks.append(chunk)

        logger.debug(
            "Page %d → %d chunk(s) from %d characters.",
            page.page_number,
            len(chunks),
            len(text),
        )
        return chunks

    def _sliding_window(self, text: str) -> List[str]:
        """Apply the core sliding-window split to a normalised text string.

        Algorithm
        ---------
        Start at position 0. Advance by ``(chunk_size - overlap)`` each
        step. When ``respect_paragraphs`` is True, look backwards from the
        candidate end position for the last paragraph break (``\\n\\n``)
        within a tolerance of 20 % of ``chunk_size``; snap to that break if
        found. This softens hard cuts without allowing chunks to grow
        unboundedly.

        The loop terminates when ``start`` reaches or passes ``len(text)``.
        The final window may be shorter than ``chunk_size``.

        Args:
            text: Pre-normalised page text.

        Returns:
            List of text windows (may be empty strings for blank input).
        """
        if not text:
            return []

        windows: List[str] = []
        start = 0
        step = self.chunk_size - self.overlap  # guaranteed > 0 by validation

        while start < len(text):
            end = start + self.chunk_size

            if self.respect_paragraphs and end < len(text):
                end = self._snap_to_paragraph(text, start, end)

            windows.append(text[start:end])
            start += step

        return windows

    def _snap_to_paragraph(self, text: str, start: int, end: int) -> int:
        """Attempt to move *end* backward to the nearest paragraph break.

        We search within a tolerance window of 20 % of ``chunk_size``
        so that paragraph-snapping never shrinks a chunk by more than
        a fifth of its target size. If no break is found in that range
        the original *end* is returned unchanged.

        ``\\n\\n`` is used as the paragraph separator because it is the
        universal convention in plain text (CommonMark, reStructuredText,
        and most PDF text extraction outputs). Single ``\\n`` is too
        aggressive — it would break mid-list-item or mid-equation.

        Args:
            text: Full page text being split.
            start: Start position of the current window.
            end: Candidate end position (``start + chunk_size``).

        Returns:
            Adjusted end position, always in ``[end - tolerance, end]``.
        """
        tolerance = max(1, self.chunk_size // 5)
        search_start = max(start + 1, end - tolerance)

        # rfind searches right-to-left — we want the *last* break before
        # `end` so we keep as much text as possible in the chunk.
        break_pos = text.rfind("\n\n", search_start, end)

        if break_pos != -1:
            # +2 to include the separator itself in the current chunk so
            # the next chunk doesn't start with a leading blank line.
            return break_pos + 2

        return end

    @staticmethod
    def _validate_config(chunk_size: int, overlap: int) -> None:
        """Raise early with a clear message if the configuration is invalid.

        Static method because it doesn't need instance state and may be
        called before ``__init__`` finishes.

        Args:
            chunk_size: Proposed chunk size.
            overlap:    Proposed overlap.

        Raises:
            ValueError: For any invalid combination.
            TypeError:  If arguments are not integers.
        """
        if not isinstance(chunk_size, int) or not isinstance(overlap, int):
            raise TypeError(
                f"chunk_size and overlap must be integers; "
                f"got {type(chunk_size).__name__} and {type(overlap).__name__}."
            )
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer; got {chunk_size}."
            )
        if overlap < 0:
            raise ValueError(
                f"overlap must be non-negative; got {overlap}."
            )
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than "
                f"chunk_size ({chunk_size}); otherwise the sliding window "
                f"never advances and produces an infinite loop."
            )
