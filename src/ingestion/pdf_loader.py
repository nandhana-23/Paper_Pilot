"""
pdf_loader.py — PaperPilot Phase 1: PDF Ingestion Module

Responsibility: Load a PDF from disk and extract its text content,
page by page, into structured Python objects. Nothing more.

Design philosophy
-----------------
This module is intentionally narrow. It does one thing — turn a file
path into a list of page-level text strings — and does it reliably.
Chunking, embedding, and retrieval are downstream concerns that must
not bleed into this layer. Keeping the boundary clean means each phase
can be tested, replaced, or scaled independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF — pip install pymupdf

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# We use __name__ so callers can silence or redirect this module's logs
# independently:  logging.getLogger("src.ingestion.pdf_loader").setLevel(...)
# Using the root logger would be invasive and is a production anti-pattern.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    """Holds the extracted text for a single PDF page.

    A plain dataclass (not a dict) is used so that:
    - Field access is explicit and IDE-friendly.
    - Type checkers can verify downstream consumers.
    - The representation is self-documenting in logs and debuggers.

    Attributes:
        page_number: 1-based page number (human-friendly; matches PDF viewers).
        text: Raw extracted text for this page. May be an empty string if the
              page contains no selectable text (e.g. a scanned image page).
    """

    page_number: int
    text: str


@dataclass
class PDFDocument:
    """Aggregates all extracted content from a single PDF file.

    Keeping metadata (path, page count) alongside content means callers
    never have to re-open the file to answer basic provenance questions.
    This is important for RAG pipelines where every chunk must carry
    source attribution.

    Attributes:
        source_path: Absolute, resolved path to the source file.
        page_count: Total number of pages in the document.
        pages: Ordered list of :class:`PageContent` objects, one per page.
    """

    source_path: Path
    page_count: int
    pages: List[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Return all page texts joined by a newline separator.

        Convenience accessor for callers that don't need per-page granularity.
        The separator is a single newline — double newlines would introduce
        artificial whitespace that could confuse a downstream tokeniser.
        """
        return "\n".join(page.text for page in self.pages)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class PDFLoader:
    """Loads a PDF from disk and extracts its text using PyMuPDF.

    Why PyMuPDF (fitz)?
    -------------------
    - Fastest pure-Python-callable PDF library for text extraction.
    - Handles a wide range of encodings and PDF versions robustly.
    - Exposes per-page, per-block, and per-word granularity — useful if
      Phase 2 needs finer-grained chunking without swapping libraries.
    - Actively maintained and well-documented.

    Usage::

        loader = PDFLoader()
        doc = loader.load("papers/attention_is_all_you_need.pdf")
        for page in doc.pages:
            print(page.page_number, page.text[:80])

    The class is instantiated rather than implemented as a free function so
    that future phases can inject configuration (e.g. extraction flags,
    OCR fallback settings) without changing the call site.
    """

    # ------------------------------------------------------------------
    # PyMuPDF text extraction flag
    # ------------------------------------------------------------------
    # TEXT_PRESERVE_WHITESPACE retains spatial whitespace, giving cleaner
    # paragraph boundaries than the default. This matters for research
    # papers where multi-column layouts can otherwise run together.
    # We define it at class level so subclasses or tests can override it.
    _EXTRACTION_FLAGS: int = fitz.TEXT_PRESERVE_WHITESPACE

    def load(self, path: str | Path) -> PDFDocument:
        """Load a PDF file and extract text from every page.

        Args:
            path: Filesystem path to the PDF. Accepts ``str`` or
                  :class:`pathlib.Path`. The path is resolved to an absolute
                  path before opening, so relative paths work correctly
                  regardless of the process working directory.

        Returns:
            A :class:`PDFDocument` containing per-page text and metadata.

        Raises:
            FileNotFoundError: If *path* does not exist or is not a file.
            ValueError: If *path* does not have a ``.pdf`` extension.
            RuntimeError: If PyMuPDF cannot open or parse the file.
        """
        resolved = self._validate_path(path)
        logger.info("Loading PDF: %s", resolved)

        try:
            # fitz.open returns a context-manager-compatible Document object.
            # Using `with` ensures the file handle is released even if
            # extraction raises an exception mid-document.
            with fitz.open(str(resolved)) as doc:
                page_count = len(doc)
                logger.debug("Opened PDF — pages: %d", page_count)

                pages = self._extract_pages(doc)

        except fitz.FileDataError as exc:
            # fitz raises FileDataError for corrupt or password-protected PDFs.
            # We re-raise as RuntimeError to keep callers decoupled from the
            # PyMuPDF exception hierarchy.
            raise RuntimeError(
                f"PyMuPDF could not parse the file: {resolved}"
            ) from exc

        document = PDFDocument(
            source_path=resolved,
            page_count=page_count,
            pages=pages,
        )

        logger.info(
            "Ingestion complete — path: %s | pages: %d | total chars: %d",
            resolved,
            document.page_count,
            len(document.full_text),
        )
        return document

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_path(self, path: str | Path) -> Path:
        """Resolve and validate the given path.

        Centralising validation here means :meth:`load` stays readable,
        and subclasses can override validation rules without touching the
        core extraction logic.

        Args:
            path: Raw path provided by the caller.

        Returns:
            Resolved absolute :class:`pathlib.Path`.

        Raises:
            FileNotFoundError: If the path does not point to an existing file.
            ValueError: If the file extension is not ``.pdf``.
        """
        resolved = Path(path).resolve()

        if not resolved.is_file():
            raise FileNotFoundError(f"PDF not found: {resolved}")

        # Case-insensitive extension check — ".PDF" is valid on some systems.
        if resolved.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a .pdf file, got: {resolved.suffix!r} ({resolved})"
            )

        return resolved

    def _extract_pages(self, doc: fitz.Document) -> List[PageContent]:
        """Iterate over every page and extract its text.

        Extraction is isolated here so it can be unit-tested with a mock
        Document, and so the per-page logic stays separate from file I/O.

        Empty pages (no selectable text — scanned images, blank pages) are
        included in the output with ``text=""`` rather than skipped. Skipping
        them would misalign ``page_number`` with the physical document, which
        would break any citation or source-attribution feature in Phase 2.

        Args:
            doc: An open :class:`fitz.Document`.

        Returns:
            List of :class:`PageContent`, one per page, in document order.
        """
        pages: List[PageContent] = []

        for zero_based_index, page in enumerate(doc):
            # PyMuPDF uses 0-based indexing; we expose 1-based to match
            # what a human sees in a PDF viewer.
            page_number = zero_based_index + 1

            try:
                text: str = page.get_text(flags=self._EXTRACTION_FLAGS)
            except Exception as exc:  # noqa: BLE001 — fitz can raise broadly
                # Log and continue rather than aborting the entire document.
                # A single unreadable page should not prevent extraction of
                # the rest. Downstream consumers can detect empty pages.
                logger.warning(
                    "Failed to extract text from page %d — skipping. Error: %s",
                    page_number,
                    exc,
                )
                text = ""

            if not text.strip():
                logger.debug("Page %d contains no selectable text.", page_number)

            pages.append(PageContent(page_number=page_number, text=text))

        return pages
