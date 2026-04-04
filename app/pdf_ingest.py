"""
PDF text extraction and chunking.

Considerations
--------------
- **Native text:** Per page we take the longer of pypdf (plain vs layout) and PyMuPDF `get_text`.
- **Scanned PDFs:** No OCR; image-only pages yield no text.
- **Chunk size:** Character windows with overlap; see module constants.
- **Metadata:** Page numbers on chunks for citations.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from pypdf import PdfReader

# Allow short tail segments when using fine chunk_size (e.g. ~10-token windows).
_MIN_PIECE_CHARS = 12


@dataclass
class TextChunk:
    id: str
    text: str
    source_file: str
    page_start: int
    page_end: int


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _pypdf_page_text(reader: PdfReader, page_index: int) -> str:
    if page_index >= len(reader.pages):
        return ""
    page = reader.pages[page_index]
    plain = _clean_text(page.extract_text(extraction_mode="plain") or "")
    layout = _clean_text(page.extract_text(extraction_mode="layout") or "")
    return plain if len(plain) >= len(layout) else layout


def _best_native_for_page(reader: PdfReader, doc: object, page_index: int) -> str:
    pp = _pypdf_page_text(reader, page_index)
    page = doc.load_page(page_index)
    fz = _clean_text(page.get_text("text") or "")
    return fz if len(fz) >= len(pp) else pp


def extract_pages_pdf(file_path: str, source_name: str) -> list[tuple[int, str]]:
    """Extract text per page using the better of pypdf and PyMuPDF native layers."""
    try:
        import fitz
    except ImportError:
        return _legacy_extract_without_fitz(file_path)

    pages_out: list[tuple[int, str]] = []
    reader = PdfReader(file_path, strict=False)
    with fitz.open(file_path) as doc:
        for i in range(len(doc)):
            text = _best_native_for_page(reader, doc, i)
            if text:
                pages_out.append((i + 1, text))
    return pages_out


def _legacy_extract_without_fitz(file_path: str) -> list[tuple[int, str]]:
    reader = PdfReader(file_path, strict=False)
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        plain = _clean_text(page.extract_text(extraction_mode="plain") or "")
        layout = _clean_text(page.extract_text(extraction_mode="layout") or "")
        text = plain if len(plain) >= len(layout) else layout
        if text:
            out.append((i + 1, text))
    return out


def _chunk_single_page_text(
    page_num: int,
    text: str,
    source_file: str,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    chunks: list[TextChunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        piece = text[start:end].strip()
        if len(piece) >= _MIN_PIECE_CHARS:
            chunks.append(
                TextChunk(
                    id=str(uuid.uuid4()),
                    text=f"[Page {page_num}]\n{piece}",
                    source_file=source_file,
                    page_start=page_num,
                    page_end=page_num,
                )
            )
        if end >= n:
            break
        start = max(0, end - overlap)
    if not chunks and text.strip():
        chunks.append(
            TextChunk(
                id=str(uuid.uuid4()),
                text=f"[Page {page_num}]\n{text.strip()}",
                source_file=source_file,
                page_start=page_num,
                page_end=page_num,
            )
        )
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    source_file: str,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    """Chunk per page so citations map to concrete PDF pages."""
    out: list[TextChunk] = []
    for page_num, text in pages:
        out.extend(_chunk_single_page_text(page_num, text, source_file, chunk_size, overlap))
    return out
