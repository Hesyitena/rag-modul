"""Ekstraksi teks PDF dan pemecahan jadi chunk.

Chunk dibuat per halaman supaya setiap jawaban bisa menyebut nomor halaman sumbernya.
"""
import re
from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return [(nomor_halaman, teks), ...]. Halaman kosong (misal hasil scan) dilewati."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pecah teks jadi potongan ~size karakter dengan overlap.

    Batas potongan digeser mundur ke spasi/baris baru terdekat supaya kata tidak terpotong.
    """
    if overlap >= size:
        raise ValueError("overlap harus lebih kecil dari size")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks


def build_chunks(pdf_path: str | Path, doc_name: str) -> list[dict]:
    """Return list chunk siap disimpan: {"text", "source", "page"}."""
    out = []
    for page_no, text in extract_pages(pdf_path):
        for piece in chunk_text(text):
            out.append({"text": piece, "source": doc_name, "page": page_no})
    return out
