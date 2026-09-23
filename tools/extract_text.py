"""Part 2 -- Document Text Extraction.

Strategy:
  1. If it's a text-based PDF, pull text natively with PyMuPDF (fast, free,
     no LLM call).
  2. If that yields little/no text (scanned PDF) or the upload is an image,
     fall back to the Gemini VLM to read it directly.

This keeps the common case (digital PDFs) cheap and fast, while still
covering scanned documents without needing a separate OCR engine like
Tesseract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf  # PyMuPDF

from vision.gemini_client import ocr_document

MIN_NATIVE_TEXT_CHARS = 40  # below this, treat the PDF as "scanned" and use vision


@dataclass
class ExtractionResult:
    raw_text: str
    method: str  # "pdf_native" | "vlm"
    tokens_used: int = 0


def _mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "application/octet-stream")


def _extract_pdf_native(path: Path) -> Optional[str]:
    with pymupdf.open(path) as doc:
        pages = [page.get_text() for page in doc]
    text = "\n".join(pages).strip()
    return text if len(text) >= MIN_NATIVE_TEXT_CHARS else None


def extract_text(file_path: str) -> ExtractionResult:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    if path.suffix.lower() == ".pdf":
        native = _extract_pdf_native(path)
        if native is not None:
            return ExtractionResult(raw_text=native, method="pdf_native")

    # Scanned PDF or image -> Gemini vision
    file_bytes = path.read_bytes()
    result = ocr_document(file_bytes, _mime_for(path))
    return ExtractionResult(raw_text=result.text, method="vlm", tokens_used=result.tokens_used)
