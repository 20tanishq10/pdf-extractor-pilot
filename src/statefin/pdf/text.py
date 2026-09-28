"""Level 1 document processing: page-by-page text with PyMuPDF.

No OCR happens here. Pages without a usable text layer are marked `needs_ocr` so that a
later OCR pass (pdf/ocr.py) can target only those pages.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import fitz  # PyMuPDF

from statefin.models import PageText

_PRINTED_PAGE_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(\d{1,4})\s*\|\s*P\s*a\s*g\s*e\b"),     # CAG style "24 | P a g e"
    re.compile(r"(?:^|\n)\s*Page\s+(\d{1,4})\b", re.I),
]


def _page_label(page: "fitz.Page", text: str) -> Optional[str]:
    try:
        label = page.get_label()
    except Exception:  # pragma: no cover - older PyMuPDF / malformed label trees
        label = ""
    if label:
        return label
    for pat in _PRINTED_PAGE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def extract_page_texts(pdf_path: Path, min_chars_for_text_layer: int = 50,
                       on_page: Optional[Callable[[int, int], None]] = None) -> List[PageText]:
    """Return the text of every page (1-based page numbers)."""
    pages: List[PageText] = []
    with fitz.open(pdf_path) as doc:
        total = doc.page_count
        for index, page in enumerate(doc):
            if on_page:
                on_page(index + 1, total)
            text = page.get_text("text")
            chars = len(text.strip())
            images = len(page.get_images(full=False))
            pages.append(PageText(
                page_number=index + 1,
                page_label=_page_label(page, text),
                text=text,
                char_count=chars,
                rotation=page.rotation,
                image_count=images,
                needs_ocr=chars < min_chars_for_text_layer and images > 0,
            ))
    return pages


def save_pages(pages: Iterable[PageText], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_pages(path: Path) -> List[PageText]:
    with open(path, encoding="utf-8") as fh:
        return [PageText.from_dict(json.loads(line)) for line in fh if line.strip()]


def normalize_for_search(text: str) -> str:
    """Lower-case, join hyphenated line breaks, collapse whitespace."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).lower()


def _pdf_date(value: Optional[str]) -> Optional[str]:
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", value or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def pdf_dates(pdf_path: Path):
    """(creation, modification) dates from PDF metadata as YYYY-MM-DD, where present."""
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata or {}
    return _pdf_date(meta.get("creationDate")), _pdf_date(meta.get("modDate"))
