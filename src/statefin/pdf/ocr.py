"""Level 3 document processing: OCR, only for pages that need it.

Phase 1 detects pages without a text layer (see pdf/text.py) and reports them; it does not
OCR anything. The intended Phase 2+ implementation is to run OCRmyPDF with
`--skip-text` on a *copy* of the document so that pages that already have text are left
untouched, then re-run text extraction for the flagged pages only.
"""
from __future__ import annotations

import shutil
from typing import List

from statefin.models import PageText


def ocr_available() -> bool:
    return shutil.which("ocrmypdf") is not None or shutil.which("tesseract") is not None


def pages_needing_ocr(pages: List[PageText]) -> List[int]:
    return [p.page_number for p in pages if p.needs_ocr]


def ocr_pages(*_args, **_kwargs):
    raise NotImplementedError(
        "OCR is not part of Phase 1. Pages without a text layer are listed as needs_ocr in "
        "pages.jsonl; install OCRmyPDF/Tesseract and implement this hook in a later phase."
    )
