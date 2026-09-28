"""Review images: the source table with the exact value cell highlighted, or the whole page.

Rendering happens on an in-memory copy of the cached PDF; nothing is ever saved back to the
document. Images are cached next to the document's other artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import fitz

HIGHLIGHT = (0.82, 0.23, 0.23)
FILL = (1.0, 0.85, 0.2)


def _open_page(pdf_path: Path, page_number: int):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    if page.rotation:
        page.set_rotation(0)    # bounding boxes were recorded in the unrotated frame
    return doc, page


def render_observation(pdf_path: Path, obs: Dict[str, Any], out_path: Path, whole_page: bool = False,
                       zoom: float = 1.8) -> Path:
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc, page = _open_page(pdf_path, int(obs["source_page"]))
    try:
        table = fitz.Rect(obs.get("table_bbox") or page.rect)
        cell: Optional[fitz.Rect] = fitz.Rect(obs["cell_bbox"]) if obs.get("cell_bbox") else None
        if cell is not None:
            page.draw_rect(cell + (-1.5, -1.5, 1.5, 1.5), color=HIGHLIGHT, fill=FILL, fill_opacity=0.28,
                           width=1.4, overlay=True)
        if whole_page:
            page.draw_rect(table + (-3, -3, 3, 3), color=HIGHLIGHT, width=0.8, overlay=True)
            clip = page.rect
            zoom = min(zoom, 1.3)
        else:
            clip = (table + (-12, -46, 12, 24)) & page.rect
        page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip).save(str(out_path))
    finally:
        doc.close()
    return out_path
