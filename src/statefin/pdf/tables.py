"""Level 2 document processing: table detection with PyMuPDF.

This module only *locates* tables and records their cells (text + bounding box) together
with the surrounding context a human would use to read them: the caption, the declared
unit, the "Source:" note, and footnotes. Deciding which column is which year is left to
extraction/structure.py, so the detection engine can be swapped or supplemented later
(pdfplumber, Camelot, an OCR layout model) without touching interpretation.

Landscape pages: CAG appendices are often stored as landscape content with /Rotate 270.
PyMuPDF's table finder works in the rotated frame and returns such tables transposed, so
rotation is reset to 0 *in memory* before detection. The PDF on disk is never modified.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from statefin.config import Config
from statefin.extraction.parsers import find_financial_years, parse_number
from statefin.models import Cell, ExtractedTable
from statefin.pdf.wordgrid import extend_ruled_header, find_wordgrid_tables

ENGINE = "pymupdf.find_tables"

_FOOTNOTE_BLOCK = re.compile(r"^\s*(\d{1,2}|[*#@$†‡^]+)\s*(?:\n|\s)\s*(\S.*)", re.S)


def _lines(page: "fitz.Page") -> List[Tuple[float, float, float, float, str]]:
    """Text lines as (x0, y0, x1, y1, text), top to bottom."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                x0, y0, x1, y1 = line["bbox"]
                out.append((x0, y0, x1, y1, text))
    out.sort(key=lambda l: (round(l[1], 1), l[0]))
    return out


def _cells(table) -> List[Cell]:
    data = table.extract()
    cells: List[Cell] = []
    header = table.header
    if getattr(header, "external", False):
        for c, (bbox, name) in enumerate(zip(header.cells, header.names)):
            if bbox and name and str(name).strip():
                cells.append(Cell(-1, c, str(name).strip(), [round(v, 2) for v in bbox]))
    for r, row in enumerate(table.rows):
        for c, bbox in enumerate(row.cells):
            if bbox is None:
                continue
            text = data[r][c] if r < len(data) and c < len(data[r]) else None
            if text is None or not str(text).strip():
                continue
            cells.append(Cell(r, c, str(text).strip(), [round(v, 2) for v in bbox]))
    return cells


def _is_chart_dump(cell: Cell) -> bool:
    """Charts drawn with vector lines are sometimes detected as one giant cell of numbers."""
    lines = cell.text.count("\n") + 1
    numbers = sum(1 for tok in cell.text.split() if parse_number(tok).ok)
    return lines > 6 and numbers > 12


def _fy_signature(cells: Sequence[Cell], max_row: int = 4) -> Tuple[str, ...]:
    rows = sorted({c.row for c in cells})[:max_row]
    return tuple(sorted({fy for c in cells if c.row in rows for fy, _, _ in find_financial_years(c.text)}))


def _is_numbering(row_cells: Sequence[Cell]) -> bool:
    nums = [parse_number(c.text) for c in row_cells]
    if len(nums) < 3 or not all(n.ok and n.value is not None and float(n.value).is_integer() for n in nums):
        return False
    ints = sorted(int(n.value) for n in nums)
    return ints[0] <= 2 and ints == list(range(ints[0], ints[0] + len(ints)))


def _header_cells(cells: Sequence[Cell]) -> List[Cell]:
    """Cells of the header band: up to a column-number row, else before the first numeric row."""
    by_row: Dict[int, List[Cell]] = {}
    for c in cells:
        by_row.setdefault(c.row, []).append(c)
    rows = sorted(by_row)
    for i, r in enumerate(rows[:10]):
        if _is_numbering(by_row[r]):
            return [c for rr in rows[: i + 1] for c in by_row[rr]]
    for i, r in enumerate(rows):
        if any(parse_number(c.text).ok for c in by_row[r]):
            return [c for rr in rows[:i] for c in by_row[rr]]
    return []


def _stacked_body(cells: Sequence[Cell], header: Sequence[Cell]) -> bool:
    """Columns ruled but rows not: the body comes back as tall cells of newline-joined values."""
    hdr = {id(c) for c in header}
    for c in cells:
        if id(c) in hdr:
            continue
        lines = [ln for ln in c.text.split("\n") if ln.strip()]
        if len(lines) >= 3 and sum(1 for ln in lines if parse_number(ln).ok) >= 2:
            return True
    return False


def _header_only(cells: Sequence[Cell]) -> bool:
    """A ruled table holding only a header (years, maybe a column-number row), no data."""
    if not any(find_financial_years(c.text) for c in cells):
        return False
    by_row: Dict[int, List[Cell]] = {}
    for c in cells:
        by_row.setdefault(c.row, []).append(c)
    for row_cells in by_row.values():
        nums = [parse_number(c.text) for c in row_cells]
        ok = [n for n in nums if n.ok]
        if not ok:
            continue
        ints = sorted(int(n.value) for n in ok if n.value is not None and float(n.value).is_integer())
        if len(ints) == len(row_cells) and ints == list(range(ints[0], ints[0] + len(ints))):
            continue                                    # column numbers "1 2 3 ..."
        return False
    return True


def _find_tables(page: "fitz.Page", cfg: Config) -> List[Tuple[List[float], List[Cell], str, int, int]]:
    """All tables on a page as (bbox, cells, strategy, n_rows, n_cols).

    Ruled tables come from PyMuPDF; then the word-geometry engine reads every year header
    that is not inside one of them (scanned pages, unruled tables)."""
    primary = cfg.get("tables.primary_strategy", "lines")
    fallback = cfg.get("tables.fallback_strategy", "wordgrid")
    out = []
    for t in page.find_tables(strategy=primary).tables:
        cells = [c for c in _cells(t) if not _is_chart_dump(c)]
        bbox, strategy = list(t.bbox), primary
        header = _header_cells(cells)
        if header and (_header_only(cells) or _stacked_body(cells, header)):
            hb = [min(c.bbox[0] for c in header), min(c.bbox[1] for c in header),
                  max(c.bbox[2] for c in header), max(c.bbox[3] for c in header)]
            extended = extend_ruled_header(page, hb, header)
            if extended:
                (bbox, cells), strategy = extended, "lines+wordgrid-body"
        out.append((bbox, cells, strategy, t.row_count, t.col_count))
    if fallback == "wordgrid":
        occupied = [tuple(b) for b, *_ in out]
        for bbox, cells in find_wordgrid_tables(page, cfg.all_qualifier_patterns, occupied):
            out.append((bbox, cells, "wordgrid", len({c.row for c in cells}), len({c.col for c in cells})))
    elif fallback and not out and len(find_financial_years(page.get_text())) >= 3:
        for t in page.find_tables(strategy=fallback).tables:
            out.append((list(t.bbox), _cells(t), fallback, t.row_count, t.col_count))
    return out


def _attach_context(table: ExtractedTable, lines, page_height: float, tables_above_y: float,
                    prev: Optional[ExtractedTable], cfg: Config) -> None:
    x0, y0, x1, y1 = table.bbox
    dist = cfg.get("tables.caption_search_distance_pt", 260)

    # Lines above this table (and below any earlier table on the same page), closest first.
    above = [l for l in lines if l[3] <= y0 + 2 and l[1] >= max(tables_above_y, y0 - dist)]
    above.sort(key=lambda l: -l[3])
    caption_idx = None
    for i, l in enumerate(above):
        if any(p.search(l[4]) for p in cfg.caption_patterns):
            caption_idx = i
            break
    if caption_idx is not None:
        caption = above[caption_idx][4]
        # "Appendix – 1.1" alone: append the first descriptive line below it.
        if len(re.sub(r"[\W\d_]+", "", caption)) < 12:
            for l in reversed(above[:caption_idx]):
                if not l[4].startswith("(") and not cfg.detect_unit(l[4]):
                    caption = f"{caption}: {l[4]}"
                    break
        table.caption = re.sub(r"\s+", " ", caption).strip()
        search = above[: caption_idx + 1]
    else:
        search = above[:4]
        # Finance Accounts print the statement title inside the table's first rows
        for c in sorted(table.cells, key=lambda c: (c.row, c.bbox[0])):
            if c.row > min(x.row for x in table.cells) + 1:
                break
            first_line = c.text.strip().split("\n")[0]
            if any(p.search(first_line) for p in cfg.caption_patterns):
                table.caption = re.sub(r"\s+", " ", first_line).strip()
                break

    # Unit: header cells first (a unit line can sit below the column numbers), then
    # context lines between caption and table.
    header_rows = sorted({c.row for c in table.cells})[:8]
    for c in table.cells:
        if c.row in header_rows:
            u = cfg.detect_unit(c.text)
            if u and u.kind == "currency":
                table.unit, table.unit_text, table.unit_source = u.unit, u.matched_text, "header"
                break
    if table.unit is None:
        for l in search:
            u = cfg.detect_unit(l[4])
            if u:
                table.unit, table.unit_text, table.unit_source = u.unit, u.matched_text, "context_above"
                break

    # Continuation of the previous page's table (repeated header, no caption).
    if (table.caption is None and prev is not None and prev.page_number == table.page_number - 1
            and table.index_on_page == 0 and y0 < page_height * 0.3
            and _fy_signature(table.cells) and _fy_signature(table.cells) == _fy_signature(prev.cells)):
        base = prev.caption or f"table on page {prev.page_number}"
        table.caption = base if base.endswith("(continued)") else f"{base} (continued)"
        table.caption_inherited_from = prev.table_id
        if table.unit is None and prev.unit:
            table.unit, table.unit_text, table.unit_source = prev.unit, prev.unit_text, "inherited"

    # "Source: Finance Accounts" directly below the table.
    below_dist = cfg.get("tables.source_note_search_distance_pt", 90)
    for l in lines:
        if y1 - 2 <= l[1] <= y1 + below_dist and any(p.search(l[4]) for p in cfg.cited_source_patterns):
            table.cited_source = re.sub(r"^\s*sources?\s*[:\-–]\s*", "", l[4], flags=re.I).strip()
            break


def _footnotes(page: "fitz.Page", below_y: float, table_rects: Sequence["fitz.Rect"]) -> Dict[str, str]:
    """Footnote text by marker ("4", "#", "*") from text blocks below the table."""
    notes: Dict[str, str] = {}
    for b in page.get_text("blocks"):
        if b[6] != 0 or b[1] < below_y:
            continue
        if any(fitz.Rect(b[:4]).intersects(r) for r in table_rects):
            continue
        m = _FOOTNOTE_BLOCK.match(b[4])
        if m and "p a g e" not in b[4].lower():
            notes.setdefault(m.group(1), re.sub(r"\s+", " ", m.group(2)).strip())
    return notes


def extract_tables(pdf_path: Path, document_id: str, page_numbers: Sequence[int], cfg: Config,
                   page_labels: Optional[Dict[int, Optional[str]]] = None,
                   on_page: Optional[Callable[[int, int], None]] = None) -> List[ExtractedTable]:
    """Detect tables on the given (1-based) pages."""
    page_labels = page_labels or {}
    tables: List[ExtractedTable] = []
    prev: Optional[ExtractedTable] = None
    pages = sorted(set(page_numbers))
    with fitz.open(pdf_path) as doc:
        for i, pn in enumerate(pages, start=1):
            if on_page:
                on_page(i, len(pages))
            page = doc[pn - 1]
            rotation = page.rotation
            if rotation:
                page.set_rotation(0)
            found = _find_tables(page, cfg)
            if not found:
                continue
            lines = _lines(page)
            found.sort(key=lambda t: (t[0][1], t[0][0]))
            table_rects = [fitz.Rect(t[0]) for t in found]
            tables_above_y = 0.0
            index = 0
            for bbox, cells, strategy, n_rows, n_cols in found:
                if len(cells) < 4:
                    continue
                table = ExtractedTable(
                    table_id=f"p{pn:03d}_t{index:02d}",
                    document_id=document_id,
                    page_number=pn,
                    page_label=page_labels.get(pn),
                    index_on_page=index,
                    bbox=[round(v, 2) for v in bbox],
                    engine=ENGINE if strategy != "wordgrid" else "statefin.wordgrid",
                    strategy=strategy,
                    page_rotation=rotation,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    cells=cells,
                )
                _attach_context(table, lines, page.rect.height, tables_above_y, prev, cfg)
                table.footnotes = _footnotes(page, bbox[3], table_rects)
                tables.append(table)
                prev = table
                tables_above_y = bbox[3]
                index += 1
    return tables


def render_snapshots(pdf_path: Path, tables: Sequence[ExtractedTable], out_dir: Path,
                     zoom: float = 1.6) -> Dict[str, Path]:
    """PNG of each table region (plus its caption) so reviewers can check values by eye."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    with fitz.open(pdf_path) as doc:
        for t in tables:
            page = doc[t.page_number - 1]
            if page.rotation:
                page.set_rotation(0)
            clip = fitz.Rect(t.bbox[0] - 12, t.bbox[1] - 60, t.bbox[2] + 12, t.bbox[3] + 30) & page.rect
            path = out_dir / f"{t.table_id}.png"
            page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip).save(str(path))
            paths[t.table_id] = path
    return paths
