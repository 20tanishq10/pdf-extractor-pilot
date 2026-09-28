"""Word-geometry table reader for pages without ruled lines (scanned + OCR'd reports, and
unruled tables in born-digital ones).

PyMuPDF's line-based table finder needs vector rulings; a scanned page has none. This
engine rebuilds tables from word positions instead:

  1. a header line is a visual line with >= 2 financial-year tokens; their x-centres
     define the year columns;
  2. words below it are grouped into visual lines; a line with numbers in the column
     zones is a data row, a line with only parenthesised small numbers is a
     percentage-share annotation (dropped: "(49)" is not -49), anything else is label text;
  3. label text that wraps above/below its numbers (vertically centred cells) is attached
     to the nearest row that has no label of its own, else continues the row above, else
     prefixes the row below when it starts with a serial ("4. Total revenue and Non-").

Output uses the same Cell structure as the ruled-table engine, so interpretation
(extraction/structure.py), mapping, checks and provenance are shared. OCR artefacts from
table borders ("|", "]") are stripped; everything else is left to the number parser, which
refuses rather than guesses.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from statefin.extraction.parsers import dominant_financial_year, is_serial_cell, parse_number
from statefin.models import Cell

_BORDER_JUNK = re.compile(r"^[|\[\]{}“”\"'’‘`©®°•·,.:;~_=\-—–]+$")
_TRAILING_JUNK = re.compile(r"[|\[\]{};:,]+$")
_LEADING_JUNK = re.compile(r"^[|\[\]{}]+")
_SHARE = re.compile(r"^\(\s*\d{1,3}(?:\.\d+)?\s*\)$")
_ENUM = re.compile(r"^\(?\s*(?:\d{1,3}[a-z]?|[ivxlc]{1,5}|[a-z])\s*[.)]", re.I)
_CAPTION = re.compile(r"^\s*(?:table|appendix|annexure|chart|statement)\s*[-–—:]?\s*[\dA-Z]", re.I)


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass
class Line:
    words: List[Word]
    kind: str = ""                              # header | data | share | label | noise
    label: List[Word] = field(default_factory=list)
    values: List[Tuple[int, Word]] = field(default_factory=list)      # (column, word)
    extra_label: List["Line"] = field(default_factory=list)

    @property
    def yc(self) -> float:
        return statistics.median(w.yc for w in self.words)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)


def _clean(words) -> List[Word]:
    heights = sorted(w[3] - w[1] for w in words if w[3] - w[1] > 1)
    min_h = 0.5 * heights[len(heights) // 2] if heights else 0      # half the median text height
    out = []
    for w in words:
        text = _LEADING_JUNK.sub("", _TRAILING_JUNK.sub("", w[4].strip()))
        if text and not _BORDER_JUNK.match(text) and (w[3] - w[1]) >= min_h:
            out.append(Word(w[0], w[1], w[2], w[3], text))
    return out


def _merge_split_years(words: List[Word]) -> List[Word]:
    """'2019-' + '20' on the same line -> '2019-20'."""
    out: List[Word] = []
    for w in sorted(words, key=lambda w: (round(w.yc), w.x0)):
        prev = out[-1] if out else None
        if prev and re.fullmatch(r"(19|20)\d{2}\s*[-–]", prev.text) and re.fullmatch(r"\d{2,4}", w.text) \
                and abs(prev.yc - w.yc) < 3 and w.x0 - prev.x1 < 8:
            out[-1] = Word(prev.x0, min(prev.y0, w.y0), w.x1, max(prev.y1, w.y1), prev.text + w.text)
        else:
            out.append(w)
    return out


def _group_lines(words: List[Word]) -> List[Line]:
    if not words:
        return []
    heights = [w.h for w in words if w.h > 2]
    tol = max(2.5, 0.35 * (statistics.median(heights) if heights else 9))
    lines: List[Line] = []
    for w in sorted(words, key=lambda w: w.yc):
        if lines and abs(w.yc - lines[-1].yc) <= tol:
            lines[-1].words.append(w)
        else:
            lines.append(Line([w]))
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    return lines


def _fy(w: Word, qualifiers) -> Optional[str]:
    return dominant_financial_year(w.text, qualifiers)


def _columns(header: Line, qualifiers) -> Optional[List[Tuple[float, float, float]]]:
    """(left, centre, right) of each year column, from the header's year tokens."""
    years = sorted((w for w in header.words if _fy(w, qualifiers)), key=lambda w: w.xc)
    if len(years) < 2:
        return None
    centres = [w.xc for w in years]
    gaps = [b - a for a, b in zip(centres, centres[1:])]
    if min(gaps) < 12:
        return None
    half = min(gaps) / 2
    cols = []
    for i, c in enumerate(centres):
        left = (centres[i - 1] + c) / 2 if i else c - half
        right = (c + centres[i + 1]) / 2 if i + 1 < len(centres) else c + half + 6
        cols.append((left, c, right))
    return cols


def _is_label_text(words: Sequence[Word]) -> bool:
    return any(sum(ch.isalpha() for ch in w.text) >= 2 for w in words)


def _column_of(w: Word, cols) -> Optional[int]:
    # numbers are usually right-aligned under a centred header: test the right edge too
    for i, (left, _c, right) in enumerate(cols):
        if left <= w.xc <= right or left <= w.x1 - 2 <= right:
            return i
    return None


def _classify(lines: List[Line], cols, label_limit: float) -> None:
    for ln in lines:
        ln.label, ln.values, shares, junk_numbers = [], [], [], []
        for w in ln.words:
            if w.x1 <= label_limit or w.xc < label_limit:
                ln.label.append(w)
                continue
            col = _column_of(w, cols)
            parsed = parse_number(w.text)
            if col is not None and _SHARE.match(w.text) and parsed.ok and abs(parsed.value) <= 100:
                shares.append((col, w))
            elif col is not None and parsed.ok:
                ln.values.append((col, w))
            elif parsed.ok:
                junk_numbers.append(w)
            elif sum(ch.isalpha() for ch in w.text) >= 3:
                ln.label.append(w)          # text inside the value zone (wide labels)
        # a lone footnote digit at the end of a label is a marker, not a value
        if ln.label and re.fullmatch(r"\d{1,2}|[!?’'*#@®°]", ln.label[-1].text) and len(ln.label) > 1:
            marker = ln.label.pop()
            last = ln.label[-1]
            ln.label[-1] = Word(last.x0, last.y0, marker.x1, last.y1, last.text + marker.text)
        # keep one value per column: prefer a plain number over a share in the same column
        by_col = {}
        for col, w in ln.values:
            by_col.setdefault(col, []).append(w)
        ln.values = [(col, ws[0]) for col, ws in sorted(by_col.items()) if len(ws) == 1]
        ambiguous = [col for col, ws in by_col.items() if len(ws) > 1]
        if ambiguous:
            ln.values = []                  # two numbers under one year: refuse the whole line
            ln.kind = "noise"
            continue
        label_ok = _is_label_text(ln.label)
        if ln.values and (len(ln.values) >= 2 or label_ok):
            ln.kind = "data"
        elif shares and not ln.values:
            ln.kind = "share" if not label_ok else "label"
        elif label_ok:
            ln.kind = "label"
        else:
            ln.kind = "noise"


# Label lines that head a block of rows rather than continue one
_HEADING = re.compile(r"^(?:as\s+(?:a\s+)?(?:percentage|per\s*cent|share|proportion)\s+of|components?\s+of|of\s+which"
                      r"|part\s*[-:]?\s*[a-z0-9]\b|memo|details\s+of)\b|\(\s*(?:in\s+)?(?:per\s*cent|%)\s*\)\s*$"
                      # Finance Accounts heads: "2071 Pensions and Other Retirement Benefits-", "01 Civil-",
                      # "(e) Pensions and Miscellaneous General Services-", "A. General Services - contd."
                      r"|^(?:\d{2,4}|\([a-z]{1,3}\)|[A-Z]\.|\([ivx]+\))\s+\S.*[A-Za-z)]\s*-\s*$"
                      r"|^(?:\d{4}|\d{2})\s+[A-Z][A-Za-z]"      # major (4-digit) / sub-major (2-digit) head headings
                      r"|\b(?:contd|concld)\.?\s*$", re.I)
_CONNECTOR_END = re.compile(r"(?:\b(?:and|of|the|to|in|for|on|by|from|with|under|including|excluding)|[&,\-(/])\s*$", re.I)


def _text(ln: Line) -> str:
    return " ".join(w.text for w in ln.label)


def _unbalanced_open(text: str) -> bool:
    return text.count("(") > text.count(")")


def _looks_continuation(text: str) -> bool:
    """'taxes and duties', '& Means Advances', 'Overdrafts)' continue the line above."""
    first = text.lstrip()[:1]
    return bool(first) and (first.islower() or first in "&)" or (text.count(")") > text.count("(")))


def _full_label(row: Line) -> str:
    parts = sorted(([row] if row.label else []) + row.extra_label, key=lambda x: x.yc)
    return " ".join(_text(p) for p in parts)


def _attach_labels(lines: List[Line], line_h: float) -> List[Line]:
    """Give every data row its full label (wrapped lines above/below); return rows in order.

    Consecutive label-only lines form a block. A block goes to the nearest data row that
    has no label of its own; otherwise its leading continuation lines extend the row
    above (lowercase / '&' / closing bracket, or the row above ends in a connector or an
    open bracket) and the rest prefixes the row below (serial numbers, open brackets,
    or a row below whose own label is a continuation)."""
    data = [ln for ln in lines if ln.kind == "data"]
    unlabeled = {id(ln) for ln in data if not _is_label_text(ln.label)}
    window = 1.7 * line_h

    # blocks of consecutive label lines
    blocks: List[List[Line]] = []
    for ln in lines:
        if ln.kind == "label":
            if blocks and blocks[-1] and lines.index(blocks[-1][-1]) == lines.index(ln) - 1 \
                    and ln.yc - blocks[-1][-1].yc <= window:
                blocks[-1].append(ln)
            else:
                blocks.append([ln])

    for block in blocks:
        # "Part D: Other data", "As a percentage of RR", "PART B" are headings, even inside a block
        while block and (_HEADING.search(_text(block[0]))
                         or (_text(block[0]).isupper() and len(_text(block[0])) > 3)):
            block[0].kind = "section"
            block = block[1:]
        if not block:
            continue
        top, bottom = block[0].yc, block[-1].yc
        above = [d for d in data if d.yc < top and top - d.yc <= window + 2]
        below = [d for d in data if d.yc > bottom and d.yc - bottom <= window + 2]
        inside = [d for d in data if top < d.yc < bottom]
        free = [d for d in above + below + inside if id(d) in unlabeled]
        up = max(above, key=lambda d: d.yc) if above else None
        down = min(below, key=lambda d: d.yc) if below else None
        far_above = [d for d in data if d.yc < top and top - d.yc <= 3 * window]
        up_far = max(far_above, key=lambda d: d.yc) if far_above else None
        if free and len(block) <= 3:
            for ln in block:
                if up is not None and id(up) not in unlabeled and ln.yc > up.yc and (
                        _looks_continuation(_text(ln)) or _CONNECTOR_END.search(_full_label(up))):
                    up.extra_label.append(ln)       # "taxes and duties" finishes the row above
                else:
                    min(free, key=lambda d: abs(d.yc - ln.yc)).extra_label.append(ln)
            continue
        up_text = _full_label(up) if up else ""
        i = 0
        # leading continuation lines extend the row above
        while i < len(block) and up is not None and (
                _looks_continuation(_text(block[i])) or _CONNECTOR_END.search(up_text) or _unbalanced_open(up_text)):
            up.extra_label.append(block[i])
            up_text = _full_label(up)
            i += 1
        rest = block[i:]
        if not rest:
            continue
        rest_text = " ".join(_text(ln) for ln in rest)
        down_text = _text(down) if down else ""
        if down is not None and (_ENUM.match(rest_text) or _unbalanced_open(rest_text) or _CONNECTOR_END.search(rest_text)
                                 or _looks_continuation(down_text) or not _is_label_text(down.label)):
            down.extra_label.extend(rest)
        elif up_far is not None:
            up_far.extra_label.extend(rest)     # the row below starts its own label: finish the row above
        elif down is not None:
            down.extra_label.extend(rest)
        else:
            for ln in rest:
                ln.kind = "section"
    return [ln for ln in lines if ln.kind in ("data", "section")]


def find_wordgrid_tables(page, qualifiers, occupied: Sequence[Tuple[float, float, float, float]] = ()
                         ) -> List[Tuple[List[float], List[Cell]]]:
    """Tables on `page` built from word positions: [(bbox, cells)].

    `occupied`: bboxes of tables already found by another engine; year headers inside them
    are skipped so the same table is not read twice.
    """
    words = _merge_split_years(_clean(page.get_text("words")))
    lines = _group_lines(words)
    heights = [w.h for w in words if w.h > 2]
    line_h = statistics.median(heights) if heights else 9.0
    gaps = [b.yc - a.yc for a, b in zip(lines, lines[1:]) if b.yc - a.yc > 1]
    spacing = statistics.median(gaps) if gaps else 12.0

    headers = []
    for i, ln in enumerate(lines):
        if sum(1 for w in ln.words if _fy(w, qualifiers)) >= 2:
            x0, y0 = min(w.x0 for w in ln.words), ln.y0
            if any(ox0 - 2 <= x0 <= ox1 + 2 and oy0 - 2 <= y0 <= oy1 + 2 for ox0, oy0, ox1, oy1 in occupied):
                continue
            cols = _columns(ln, qualifiers)
            if cols:
                headers.append((i, cols))

    tables = []
    for n, (hi, cols) in enumerate(headers):
        end = headers[n + 1][0] if n + 1 < len(headers) else len(lines)
        region = []
        last_y = lines[hi].y1
        for ln in lines[hi + 1:end]:
            text = " ".join(w.text for w in ln.words)
            prose = sum(1 for w in ln.words if w.xc > cols[0][0] and sum(ch.isalpha() for ch in w.text) >= 3)
            if ln.yc - last_y > max(3.5 * spacing, 40) or (_CAPTION.match(text) and region) or (prose >= 3 and region):
                break
            region.append(ln)
            last_y = ln.y1
        if not region:
            continue
        label_limit = cols[0][0] - 4
        _classify(region, cols, label_limit)
        # header band: lines before the first data row (qualifiers such as "(Actuals)", "BE")
        first_data = next((i for i, ln in enumerate(region) if ln.kind == "data"), None)
        if first_data is None:
            continue
        # header band: lines before the first data row with text under the year columns
        # ("(Actuals) (BE) (Actuals)", "Percentage of Actuals to BE")
        pre = region[:first_data]
        is_header = {id(ln) for ln in pre if any(w.xc >= label_limit and not parse_number(w.text).ok for w in ln.words)}
        header_lines = [lines[hi]] + [ln for ln in pre if id(ln) in is_header]
        body = _attach_labels([ln for ln in region if id(ln) not in is_header], line_h)
        # drop trailing sections (footnotes, next caption) after the last data row
        while body and body[-1].kind == "section":
            body.pop()
        if sum(1 for ln in body if ln.kind == "data") < 1:
            continue

        cells: List[Cell] = []
        for r, hl in enumerate(header_lines):
            for w in hl.words:
                if w.xc >= label_limit or r == 0:
                    cells.append(Cell(r, len(cells), w.text, [w.x0, w.y0, w.x1, w.y1]))
        r0 = len(header_lines)
        for r, ln in enumerate(body, start=r0):
            own = [ln] if ln.label and (_is_label_text(ln.label) or not ln.extra_label) else []
            parts = sorted(own + ln.extra_label, key=lambda x: x.yc)
            label_words = [w for p in parts for w in p.label]
            if label_words:
                text = " ".join(_text(p) for p in parts)
                serial = None
                first = label_words[0].text
                if is_serial_cell(first) and len(label_words) > 1:
                    serial = first
                bbox = [min(w.x0 for w in label_words), min(w.y0 for w in label_words),
                        max(w.x1 for w in label_words), max(w.y1 for w in label_words)]
                if serial:
                    cells.append(Cell(r, 0, serial, [label_words[0].x0, label_words[0].y0,
                                                     label_words[0].x1, label_words[0].y1]))
                    text = text[len(serial):].strip()
                    bbox[0] = label_words[1].x0
                cells.append(Cell(r, 1, text, bbox))
            for col, w in ln.values:
                cells.append(Cell(r, col + 2, w.text, [w.x0, w.y0, w.x1, w.y1]))
        all_words = [w for hl in header_lines for w in hl.words] + \
                    [w for ln in body for p in [ln] + ln.extra_label for w in p.words]
        bbox = [min(w.x0 for w in all_words), min(w.y0 for w in all_words),
                max(w.x1 for w in all_words), max(w.y1 for w in all_words)]
        tables.append(([round(v, 2) for v in bbox], cells))
    return tables


def extend_ruled_header(page, header_bbox: Sequence[float], header_cells: List[Cell]
                        ) -> Optional[Tuple[List[float], List[Cell]]]:
    """A table whose header is ruled but whose body is not (Finance Accounts statements).

    The header row with the most cells (usually the column-number row "1 2 3 4 5 6")
    gives exact column edges; words below the header are read into rows with the same
    line/label logic as the word-geometry engine. Returns (bbox, header + body cells)."""
    by_row: dict = {}
    for c in header_cells:
        by_row.setdefault(c.row, []).append(c)
    leaf = max(by_row.values(), key=len)
    if len(leaf) < 3:
        return None
    leaf = sorted(leaf, key=lambda c: c.bbox[0])
    cols = [(c.bbox[0], (c.bbox[0] + c.bbox[2]) / 2, c.bbox[2]) for c in leaf[1:]]
    label_limit = leaf[1].bbox[0] - 2
    all_words = _clean(page.get_text("words"))
    words = [w for w in all_words if w.y0 >= header_bbox[3] - 1]
    # a unit printed inside the header's last ruled cell ("(₹ in lakh)" under the column numbers)
    unit_words = [w for w in all_words if header_bbox[1] <= w.yc <= header_bbox[3] + 12]
    unit_text = " ".join(w.text for w in sorted(unit_words, key=lambda w: (round(w.yc), w.x0)))
    unit_match = re.search(r"\(?\s*\S?\s*in\s*(?:lakhs?|crores?|thousands?)\s*\)?", unit_text, re.I)
    lines = _group_lines(words)
    if not lines:
        return None
    heights = [w.h for w in words if w.h > 2]
    line_h = statistics.median(heights) if heights else 9.0
    gaps = [b.yc - a.yc for a, b in zip(lines, lines[1:]) if b.yc - a.yc > 1]
    spacing = statistics.median(gaps) if gaps else 12.0
    region, last_y = [], header_bbox[3]
    for ln in lines:
        text = " ".join(w.text for w in ln.words)
        if region and (ln.yc - last_y > max(4 * spacing, 48) or _CAPTION.match(text)):
            break
        region.append(ln)
        last_y = ln.y1
    _classify(region, cols, label_limit)
    first_data = next((i for i, ln in enumerate(region) if ln.kind == "data"), None)
    if first_data is None:
        return None
    unit_line = re.compile(r"\bin\s*(?:lakhs?|crores?|thousands?)\b", re.I)
    pre = [ln for ln in region[:first_data] if ln.kind != "label" or not _is_label_text(ln.label)
           or unit_line.search(" ".join(w.text for w in ln.words))]
    body = _attach_labels([ln for ln in region if ln not in pre], line_h)
    while body and body[-1].kind == "section":
        body.pop()
    cells = list(header_cells)
    r0 = max(c.row for c in header_cells) + 1
    if unit_match and not any(re.search(r"in\s*(?:lakh|crore|thousand)", c.text, re.I) for c in cells):
        cells.append(Cell(r0, 0, unit_match.group(0).strip(), [header_bbox[0], header_bbox[3] - 12,
                                                              header_bbox[2], header_bbox[3]]))
        r0 += 1
    for ln in pre:                      # e.g. "(₹ in lakh)" printed under the column numbers
        cells.append(Cell(r0, 0, " ".join(w.text for w in ln.words),
                          [min(w.x0 for w in ln.words), ln.y0, max(w.x1 for w in ln.words), ln.y1]))
        r0 += 1
    for r, ln in enumerate(body, start=r0):
        own = [ln] if ln.label and (_is_label_text(ln.label) or not ln.extra_label) else []
        parts = sorted(own + ln.extra_label, key=lambda x: x.yc)
        label_words = [w for p in parts for w in p.label]
        if label_words:
            cells.append(Cell(r, 0, " ".join(_text(p) for p in parts),
                              [min(w.x0 for w in label_words), min(w.y0 for w in label_words),
                               max(w.x1 for w in label_words), max(w.y1 for w in label_words)]))
        for col, wd in ln.values:
            cells.append(Cell(r, col + 1, wd.text, [wd.x0, wd.y0, wd.x1, wd.y1]))
    bottom = max(c.bbox[3] for c in cells)
    return [header_bbox[0], header_bbox[1], header_bbox[2], round(bottom, 2)], cells
