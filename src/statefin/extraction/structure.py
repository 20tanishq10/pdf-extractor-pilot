"""Interpret a detected table: which cells are year columns and which are row labels.

Works only on cell text and geometry, independently of the detection engine, and handles:

* years in columns (the common case) and years in rows (transposed tables);
* multi-row headers, where the year and its qualifier sit in different rows
  ("2024-25" over "(BE)"), including one year spanning several qualifier sub-columns
  ("2020-21" over "Projection | Actual");
* merged header cells that shift column indices. Values are matched to year columns by
  horizontal position (bounding boxes), not by column index;
* non-level columns: "Percentage of Actuals to BE", or year sub-headers under a parent
  such as "Per cent share to total for the year" or "Change over 2005-06";
* serial-number cells, section rows ("Part-B: Expenditure"), labels split over two rows,
  and section-level units ("Part E: Fiscal Health Indicators (in per cent)").

Nothing here decides *what* a row means; that is extraction/mapper.py.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from statefin.config import Config
from statefin.extraction.parsers import (ParsedNumber, dominant_financial_year, is_serial_cell,
                                         parse_number)
from statefin.models import Anchor, Cell, ExtractedTable, TableModel, ValueCell

HEADER_SEARCH_ROWS = 8     # year headers are looked for in the first N rows
NARROW_CHILD = 0.8         # a qualifier cell narrower than this share of its year cell splits it
TOL = 1.5                  # points


def _cx(c: Cell) -> float:
    return (c.bbox[0] + c.bbox[2]) / 2


def _cy(c: Cell) -> float:
    return (c.bbox[1] + c.bbox[3]) / 2


def _width(c: Cell) -> float:
    return max(c.bbox[2] - c.bbox[0], 0.1)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class _Info:
    __slots__ = ("fy", "num", "serial")

    def __init__(self, fy: Optional[str], num: ParsedNumber, serial: bool):
        self.fy, self.num, self.serial = fy, num, serial


def _infer_composition_unit(model: TableModel) -> TableModel:
    """A table whose 'Total' row is 100 in every year column is a percentage composition."""
    totals = [vc for vc in model.value_cells
              if re.sub(r"[^a-z]", "", vc.raw_label.lower()) in ("total", "grandtotal") and vc.value is not None]
    if len(totals) >= 2 and all(99.5 <= vc.value <= 100.5 for vc in totals):
        model.inferred_unit = "percent"
        model.notes.append("'Total' row equals 100 in every column: treated as a percentage composition table")
    return model


def analyze_table(table: ExtractedTable, cfg: Config) -> TableModel:
    return _infer_composition_unit(_analyze(table, cfg))


def _analyze(table: ExtractedTable, cfg: Config, allow_split: bool = True) -> TableModel:
    cells = [c for c in table.cells if c.text.strip()]
    rows: "OrderedDict[int, List[Cell]]" = OrderedDict()
    for c in sorted(cells, key=lambda c: (c.row, c.bbox[0])):
        rows.setdefault(c.row, []).append(c)
    info = {id(c): _Info(dominant_financial_year(c.text, cfg.all_qualifier_patterns),
                         parse_number(c.text), is_serial_cell(c.text)) for c in cells}

    first_rows = list(rows)[:HEADER_SEARCH_ROWS]
    leftmost = {r: min(cs, key=lambda c: c.bbox[0]) for r, cs in rows.items()}
    table_left = min((c.bbox[0] for c in cells), default=0.0)
    # a year cell is a column header unless it is the row's leftmost cell at the table's left edge
    in_columns = [c for c in cells if info[id(c)].fy and c.row in first_rows
                  and (c is not leftmost[c.row] or c.bbox[0] > table_left + 60)]
    in_rows = [c for c in cells if info[id(c)].fy and c is leftmost[c.row]
               and c.bbox[0] <= table_left + 40]

    if len(in_columns) >= 2 and len(in_columns) >= len(in_rows):
        boundary = _split_side_by_side(table, cells, info) if allow_split else None
        if boundary is not None:
            year_cells = [c for c in cells if info[id(c)].fy and c.row in first_rows]
            left = sum(1 for c in year_cells if _cx(c) < boundary)
            if 2 <= left <= len(year_cells) - 2:          # each half keeps its own year columns
                return _analyze_halves(table, cells, boundary, cfg)
        return _years_in_columns(table, rows, info, cfg)
    if len(in_rows) >= 2:
        return _years_in_rows(table, rows, info, cfg)
    return TableModel(table.table_id, "no_year_axis", [], [],
                      ["no financial-year header found; table not interpreted"])


def _analyze_halves(table: ExtractedTable, cells: List[Cell], boundary: float, cfg: Config) -> TableModel:
    """'Receipts | Disbursements' statements: two tables printed side by side."""
    import dataclasses
    merged = TableModel(table.table_id, "years_in_columns", [], [], [f"side-by-side table split at x={boundary:.0f}"])
    for tag, part in (("L", [c for c in cells if _cx(c) < boundary]), ("R", [c for c in cells if _cx(c) >= boundary])):
        half = dataclasses.replace(table, cells=part)
        m = _analyze(half, cfg, allow_split=False)
        for a in m.anchors:
            a.anchor_id = tag + a.anchor_id
        for vc in m.value_cells:
            vc.anchor_id = f"{tag}{vc.anchor_id}" if vc.anchor_id else vc.anchor_id
            vc.row_index = vc.row_index * 2 + (1 if tag == "R" else 0)   # distinct rows per half
        merged.anchors += m.anchors
        merged.value_cells += m.value_cells
        merged.notes += [f"{tag}: {n}" for n in m.notes]
    return merged


# ------------------------------------------------------------------ years in columns

def _column_rejection(header_text_wo_year: str, value_types: List[str], cfg: Config) -> Optional[str]:
    for pat in cfg.column_exclude_patterns:
        if pat.search(header_text_wo_year):
            return f"non-level column (header matches '{pat.pattern}')"
    if len(value_types) > 1:
        return f"conflicting column qualifiers {value_types}"
    if value_types and value_types[0] not in cfg.accepted_value_types:
        return f"value type '{value_types[0]}' not accepted"
    return None


def _build_column_anchors(header_cells: List[Cell], info, cfg: Config,
                          projection_caption: bool = False) -> Tuple[List[Anchor], List[str]]:
    """Year columns from the header band.

    A year may be split into sub-columns printed in a lower header row: qualifiers
    ("Projection | Actual") or components ("Revenue | Capital | Total", "Charged | Voted |
    Total"). Each such child is assigned to the nearest year cell above it, which works
    whether the year sits in a merged cell or is just centred over its sub-columns.
    """
    notes: List[str] = []
    anchors: List[Anchor] = []
    fy_cells = [c for c in header_cells if info[id(c)].fy]
    others = [c for c in header_cells if not info[id(c)].fy]

    def child_kind(h: Cell) -> Optional[str]:
        if cfg.value_types(h.text):
            return "qualifier"
        if cfg.column_component(h.text):
            return "component"
        return None

    kids: Dict[int, List[Cell]] = {id(fc): [] for fc in fy_cells}
    for h in others:
        if not child_kind(h):
            continue
        above = [fc for fc in fy_cells if fc.row < h.row]
        if not above:
            continue
        nearest_row = max(fc.row for fc in above)
        row_fys = [fc for fc in above if fc.row == nearest_row]
        containing = [f for f in row_fys if f.bbox[0] - TOL <= _cx(h) <= f.bbox[2] + TOL]
        fc = min(containing or row_fys, key=lambda f: abs(_cx(f) - _cx(h)))
        # a same-width cell directly below the year is a label for the whole column, not a split
        kids[id(fc)].append(h)

    for fc in fy_cells:
        lo, hi = fc.bbox[0], fc.bbox[2]
        fy = info[id(fc)].fy
        children = sorted(kids[id(fc)], key=lambda h: h.bbox[0])
        split = len(children) >= 2
        child_ids = {id(h) for h in children}
        context = [h for h in others if id(h) not in child_ids
                   and _overlap(lo, hi, h.bbox[0], h.bbox[2]) >= 0.5 * min(_width(h), _width(fc))]
        context.sort(key=lambda h: (h.row, h.bbox[0]))
        parent_text = " ".join(_clean(h.text) for h in context if h.row < fc.row)
        below_text = " ".join(_clean(h.text) for h in context if h.row >= fc.row)
        fy_text_rest = _clean(fc.text)

        if split:                     # "2023-24" over "Revenue | Capital | Total"
            parts = [(child.bbox[0], child.bbox[2], _clean(child.text)) for child in children]
        else:                         # one qualifier under the year, e.g. "(BE)"
            parts = [(lo, hi, _clean(children[0].text) if children else "")]
        for sub_lo, sub_hi, child_text in parts:
            header_text = _clean(" ".join(t for t in (parent_text, fy_text_rest, below_text, child_text) if t))
            vtypes = cfg.value_types(" ".join((fy_text_rest, below_text, child_text, parent_text)))
            component = cfg.column_component(child_text) or cfg.column_component(
                re.sub(r"(?:19|20)\d{2}\s*[-–]\s*\d{2,4}", "", fy_text_rest))
            without_year = _clean(" ".join((parent_text, below_text, child_text)))
            reason = _column_rejection(without_year, vtypes, cfg)
            if reason is None and not vtypes and projection_caption:
                vtypes, reason = ["projection"], "caption describes a forecast/projection; column has no Actuals qualifier"
            anchors.append(Anchor(
                anchor_id=f"c{len(anchors)}",
                fy=fy,
                value_type=vtypes[0] if len(vtypes) == 1 else None,
                header_text=header_text,
                lo=round(sub_lo, 2), hi=round(sub_hi, 2),
                accepted=reason is None,
                skip_reason=reason,
                component=component,
            ))
            if reason:
                notes.append(f"skipped column '{header_text}': {reason}")
    return anchors, notes


def _is_numbering_row(cells: List[Cell], info) -> bool:
    """'1 2 3 4 5 6' under a header: column numbers, not data."""
    nums = [info[id(c)].num for c in cells]
    if len(nums) < 3 or not all(n.ok and n.value is not None and float(n.value).is_integer() for n in nums):
        return False
    vals = sorted(int(n.value) for n in nums)
    return vals[0] <= 2 and vals == list(range(vals[0], vals[0] + len(vals)))


def _split_side_by_side(table: ExtractedTable, cells: List[Cell], info) -> Optional[float]:
    """x boundary of a 'Receipts | Disbursements' table whose year sequence repeats, else None."""
    header = sorted([c for c in cells if info[id(c)].fy and c.row <= min(c.row for c in cells) + HEADER_SEARCH_ROWS],
                    key=lambda c: c.bbox[0])
    rows_with_fy: Dict[int, List[Cell]] = {}
    for c in header:
        rows_with_fy.setdefault(c.row, []).append(c)
    for row_cells in rows_with_fy.values():
        fys = [info[id(c)].fy for c in row_cells]
        n = len(fys)
        if n >= 4 and n % 2 == 0 and fys[: n // 2] == fys[n // 2:]:
            left_last, right_first = row_cells[n // 2 - 1], row_cells[n // 2]
            # labels of the right half must sit between the two year groups
            gap_text = [c for c in cells if c.bbox[0] >= left_last.bbox[2] - TOL and c.bbox[2] <= right_first.bbox[0] + TOL
                        and not info[id(c)].num.ok and len(c.text.strip()) > 3]
            if gap_text:
                return (left_last.bbox[2] + min(c.bbox[0] for c in gap_text)) / 2
    return None


def _nearest_anchor(x_center: float, anchors: Sequence[Anchor]) -> Optional[Anchor]:
    inside = [a for a in anchors if a.lo - TOL <= x_center <= a.hi + TOL]
    if inside:
        return min(inside, key=lambda a: a.hi - a.lo)
    best = min(anchors, key=lambda a: abs((a.lo + a.hi) / 2 - x_center), default=None)
    if best and abs((best.lo + best.hi) / 2 - x_center) <= 0.75 * (best.hi - best.lo):
        return best
    return None


def _years_in_columns(table: ExtractedTable, rows, info, cfg: Config) -> TableModel:
    notes: List[str] = []
    row_ids = list(rows)
    fy_hdr = [c for r in row_ids[:HEADER_SEARCH_ROWS] for c in rows[r] if info[id(c)].fy]
    x_values = min(c.bbox[0] for c in fy_hdr)

    # Header band = rows before the first row holding a number in the value zone
    # (a row of column numbers "1 2 3 ..." still counts as header).
    first_body = None
    numbering_rows = set()
    for r in row_ids:
        if _is_numbering_row(rows[r], info):
            numbering_rows.add(r)
            continue
        if any(_cx(c) >= x_values - TOL and info[id(c)].num.status in ("ok", "multi")
               and not (info[id(c)].serial and c is min(rows[r], key=lambda x: x.bbox[0]))
               for c in rows[r]):
            first_body = r
            break
    if first_body is None:
        return TableModel(table.table_id, "years_in_columns", [], [], ["year header but no numeric rows"])
    header_rows = [r for r in row_ids if r < first_body] or [row_ids[0]]
    header_cells = [c for r in header_rows if r not in numbering_rows for c in rows[r]]
    projection_caption = any(p.search(table.caption or "") for p in cfg.projection_caption_patterns)
    anchors, anchor_notes = _build_column_anchors(header_cells, info, cfg, projection_caption)
    notes.extend(anchor_notes)
    if not anchors:
        return TableModel(table.table_id, "years_in_columns", [], [], ["year cells not in header band"])
    x_label_limit = min(a.lo for a in anchors) - TOL
    # Header cells over value columns that are not years ("Percentage of Actuals to BE").
    non_year_cols = [h for h in header_cells if not info[id(h)].fy and _cx(h) >= x_label_limit
                     and not any(_overlap(a.lo, a.hi, h.bbox[0], h.bbox[2]) > 0.5 * _width(h) for a in anchors)]
    ignored_cols: Dict[str, bool] = {}

    value_cells: List[ValueCell] = []
    section: Optional[str] = None
    section_unit: Optional[str] = None
    pending_label: Optional[Tuple[str, Optional[str], Optional[str]]] = None   # label-only row
    last_value_row: List[ValueCell] = []

    for r in row_ids:
        if r in header_rows:
            continue
        label_parts, serial, values, missing = [], None, [], []
        for c in rows[r]:
            num = info[id(c)].num
            in_value_zone = _cx(c) >= x_label_limit
            if in_value_zone and num.status in ("ok", "multi"):
                values.append(c)
            elif in_value_zone and num.status == "missing":
                missing.append(c)
            elif info[id(c)].serial and not label_parts and serial is None:
                serial = _clean(c.text)
            else:
                label_parts.append(_clean(c.text))
        label = _clean(" ".join(label_parts))

        if not values and not missing:
            if not label:
                continue
            if last_value_row and (label[:1].islower() or label.startswith("(")):
                for vc in last_value_row:           # wrapped label continuation
                    vc.raw_label = _clean(vc.raw_label + " " + label)
                continue
            unit = cfg.normalize_label(label).unit or cfg.detect_unit(label)
            if unit:
                section_unit = unit.unit
            elif not re.match(r"^\s*(?:[ivx]{1,4}|[a-h])[.)]?\s+[A-Z]", label):
                # a new heading without its own unit ends a unit-declaring block; roman /
                # letter sub-headings ("II Expenditure Management") stay inside it
                section_unit = None
            pending_label = (label, section, section_unit)
            section = label
            last_value_row = []
            continue

        if not label and pending_label:             # label printed on the row above its numbers
            label, section, _ = pending_label[0], pending_label[1], pending_label[2]
        pending_label = None
        row_unit_match = cfg.normalize_label(label).unit if label else None
        row_unit = row_unit_match.unit if row_unit_match else section_unit
        row_text = _clean(f"{serial or ''} {label} | " + " | ".join(_clean(c.text) for c in sorted(values + missing, key=lambda c: c.bbox[0])))

        current: List[ValueCell] = []
        for c in values:
            num = info[id(c)].num
            if num.status == "multi":
                spanned = sorted([a for a in anchors if _overlap(a.lo, a.hi, c.bbox[0], c.bbox[2]) > 0],
                                 key=lambda a: a.lo)
                if len(spanned) != len(num.parts):
                    notes.append(f"row '{label}': cell '{_clean(c.text)}' holds {len(num.parts)} numbers "
                                 f"but spans {len(spanned)} year columns; skipped")
                    continue
                pairs = [(a, p, ["split_cell"] + p.notes) for a, p in zip(spanned, num.parts)]
            else:
                a = _nearest_anchor(_cx(c), anchors)
                if a is None:
                    col = next((h for h in non_year_cols if h.bbox[0] - TOL <= _cx(c) <= h.bbox[2] + TOL), None)
                    if col is not None:
                        ignored_cols.setdefault(_clean(col.text), True)
                    else:
                        notes.append(f"row '{label}': value '{_clean(c.text)}' not under any year column")
                    continue
                pairs = [(a, num, list(num.notes))]
            for a, p, pnotes in pairs:
                current.append(ValueCell(
                    row_index=r, raw_label=label, row_serial=serial, section=section, row_unit=row_unit,
                    anchor_id=a.anchor_id, fy=a.fy, value_type=a.value_type, column_header=a.header_text,
                    raw_text=_clean(p.raw), value=p.value, parse_status=p.status, parse_notes=pnotes,
                    bbox=c.bbox, row_text=row_text, accepted=a.accepted, skip_reason=a.skip_reason,
                    component=a.component,
                ))
        value_cells.extend(current)
        last_value_row = current
    notes.extend(f"ignored non-year column '{name}'" for name in ignored_cols)
    return TableModel(table.table_id, "years_in_columns", anchors, value_cells, notes)


# ------------------------------------------------------------------ years in rows

def _years_in_rows(table: ExtractedTable, rows, info, cfg: Config) -> TableModel:
    notes: List[str] = []
    row_ids = list(rows)
    leftmost = {r: min(cs, key=lambda c: c.bbox[0]) for r, cs in rows.items()}
    fy_rows = [r for r in row_ids if info[id(leftmost[r])].fy]
    header_rows = [r for r in row_ids if r < fy_rows[0]]
    header_cells = [c for r in header_rows for c in rows[r] if c is not leftmost.get(r) or not info[id(c)].fy]

    anchors: List[Anchor] = []
    value_cells: List[ValueCell] = []
    for r in row_ids:
        if r in header_rows:
            continue
        lead = leftmost[r]
        fy = info[id(lead)].fy
        if not fy:
            notes.append(f"row '{_clean(lead.text)}' has no financial year; skipped")
            continue
        vtypes = cfg.value_types(lead.text)
        reason = _column_rejection("", vtypes, cfg)
        anchor = Anchor(anchor_id=f"r{len(anchors)}", fy=fy, value_type=vtypes[0] if len(vtypes) == 1 else None,
                        header_text=_clean(lead.text), lo=lead.bbox[1], hi=lead.bbox[3],
                        accepted=reason is None, skip_reason=reason)
        anchors.append(anchor)
        others = sorted([c for c in rows[r] if c is not lead], key=lambda c: c.bbox[0])
        row_text = _clean(" | ".join(_clean(c.text) for c in [lead] + others))
        for c in others:
            num = info[id(c)].num
            if num.status != "ok":
                continue
            heads = sorted([h for h in header_cells if h.bbox[0] - TOL <= _cx(c) <= h.bbox[2] + TOL],
                           key=lambda h: (h.row, h.bbox[0]))
            label = _clean(" ".join(h.text for h in heads))
            if not label:
                notes.append(f"{fy}: value '{_clean(c.text)}' has no column header; skipped")
                continue
            unit_match = cfg.normalize_label(label).unit
            value_cells.append(ValueCell(
                row_index=r, raw_label=label, row_serial=None, section=None,
                row_unit=unit_match.unit if unit_match else None,
                anchor_id=anchor.anchor_id, fy=fy, value_type=anchor.value_type, column_header=anchor.header_text,
                raw_text=_clean(c.text), value=num.value, parse_status=num.status, parse_notes=list(num.notes),
                bbox=c.bbox, row_text=row_text, accepted=anchor.accepted, skip_reason=anchor.skip_reason,
            ))
    return TableModel(table.table_id, "years_in_rows", anchors, value_cells, notes)
