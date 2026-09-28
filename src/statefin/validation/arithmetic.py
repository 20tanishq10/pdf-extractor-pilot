"""Table-internal arithmetic: the formulas printed in row labels, and "Total" rows.

CAG tables state their own arithmetic: "4. Revenue Receipts (1+2+3)", "15. Revenue Deficit
(4-10)", "Fiscal Deficit {(4+5+6)-14}", "Total". Checking them per year column is the
strongest deterministic test available: a misread digit (common after OCR), a value put
in the wrong column, or a row attached to the wrong label almost never still adds up.

Formulas refer to row serials ("4", "(i)", "a"). A table that continues over several pages
(repeated header, inherited caption) is checked as one logical table, because appendix
formulas refer across pages ("20. Revenue Deficit (1-10)" on page 131 uses rows on 129-130).

Results, all flag-only:
  * a formula row that adds up -> info `arithmetic_ok` on its observation;
  * one that does not -> warning `arithmetic_mismatch` on the formula row and on every
    referenced row that is an observation, naming the numbers involved;
  * a deficit row whose formula is receipts minus expenditure and adds up fixes the sign
    convention from the arithmetic itself (surplus-positive if value = receipts - expenditure,
    deficit-positive if value = expenditure - receipts).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from statefin.config import Config
from statefin.models import Observation, TableModel

_GROUP = re.compile(r"[({][0-9ivxa-h\s+\-–−(){}]*[+\-–−][0-9ivxa-h\s+\-–−(){}]*[)}]", re.I)
_TOKEN = re.compile(r"\s*([+\-–−]|[({]|[)}]|[0-9]{1,3}|[ivx]{1,5}|[a-h])", re.I)
_SERIAL = re.compile(r"^\(?\s*([0-9]{1,3}|[ivxlc]{1,5}|[a-z])\s*[.),]?\s*\)?", re.I)


def parse_formula(label: str) -> Optional[List[Tuple[int, str]]]:
    """'Revenue Receipts (1+2+3)' -> [(+1,'1'),(+1,'2'),(+1,'3')]; '{(4+5+6)-14}' -> 4,5,6 minus 14."""
    groups = [m.group(0) for m in _GROUP.finditer(label or "")]
    if not groups:
        return None
    expr = max(groups, key=len)
    pos, terms, sign_stack, sign = 0, [], [1], 1
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            return None
        tok, pos = m.group(1), m.end()
        if tok in "({":
            sign_stack.append(sign_stack[-1] * sign)
            sign = 1
        elif tok in ")}":
            if len(sign_stack) > 1:
                sign_stack.pop()
            sign = 1
        elif tok in "+-–−":
            sign = -1 if tok != "+" else 1
        else:
            terms.append((sign_stack[-1] * sign, tok))      # case kept: "(A+B)" is not "(a+b)"
            sign = 1
    return terms if len(terms) >= 2 else None


def row_serial(raw_label: str, row_serial_cell: Optional[str]) -> Optional[str]:
    text = (row_serial_cell or raw_label or "").strip()
    m = _SERIAL.match(text)
    if not m:
        return None
    if not row_serial_cell and not re.match(r"^\(?\s*\w{1,5}\s*[.),]", text):
        return None      # a label that merely starts with a word, not "10." / "(i)"
    return m.group(1)


@dataclass
class Row:
    table_id: str
    row_index: int
    label: str
    serial: Optional[str]
    values: Dict[str, Tuple[float, str]]       # fy -> (value, raw text)


@dataclass
class CheckResult:
    table_id: str
    label: str
    fy: str
    formula: str
    printed: float
    computed: float
    ok: bool
    flipped: bool = False                      # value equals the negative of the formula
    magnitudes: bool = False                   # holds on absolute values (deficits printed with a sign)
    refs: List[Tuple[str, str]] = field(default_factory=list)   # (table_id, label) of referenced rows


def _decimals(raw: str) -> int:
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
    return len(digits.rsplit(".", 1)[1]) if "." in digits else 0


def _logical_tables(tables, models: Dict[str, TableModel]) -> List[List[str]]:
    """Chains of table ids linked by 'continued' captions, in page order."""
    by_id = {t.table_id: t for t in tables}
    chains: Dict[str, List[str]] = {}
    order = []
    for t in sorted(tables, key=lambda t: (t.page_number, t.index_on_page)):
        root = t.table_id
        parent = t.caption_inherited_from
        while parent and parent in by_id:
            root = parent
            parent = by_id[parent].caption_inherited_from
        if root not in chains:
            chains[root] = []
            order.append(root)
        chains[root].append(t.table_id)
    return [chains[r] for r in order if any(tid in models for tid in chains[r])]


def _rows(table_ids: List[str], models: Dict[str, TableModel]) -> List[Row]:
    rows: Dict[Tuple[str, int], Row] = {}
    for tid in table_ids:
        model = models.get(tid)
        if not model or model.orientation != "years_in_columns":
            continue
        for vc in model.value_cells:
            if not vc.accepted or vc.value is None or not vc.fy:
                continue
            key = (tid, vc.row_index)
            if key not in rows:
                rows[key] = Row(tid, vc.row_index, vc.raw_label, row_serial(vc.raw_label, vc.row_serial), {})
            rows[key].values.setdefault(vc.fy, (vc.value, vc.raw_text))
    return list(rows.values())


def check_tables(tables, models: Dict[str, TableModel]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for chain in _logical_tables(tables, models):
        rows = _rows(chain, models)
        for i, row in enumerate(rows):
            terms = parse_formula(row.label)
            if not terms:
                continue
            refs = []
            for sign, serial in terms:
                # "Total (A+B)" means the rows "Total (A)" and "Total (B)", not the first rows of A and B
                subtotal = re.compile(r"^\s*(?:sub-?\s*)?total\s*\(\s*" + re.escape(serial) + r"\s*\)\s*$", re.I)
                pool = [r for r in rows if subtotal.match(r.label)] or [r for r in rows if r.serial == serial]
                before = [r for r in pool if rows.index(r) < i]
                after = [r for r in pool if rows.index(r) > i]
                ref = before[-1] if before else (after[0] if after else None)
                if ref is None:
                    refs = None
                    break
                refs.append((sign, ref))
            if not refs:
                continue
            formula = " ".join(("+" if sg > 0 else "-") + ser for sg, ser in terms).lstrip("+")
            for fy, (printed, raw) in row.values.items():
                parts = [(s, r.values.get(fy)) for s, r in refs]
                if any(v is None for _, v in parts):
                    continue
                computed = sum(s * v[0] for s, v in parts)
                dec = min([_decimals(raw)] + [_decimals(v[1]) for _, v in parts])
                tol = max(0.51 * 10 ** -dec * (len(parts) + 1), 0.0005 * abs(printed))
                direct = abs(printed - computed) <= tol
                flipped = not direct and abs(printed + computed) <= tol and abs(computed) > tol
                # "Primary Deficit (16-11)": CAG states deficit formulas on magnitudes while
                # printing deficits as negative numbers (|FD| - IP = |PD|).
                magnitudes = False
                if not (direct or flipped) and _DEFICIT.search(row.label):
                    on_abs = sum(s * abs(v[0]) for s, v in parts)
                    magnitudes = abs(abs(printed) - abs(on_abs)) <= tol
                results.append(CheckResult(row.table_id, row.label, fy, formula, printed, computed,
                                           ok=direct or flipped or magnitudes, flipped=flipped, magnitudes=magnitudes,
                                           refs=[(r.table_id, r.label) for _, r in refs]))
    return results


_DEFICIT = re.compile(r"deficit|surplus", re.I)
_RECEIPT = re.compile(r"receipt|revenue receipts|non-?debt", re.I)
_EXPEND = re.compile(r"expenditure|disbursement|outlay", re.I)


def apply_arithmetic(docs, observations: List[Observation], cfg: Config) -> Dict[str, Dict[str, int]]:
    """Run checks for every document; flag observations; infer deficit sign conventions."""
    by_cell = defaultdict(list)
    for o in observations:
        by_cell[(o.document_id, o.table_id, o.raw_label, o.financial_year)].append(o)
    summary: Dict[str, Dict[str, int]] = {}
    for d in docs:
        results = check_tables(d.tables, d.models)
        doc_id = d.record.document_id
        per_table: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ok": 0, "mismatch": 0})
        for r in results:
            per_table[r.table_id]["ok" if r.ok else "mismatch"] += 1
            detail = (f"{r.formula} = {r.computed:,.2f}; printed {r.printed:,.2f}"
                      + (" (sign reversed)" if r.flipped else ""))
            targets = by_cell.get((doc_id, r.table_id, r.label, r.fy), [])
            for o in targets:
                if r.ok:
                    o.add_flag("arithmetic_ok", "info", f"row formula holds: {detail}")
                    spec = cfg.variables[o.canonical_variable]
                    if spec.canonical_sign_convention and o.sign_convention in (None, "unspecified") and not r.magnitudes:
                        pos = [lbl for (_t, lbl), (s, _) in zip(r.refs, parse_formula(r.label) or []) if s > 0]
                        neg = [lbl for (_t, lbl), (s, _) in zip(r.refs, parse_formula(r.label) or []) if s < 0]
                        if pos and neg and all(_RECEIPT.search(x) for x in pos) and all(_EXPEND.search(x) for x in neg):
                            o.sign_convention = "deficit_positive" if r.flipped else "surplus_positive"
                            o.add_flag("sign_convention_inferred", "info",
                                       f"from the printed formula: value = {'expenditure - receipts' if r.flipped else 'receipts - expenditure'}")
                else:
                    o.add_flag("arithmetic_mismatch", "warning", f"row formula does not hold: {detail}")
            if not r.ok:
                for tid, lbl in r.refs:
                    for o in by_cell.get((doc_id, tid, lbl, r.fy), []):
                        o.add_flag("arithmetic_mismatch", "warning",
                                   f"used in '{r.label}' ({r.fy}), which does not add up: {detail}")
        summary[doc_id] = dict(per_table)
    return summary
