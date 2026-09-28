"""Per-observation and cross-year plausibility checks. Checks only flag; they never change a value."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional

from statefin.config import Config
from statefin.extraction.parsers import fy_start_year
from statefin.models import Observation, PageText
from statefin.reconciliation.sources import CellConsensus, CellKey

_PARSE_NOTE_SEVERITY = {
    "parentheses_negative": ("warning", "value was in parentheses and read as negative"),
    "split_cell": ("warning", "value was split out of a cell holding several numbers"),
    "split_sign": ("info", "minus sign was separated from the digits in the PDF"),
    "footnote_marker": ("info", "value carried a footnote marker"),
}


def check_observation(o: Observation, cfg: Config, report_year: Optional[str] = None,
                      document_date: Optional[str] = None) -> None:
    var = cfg.variables[o.canonical_variable]

    if o.value < 0 and not var.allow_negative:
        o.add_flag("impossible_negative", "error", f"{o.canonical_variable} cannot be negative (got {o.value:,.2f})")

    if o.unit is None:
        o.add_flag("unit_unknown", "warning", "no unit declared for this table/row")
    elif o.value_crore is None:
        o.add_flag("unit_not_crore", "warning", f"unit is {o.unit}; cannot be converted to INR crore")
    else:
        if o.unit != "INR crore":
            o.add_flag("unit_converted", "info", f"printed in {o.unit}: {o.value:,.2f} = {o.value_crore:,.4f} crore "
                                                  "(conversion used for comparison and the panel; value kept as printed)")
        if var.plausible_range and not (var.plausible_range[0] <= o.value_crore <= var.plausible_range[1]):
            o.add_flag("implausible_magnitude", "warning",
                       f"{o.value_crore:,.2f} crore outside the sanity range {var.plausible_range} (possible unit/scale error)")

    start = fy_start_year(o.financial_year)
    if start < int(cfg.get("validation.fy_min_start_year", 1950)) or start > date.today().year:
        o.add_flag("fy_out_of_range", "error", f"financial year {o.financial_year} is implausible")
    if report_year and start > fy_start_year(report_year):
        o.add_flag("fy_after_report_year", "warning",
                   f"{o.financial_year} is after the document's reference year {report_year} (estimate/projection?)")

    # Actuals for a year cannot be printed before that year has closed (31 March).
    if document_date and document_date >= "1990" and f"{start + 1}-03-31" > document_date:
        o.add_flag("fy_not_closed_at_document_date", "error",
                   f"{o.financial_year} ends after the document date {document_date} (PDF metadata), so this "
                   "cannot be an actual; probably an estimate or projection")

    if var.canonical_sign_convention:
        if o.sign_convention == "deficit_positive":
            o.add_flag("sign_convention_differs", "info",
                       f"printed deficit-positive; canonical is {var.canonical_sign_convention}. Value NOT altered; "
                       "compared in canonical terms during reconciliation")
        elif o.sign_convention == "ambiguous":
            o.add_flag("sign_convention_ambiguous", "warning", "label declares conflicting sign conventions")

    for note, (severity, message) in _PARSE_NOTE_SEVERITY.items():
        if f"value parse note: {note}" in o.confidence_reasons:
            o.add_flag(f"parse_{note}", severity, message)

    if o.mapping_note:
        o.add_flag("definition_caveat", "warning", o.mapping_note)
    if o.label_footnote:
        o.add_flag("label_footnote", "info", o.label_footnote)
    if o.value_type == "unqualified":
        o.add_flag("column_unqualified", "info", "year column has no Actual/BE/RE qualifier; treated as actuals")


def check_time_series(cells: Dict[CellKey, CellConsensus], cfg: Config) -> None:
    """Flag large year-on-year movements in non-negative variables (on the consensus value)."""
    max_change = float(cfg.get("validation.yoy_max_change", 0.5))
    jump = float(cfg.get("validation.yoy_magnitude_jump", 10))
    series: Dict[tuple, List[CellConsensus]] = defaultdict(list)
    for (state, fy, var), cell in cells.items():
        if cell.value is not None and not cfg.variables[var].allow_negative:
            series[(state, var)].append(cell)
    for (_, var), items in series.items():
        items.sort(key=lambda c: c.financial_year)
        for prev, cur in zip(items, items[1:]):
            if fy_start_year(cur.financial_year) - fy_start_year(prev.financial_year) != 1 or prev.value <= 0:
                continue
            ratio = cur.value / prev.value
            rep = cur.representative
            if ratio >= jump or ratio <= 1 / jump:
                rep.add_flag("possible_scale_break", "error",
                             f"{var} moves {ratio:.1f}x from {prev.financial_year} ({prev.value:,.2f}) to "
                             f"{cur.financial_year} ({cur.value:,.2f}); possible unit error. Not corrected")
            elif abs(ratio - 1) > max_change:
                rep.add_flag("large_yoy_change", "warning",
                             f"{var} changes {100 * (ratio - 1):+.1f}% from {prev.financial_year} ({prev.value:,.2f}); "
                             "may be genuine. Not corrected")


def check_unit_consistency(observations: List[Observation]) -> None:
    """Currency units are converted to crore; a series mixing currency with percent is a problem."""
    by_series: Dict[tuple, set] = defaultdict(set)
    for o in observations:
        if o.unit:
            by_series[(o.state, o.canonical_variable)].add("currency" if o.value_crore is not None else o.unit)
    for o in observations:
        kinds = by_series[(o.state, o.canonical_variable)]
        if len(kinds) > 1:
            o.add_flag("mixed_units", "warning", f"series mixes incompatible units: {sorted(map(str, kinds))}")


def check_panel_eligibility(cells: Dict[CellKey, CellConsensus], cfg: Config) -> None:
    """Deficits enter the panel only under a known sign convention; say so rather than guess."""
    for (_, _, var), cell in cells.items():
        spec = cfg.variables[var]
        rep = cell.representative
        if rep is None or not spec.canonical_sign_convention:
            continue
        if rep.sign_convention != spec.canonical_sign_convention:
            rep.add_flag("sign_convention_unconfirmed", "warning",
                         f"sign convention is '{rep.sign_convention}', panel requires "
                         f"'{spec.canonical_sign_convention}'; excluded from panel until confirmed")


def _squash(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def check_provenance(observations: List[Observation], pages_by_doc: Dict[str, List[PageText]]) -> None:
    """Every value's printed text must occur verbatim on the page it is attributed to."""
    for o in observations:
        pages = pages_by_doc.get(o.document_id)
        if not pages or o.source_page > len(pages):
            o.add_flag("provenance_unverifiable", "warning", "source page text not available")
            continue
        page_text = _squash(pages[o.source_page - 1].text)
        digits = _squash(o.raw_value_text).lstrip("-–−(+)").rstrip("*#@$")
        if digits not in page_text:
            o.add_flag("provenance_text_mismatch", "error",
                       f"'{o.raw_value_text}' not found on page {o.source_page} of {o.source_document}")
