"""Analytical layer: a state x financial-year panel generated from reconciled observations.

The panel is never edited by hand and never read back by the pipeline; it is rebuilt from
the raw observation layer on every run. Every panel value appears in panel_long.csv with
the observation id(s) it came from, so it can be traced to a document, page and table.

A reconciled value enters the panel unless:
  * the cell is an unresolved conflict between same-priority sources;
  * its unit is not INR crore (no silent conversion in Phase 1);
  * its representative observation carries an error-severity flag;
  * it is a deficit whose sign convention is not confirmed as the canonical one.
Values with warnings do enter the panel; they are listed in `values_needing_review`.

central_transfers, if not reported as one row, is derived as the sum of its components
(union_tax_share + grants_from_centre) and marked as derived.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from statefin.config import Config
from statefin.reconciliation.sources import CellConsensus, CellKey


def _exclusion(cell: CellConsensus, cfg: Config) -> Optional[str]:
    rep = cell.representative
    if cell.status == "conflict" or rep is None:
        return "unresolved conflict between same-priority sources"
    if rep.value_crore is None:
        return f"unit is {rep.unit}; not convertible to INR crore"
    errors = [f["check"] for f in rep.validation_flags if f["severity"] == "error"]
    if errors and rep.review_decision not in ("accepted", "corrected"):
        return f"error flag(s): {', '.join(errors)}"
    spec = cfg.variables[cell.variable]
    if spec.canonical_sign_convention and rep.sign_convention != spec.canonical_sign_convention:
        return f"sign convention '{rep.sign_convention}' not confirmed as '{spec.canonical_sign_convention}'"
    return None


def _long_row(cell: CellConsensus, value: Optional[float], panel_status: str, reason: str,
              cfg: Optional[Config] = None) -> dict:
    rep = cell.representative
    # a definition shared by agreeing sources: union of their tags (one footnote is enough)
    tags = sorted({t for o in (cell.agreeing or ([rep] if rep else [])) for t in o.definition_tags})
    substantive = sorted(set(tags) & cfg.substantive_tags(cell.variable)) if cfg else tags
    return {
        "state": cell.state,
        "state_code": cell.state_code,
        "financial_year": cell.financial_year,
        "variable": cell.variable,
        "value": value,
        "unit": "INR crore" if rep and rep.value_crore is not None else (rep.unit if rep else None),
        "printed_unit": rep.unit if rep else None,
        "panel_status": panel_status,
        "reconciliation_status": cell.status,
        "reason": reason,
        "representative_observation_id": rep.observation_id if rep else None,
        "agreeing_observation_ids": " | ".join(o.observation_id for o in cell.agreeing),
        "other_observation_ids": " | ".join(o.observation_id for o in cell.observations
                                            if not any(o is a for a in cell.agreeing)),
        "n_observations": len(cell.observations),
        "source_document": rep.source_document if rep else None,
        "source_page": rep.source_page if rep else None,
        "source_table": rep.source_table if rep else None,
        "confidence": rep.confidence if rep else None,
        "review_status": rep.review_status if rep else None,
        "review_decision": rep.review_decision if rep else None,
        "raw_label": rep.raw_label if rep else None,
        "definition_tags": " | ".join(tags),
        "substantive_tags": " | ".join(substantive),
        "definition_variant": cell.variant,
        "series_base": rep.series_base if rep else None,
        "document_type": rep.document_type if rep else None,
        "break_before": "",
    }


def build_panel(cells: Dict[CellKey, CellConsensus], cfg: Config) -> Tuple[List[dict], List[dict]]:
    long_rows: List[dict] = []
    included: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)

    for key in sorted(cells):
        cell = cells[key]
        reason = _exclusion(cell, cfg)
        row = _long_row(cell, None if reason else cell.value, "excluded" if reason else "included", reason or "", cfg)
        if not reason and cell.variant:
            row["reason"] = f"definition variant '{cell.variant}' (no canonical-definition source)"
        long_rows.append(row)
        if not reason:
            included[(cell.state, cell.financial_year)][cell.variable] = row

    for var, spec in cfg.variables.items():
        if len(spec.derive_from) < 2:
            continue
        for (state, fy), by_var in included.items():
            parts = [by_var.get(p) for p in spec.derive_from]
            if var in by_var or not all(parts):
                continue
            value = round(sum(p["value"] for p in parts), 2)
            row = {
                "state": state, "state_code": parts[0]["state_code"], "financial_year": fy, "variable": var,
                "value": value, "unit": "INR crore", "panel_status": "derived", "reconciliation_status": "derived",
                "reason": " + ".join(spec.derive_from),
                "representative_observation_id": None,
                "agreeing_observation_ids": " | ".join(p["representative_observation_id"] for p in parts),
                "other_observation_ids": "", "n_observations": len(parts),
                "source_document": parts[0]["source_document"], "source_page": None, "source_table": None,
                "confidence": min(p["confidence"] for p in parts),
                "review_status": "NEEDS_REVIEW" if any(p["review_status"] == "NEEDS_REVIEW" for p in parts)
                else "AUTO_ACCEPTED",
                "review_decision": None, "raw_label": None, "definition_tags": "", "substantive_tags": "",
                "definition_variant": "", "printed_unit": "INR crore", "series_base": None,
                "document_type": parts[0].get("document_type"), "break_before": "",
            }
            long_rows.append(row)
            by_var[var] = row

    breaks = find_breaks(long_rows, cfg)
    by_cell = {(b["state"], b["financial_year"], b["variable"]): b for b in breaks if b["variable"] != "all"}
    state_breaks = {(b["state"], b["financial_year"]): b for b in breaks if b["variable"] == "all"}
    for row in long_rows:
        b = by_cell.get((row["state"], row["financial_year"], row["variable"]))
        sb = state_breaks.get((row["state"], row["financial_year"]))
        row["break_before"] = " ; ".join(f"{x['break_type']}: {x['detail']}" for x in (b, sb) if x)

    wide_rows: List[dict] = []
    variables = cfg.panel_variables
    for (state, fy) in sorted(included):
        by_var = included[(state, fy)]
        if not any(v in by_var for v in variables):
            continue
        row = {"state": state, "state_code": next(iter(by_var.values()))["state_code"], "financial_year": fy}
        for v in variables:
            row[v] = by_var[v]["value"] if v in by_var else None
        row["values_needing_review"] = " | ".join(v for v in variables if v in by_var
                                                  and by_var[v]["review_status"] == "NEEDS_REVIEW")
        row["derived_values"] = " | ".join(v for v in variables if v in by_var and by_var[v]["panel_status"] == "derived")
        gsdp = by_var.get("gsdp")
        row["gsdp_base_year"] = gsdp.get("series_base") if gsdp else None
        row["structural_breaks"] = " ; ".join(
            ([f"state: {state_breaks[(state, fy)]['detail']}"] if (state, fy) in state_breaks else [])
            + [f"{v}: {by_cell[(state, fy, v)]['break_type']} ({by_cell[(state, fy, v)]['detail']})"
               for v in variables if (state, fy, v) in by_cell and by_cell[(state, fy, v)]["break_type"] != "source_change"])
        wide_rows.append(row)
    long_rows.sort(key=lambda r: (r["state"], r["variable"], r["financial_year"]))
    return long_rows, wide_rows


def find_breaks(long_rows: List[dict], cfg: Config) -> List[dict]:
    """Where a series is not comparable across consecutive panel years.

    state_reorganisation  from states.yaml (e.g. Andhra Pradesh 2014-15)
    series_base           GSDP (or other series) base year changes, e.g. 2004-05 -> 2011-12
    definition            the definition tags of the panel value change (e.g. salary starts
                          including grants-in-aid salaries)
    source_change         the document type behind the value changes (informational)
    """
    out: List[dict] = []
    included = [r for r in long_rows if r["panel_status"] in ("included", "derived")]
    states = sorted({r["state"] for r in included})
    for state in states:
        spec = cfg.state_by_name(state)
        years = sorted({r["financial_year"] for r in included if r["state"] == state})
        for b in (spec.breaks if spec else []):
            if years and years[0] < b["fy"] <= years[-1]:
                out.append({"state": state, "variable": "all", "financial_year": b["fy"],
                            "break_type": "state_reorganisation", "detail": b["reason"]})
        series: Dict[str, List[dict]] = defaultdict(list)
        for r in included:
            if r["state"] == state:
                series[r["variable"]].append(r)
        for var, rows in series.items():
            rows.sort(key=lambda r: r["financial_year"])
            for prev, cur in zip(rows, rows[1:]):
                if int(cur["financial_year"][:4]) - int(prev["financial_year"][:4]) != 1:
                    continue
                if prev.get("series_base") and cur.get("series_base") and prev["series_base"] != cur["series_base"]:
                    out.append({"state": state, "variable": var, "financial_year": cur["financial_year"],
                                "break_type": "series_base",
                                "detail": f"base year {prev['series_base']} -> {cur['series_base']}"})
                elif (prev.get("substantive_tags") or "") != (cur.get("substantive_tags") or "") \
                        or (prev.get("definition_variant") or "") != (cur.get("definition_variant") or ""):
                    if cur["panel_status"] == "derived":
                        continue
                    before = " ".join(filter(None, [prev.get("definition_variant"), prev.get("substantive_tags")])) or "canonical"
                    after = " ".join(filter(None, [cur.get("definition_variant"), cur.get("substantive_tags")])) or "canonical"
                    out.append({"state": state, "variable": var, "financial_year": cur["financial_year"],
                                "break_type": "definition", "detail": f"[{before}] -> [{after}]"})
                elif prev.get("document_type") and cur.get("document_type") and prev["document_type"] != cur["document_type"]:
                    out.append({"state": state, "variable": var, "financial_year": cur["financial_year"],
                                "break_type": "source_change",
                                "detail": f"{prev['document_type']} -> {cur['document_type']}"})
    return out
