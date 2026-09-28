"""Write run outputs: candidates.json, extractions.json/.csv, review_queue.csv, panel files.

CSV files are UTF-8 with a byte-order mark so Excel shows ₹ and ✓ correctly.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from statefin.config import Config
from statefin.models import PIPELINE_VERSION, Observation
from statefin.reconciliation.sources import CellConsensus, CellKey

_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}

REVIEW_COLUMNS = [
    "issue_type", "severity", "state", "financial_year", "canonical_variable", "value", "unit",
    "confidence", "confidence_tier", "review_reasons", "flag_details", "raw_label", "column_header",
    "source_document", "source_page", "source_page_label", "source_table", "source_text", "page_snapshot",
    "reconciliation_status", "observation_id", "reviewer_decision", "reviewer_value", "reviewer_note",
]


SUGGESTION_COLUMNS = ["raw_label", "normalized_label", "llm_variable", "llm_status", "confidence", "model",
                      "rule_variable", "agreement", "reason", "tables", "source_document"]


def _label_suggestions(docs, cfg: Config) -> List[dict]:
    """Every label the LLM classified, vs the rules. Confirmed rows can be promoted to aliases
    in variables.yaml, which moves knowledge from the model into the deterministic layer."""
    from statefin.extraction.mapper import match_label
    rows: Dict[tuple, dict] = {}
    for d in docs:
        for tid, interp in d.interpretations.items():
            for label, dec in interp.final.items():
                rule = match_label(label, cfg)
                key = (label, dec.variable, dec.status)
                row = rows.setdefault(key, {
                    "raw_label": label, "normalized_label": rule.normalized, "llm_variable": dec.variable or "",
                    "llm_status": dec.status, "confidence": dec.confidence, "model": dec.model,
                    "rule_variable": rule.variable or ("(rejected: " + rule.rejected_reason + ")" if rule.rejected_reason else ""),
                    "agreement": ("agree" if rule.variable == dec.variable and dec.status == "mapped"
                                  else "llm_only" if dec.status == "mapped" and not rule.variable
                                  else "rules_only" if rule.variable and dec.status != "mapped"
                                  else "disagree" if rule.variable and dec.variable != rule.variable
                                  else "neither"),
                    "reason": dec.reason, "tables": [], "source_document": d.record.original_filename})
                row["tables"].append(tid)
    order = {"disagree": 0, "llm_only": 1, "rules_only": 2, "agree": 3, "neither": 4}
    return sorted(rows.values(), key=lambda r: (order[r["agreement"]], r["raw_label"]))


def _cell(value):
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "check" in value[0]:
            return " || ".join(f"{f['severity']}:{f['check']}: {f['message']}" for f in value)
        return " | ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _cell(r.get(k)) for k in columns})


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)


def _doc_summary(d) -> dict:
    r = d.record
    return {
        "document_id": r.document_id, "sha256": r.sha256, "file": r.original_filename,
        "cached_path": r.cached_path, "source_url": r.source_url, "source_kind": r.source_kind,
        "state": r.state, "state_source": r.state_source, "state_evidence": r.state_evidence,
        "document_type": r.document_type, "document_type_source": r.document_type_source,
        "report_year": r.report_year, "pages": r.page_count, "status": r.status, "error": d.error,
    }


def _review_rows(observations: List[Observation], cells: Dict[CellKey, CellConsensus], cfg: Config,
                 docs=()) -> List[dict]:
    rows = []
    for d in docs:
        for u in d.llm_unresolved:
            rows.append({
                "issue_type": "LLM_UNRESOLVED_ROW", "severity": "warning", "state": d.record.state,
                "raw_label": u["label"], "source_document": d.record.original_filename, "source_page": u["page"],
                "source_table": u["caption"], "review_reasons": [f"{u['model']}: {u['reason']}"],
                "canonical_variable": "?",
            })
    for o in observations:
        if o.review_status != "NEEDS_REVIEW":
            continue
        severities = [f["severity"] for f in o.validation_flags if f["severity"] in ("error", "warning")]
        severity = min(severities, key=_SEVERITY_RANK.get) if severities else "low_confidence"
        row = o.to_dict()
        row.update({
            "issue_type": "OBSERVATION",
            "severity": severity,
            "flag_details": [f for f in o.validation_flags if f["severity"] != "info"],
        })
        rows.append(row)

    states = sorted({o.state or "UNKNOWN" for o in observations}) or ["UNKNOWN"]
    present = {(k[0], k[2]) for k in cells}
    for state in states:
        for var in cfg.core_variables:
            spec = cfg.variables[var]
            derivable = spec.derive_from and all((state, p) in present for p in spec.derive_from)
            if (state, var) not in present and not derivable:
                rows.append({
                    "issue_type": "MISSING_VARIABLE", "severity": "warning", "state": state,
                    "canonical_variable": var,
                    "review_reasons": ["no observation extracted from any processed document"],
                })
    order = {"error": 0, "warning": 1, "low_confidence": 2}
    rows.sort(key=lambda r: (order.get(r["severity"], 3), r["issue_type"] != "OBSERVATION",
                             str(r.get("state")), str(r.get("canonical_variable")), str(r.get("financial_year"))))
    return rows


def write_outputs(out_dir: Path, docs, observations: List[Observation], cells: Dict[CellKey, CellConsensus],
                  panel_long: List[dict], panel_wide: List[dict], cfg: Config, data_dir: Path,
                  reviews: Optional[List[dict]] = None) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    paths = {name: out_dir / name for name in (
        "candidates.json", "extractions.json", "extractions.csv", "review_queue.csv",
        "panel.csv", "panel_long.csv", "summary.txt")}

    candidates = {"generated_at": now, "pipeline_version": PIPELINE_VERSION, "documents": []}
    for d in docs:
        mapped: Dict[str, Dict[str, int]] = {}
        for o in d.observations:
            mapped.setdefault(o.table_id, {}).setdefault(o.canonical_variable, 0)
            mapped[o.table_id][o.canonical_variable] += 1
        tables = []
        for t in d.tables:
            m = d.models.get(t.table_id)
            tables.append({
                "table_id": t.table_id, "page": t.page_number, "page_label": t.page_label,
                "caption": t.caption, "caption_inherited_from": t.caption_inherited_from,
                "unit": t.unit, "unit_source": t.unit_source, "cited_source": t.cited_source,
                "strategy": t.strategy, "page_rotation": t.page_rotation,
                "orientation": m.orientation if m else None,
                "year_columns": [{"fy": a.fy, "value_type": a.value_type, "header": a.header_text,
                                  "accepted": a.accepted, "skip_reason": a.skip_reason} for a in (m.anchors if m else [])],
                "value_cells": len(m.value_cells) if m else 0,
                "mapped_variables": mapped.get(t.table_id, {}),
                "rejected_labels": [r for r in d.rejected if r["table_id"] == t.table_id],
                "arithmetic_checks": d.arithmetic.get(t.table_id, {}),
                "notes": m.notes if m else [],
            })
        candidates["documents"].append({
            **_doc_summary(d),
            "pages_needing_ocr": [p.page_number for p in d.pages if p.needs_ocr],
            "candidate_pages": [c.to_dict() for c in d.candidates if c.is_candidate],
            "other_scored_pages": [{"page_number": c.page_number, "score": c.score, "rank": c.rank}
                                   for c in d.candidates if not c.is_candidate],
            "tables": tables,
        })
    _write_json(paths["candidates.json"], candidates)

    _write_json(paths["extractions.json"], {
        "generated_at": now,
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": cfg.section_hash("variables", "source_rules", "pipeline"),
        "data_dir": str(data_dir),
        "documents": [_doc_summary(d) for d in docs],
        "observations": [o.to_dict() for o in observations],
    })

    obs_columns = list(Observation.__dataclass_fields__.keys())
    ordered = sorted(observations, key=lambda o: (str(o.state), o.canonical_variable, o.financial_year,
                                                  o.source_priority, o.source_page))
    _write_csv(paths["extractions.csv"], [o.to_dict() for o in ordered], obs_columns)
    _write_csv(paths["review_queue.csv"], _review_rows(observations, cells, cfg, docs), REVIEW_COLUMNS)
    if any(d.interpretations for d in docs):
        paths["llm_label_suggestions.csv"] = out_dir / "llm_label_suggestions.csv"
        _write_csv(paths["llm_label_suggestions.csv"], _label_suggestions(docs, cfg), SUGGESTION_COLUMNS)

    wide_cols = (["state", "state_code", "financial_year"] + cfg.panel_variables
                 + ["gsdp_base_year", "values_needing_review", "derived_values", "structural_breaks"])
    _write_csv(paths["panel.csv"], panel_wide, wide_cols)
    long_cols = list(panel_long[0].keys()) if panel_long else ["state", "financial_year", "variable", "value"]
    _write_csv(paths["panel_long.csv"], panel_long, long_cols)
    from statefin.export.panel import find_breaks
    paths["panel_breaks.csv"] = out_dir / "panel_breaks.csv"
    _write_csv(paths["panel_breaks.csv"], find_breaks(panel_long, cfg),
               ["state", "variable", "financial_year", "break_type", "detail"])

    if reviews is not None:
        paths["human_reviews.csv"] = out_dir / "human_reviews.csv"
        _write_csv(paths["human_reviews.csv"], reviews, ["id", "created_at", "reviewer", "target_type", "target_id",
                                                        "decision", "value", "variable", "note", "context"])
    extra = {"llm_label_suggestions": paths["llm_label_suggestions.csv"]} if "llm_label_suggestions.csv" in paths else {}
    if "human_reviews.csv" in paths:
        extra["human_reviews"] = paths["human_reviews.csv"]
    extra["panel_breaks"] = paths["panel_breaks.csv"]
    return {**extra, "candidates": paths["candidates.json"], "extractions_json": paths["extractions.json"],
            "extractions_csv": paths["extractions.csv"], "review_queue": paths["review_queue.csv"],
            "panel": paths["panel.csv"], "panel_long": paths["panel_long.csv"], "summary": paths["summary.txt"]}
