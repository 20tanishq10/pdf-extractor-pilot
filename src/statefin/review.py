"""Apply human review decisions (from the SQLite ledger) to a run, and build the review queue.

Decisions are keyed to stable identifiers, so they survive reruns:
    observation  <observation_id>              accepted | rejected | corrected (value)
    row_label    <document_id>|<table_id>|<label>  mapped (variable) | none
    missing      <state>|<variable>            acknowledged

Effects, applied on every run/rebuild:
    accepted   the observation is final; it enters the panel even with warning/error flags
    rejected   the observation is kept in the raw layer but takes no part in reconciliation,
               checks or the panel
    corrected  `value` becomes the reviewer's value; `value_as_extracted` and `raw_value_text`
               keep what the PDF says; an info flag records who changed what and why
    mapped     the row's cells become observations with mapping_method "human:<reviewer>";
               values still come from the parsed cells
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from statefin.config import Config
from statefin.extraction.mapper import LabelMatch
from statefin.models import Observation

HUMAN_STATUS = {"accepted": "HUMAN_ACCEPTED", "corrected": "HUMAN_CORRECTED", "rejected": "HUMAN_REJECTED"}


@dataclass
class ReviewBook:
    decisions: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_store(cls, store) -> "ReviewBook":
        return cls(store.latest_reviews() if store is not None else {})

    def get(self, target_type: str, target_id: str) -> Optional[Dict[str, Any]]:
        return self.decisions.get((target_type, target_id))

    def label_matches(self, document_id: str, table_id: str, cfg: Config) -> Dict[str, LabelMatch]:
        prefix = f"{document_id}|{table_id}|"
        out = {}
        for (ttype, tid), r in self.decisions.items():
            if ttype == "row_label" and tid.startswith(prefix) and r["decision"] == "mapped":
                label = tid[len(prefix):]
                out[label] = LabelMatch(r["variable"], method=f"human:{r['reviewer']}", confidence=1.0,
                                        normalized=cfg.normalize_label(label).text,
                                        reason=r.get("note") or "mapped by reviewer")
        return out


def row_target_id(document_id: str, table_id: str, label: str) -> str:
    return f"{document_id}|{table_id}|{label}"


def apply_observation_reviews(observations: List[Observation], book: ReviewBook, cfg: Config) -> int:
    applied = 0
    for o in observations:
        if o.value_as_extracted is None:
            o.value_as_extracted = o.value
        if o.mapping_method.startswith("human:") and o.review_decision is None:
            o.review_decision, o.reviewed_by = "accepted", o.mapping_method.split(":", 1)[1]
        r = book.get("observation", o.observation_id)
        if not r:
            continue
        applied += 1
        o.review_decision = r["decision"]
        o.review_note, o.reviewed_by, o.reviewed_at = r.get("note") or None, r.get("reviewer"), r.get("created_at")
        who = f"{r.get('reviewer')} at {r.get('created_at')}"
        if r["decision"] == "corrected":
            factor = cfg.crore_factor(o.unit) or 1.0
            o.value = round(float(r["value"]) / factor, 6)      # reviewers enter INR crore
            spec = cfg.variables.get(o.canonical_variable)
            if spec and spec.canonical_sign_convention:
                o.sign_convention = spec.canonical_sign_convention    # reviewers enter canonical values
            o.add_flag("human_corrected", "info",
                       f"extracted {o.value_as_extracted:,.2f} corrected to {o.value:,.2f} {o.unit or ''} by {who}"
                       + (f": {r['note']}" if r.get("note") else ""))
        else:
            o.add_flag(f"human_{r['decision']}", "info", f"{r['decision']} by {who}"
                       + (f": {r['note']}" if r.get("note") else ""))
    return applied


def finalize_reviewed(observations: List[Observation]) -> None:
    """Human decisions override the automatic review status (after validation.finalize)."""
    for o in observations:
        if o.review_decision in HUMAN_STATUS:
            o.review_status = HUMAN_STATUS[o.review_decision]


# ---------------------------------------------------------------------- review queue

def _severity(o: Observation) -> str:
    sev = {f["severity"] for f in o.validation_flags}
    return "error" if "error" in sev else "warning" if "warning" in sev else "low_confidence"


def build_review_items(docs, observations: List[Observation], cells, cfg: Config) -> List[Dict[str, Any]]:
    """Everything a person may need to decide, most severe first. Decided items stay listed
    (the dashboard shows them under "decided"), so a decision can be revisited."""
    items: List[Dict[str, Any]] = []
    for o in observations:
        if o.review_status != "NEEDS_REVIEW" and not o.review_decision:
            continue
        kind = "conflict" if o.reconciliation_status == "conflict" else "observation"
        items.append({
            "item_id": f"obs:{o.observation_id}", "item_type": kind,
            "target_type": "observation", "target_id": o.observation_id,
            "severity": _severity(o) if o.review_status == "NEEDS_REVIEW" else "decided",
            "state": o.state, "financial_year": o.financial_year, "variable": o.canonical_variable,
            "value": o.value, "unit": o.unit, "raw_label": o.raw_label,
            "source_document": o.source_document, "source_page": o.source_page, "source_table": o.source_table,
            "confidence": o.confidence, "reasons": o.review_reasons,
            "checks": sorted({f["check"] for f in o.validation_flags if f["severity"] in ("warning", "error")}),
            "group": f"{o.state}|{o.financial_year}|{o.canonical_variable}",
            "review_status": o.review_status,
        })
    for d in docs:
        for u in d.llm_unresolved:
            tid = row_target_id(d.record.document_id, u["table_id"], u["label"])
            items.append({
                "item_id": f"row:{tid}", "item_type": "row_label", "target_type": "row_label", "target_id": tid,
                "severity": "warning", "state": d.record.state, "financial_year": None, "variable": None,
                "raw_label": u["label"], "source_document": d.record.original_filename, "source_page": u["page"],
                "source_table": u["caption"], "document_id": d.record.document_id, "table_id": u["table_id"],
                "reasons": [f"{u['model']}: {u['reason']}"], "checks": ["llm_unresolved"],
            })
    present = {(k[0], k[2]) for k in cells}
    states = sorted({o.state or "UNKNOWN" for o in observations})
    for state in states:
        for var in cfg.core_variables:
            spec = cfg.variables[var]
            derivable = spec.derive_from and all((state, p) in present for p in spec.derive_from)
            if (state, var) not in present and not derivable:
                items.append({
                    "item_id": f"missing:{state}|{var}", "item_type": "missing", "target_type": "missing",
                    "target_id": f"{state}|{var}", "severity": "warning", "state": state, "variable": var,
                    "financial_year": None, "reasons": ["no observation extracted from any processed document"],
                    "checks": ["missing_variable"],
                })
    order = {"error": 0, "warning": 1, "low_confidence": 2, "decided": 3}
    items.sort(key=lambda i: (order.get(i["severity"], 4), i["item_type"] == "missing", str(i.get("variable")),
                              str(i.get("financial_year")), str(i.get("source_page"))))
    return items
