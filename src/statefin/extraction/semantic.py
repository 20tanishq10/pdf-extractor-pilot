"""Semantic row mapping with an LLM: one already-extracted table at a time.

Division of labour:
    Python   finds the table, its year columns, every row label and every cell value.
    LLM      answers only "which canonical variable does this row label denote?"
             (a variable name, NONE, or UNKNOWN), with a confidence and a short reason.
    Python   takes the numbers from the parsed cells, and applies the same unit, ratio,
             provenance and validation checks as for rule-based mappings.

The output schema has no field for values, so a model cannot introduce a number. Its
answers are checked against the table: an unknown row id, or an echoed label that is not
the row's label, counts as a hallucination and is discarded.

Two stages: the primary (small) model answers first; the fallback (larger) model is
called only if the primary abstains, is unsure, or answers invalidly. If the two still
disagree, the row stays UNKNOWN and goes to human review.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from statefin.config import Config
from statefin.extraction.mapper import LabelMatch, match_label
from statefin.llm.base import LLMConfigurationError, LLMError, LLMProvider
from statefin.llm.settings import LLMSettings
from statefin.models import ExtractedTable, Observation, TableModel

PROMPT_VERSION = "map-rows-v1"

INSTRUCTIONS = """You classify the rows of one table taken from an official Indian state-government finance \
document (CAG Finance Accounts, CAG State Finances Audit Report, State budget, or Finance Commission study).

For every row, decide which canonical variable its LABEL denotes, using the variable definitions supplied.

Rules:
- Answer only from the table supplied. Do not use outside knowledge about the State's finances.
- Never output numbers. The sample values are context only; you classify labels, you do not extract values.
- Map a row only if its label denotes the same accounting concept as the variable definition, for the \
State as a whole. A sub-component (e.g. only "Wages", a single department, "Plan" part), a total of a \
different aggregate, a share, ratio, percentage, growth rate, per-capita figure, projection or target is NONE.
- Answer NONE when the row is clearly not one of the variables.
- Answer UNKNOWN when the label could plausibly be a variable but the table does not let you decide. \
UNKNOWN is always better than a wrong answer.
- Copy each row's label exactly as given, and return exactly one entry per row_id.
- confidence is your probability (0 to 1) that canonical_variable is correct.
- Keep reason to one short sentence."""


@dataclass
class RowItem:
    row_id: str
    label: str
    section: Optional[str]
    unit: Optional[str]
    samples: List[str]


@dataclass
class RowDecision:
    row_id: str
    label: str
    status: str                        # mapped | none | unknown | invalid
    variable: Optional[str]
    confidence: float
    reason: str
    model: str


@dataclass
class TableInterpretation:
    table_id: str
    items: List[RowItem]
    final: Dict[str, RowDecision] = field(default_factory=dict)            # by label
    primary: Dict[str, RowDecision] = field(default_factory=dict)
    fallback: Dict[str, RowDecision] = field(default_factory=dict)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None
    hallucinations: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", label or "").strip().lower()


class SemanticMapper:
    def __init__(self, cfg: Config, settings: LLMSettings, primary: LLMProvider, fallback: Optional[LLMProvider]):
        self.cfg = cfg
        self.settings = settings
        self.primary = primary
        self.fallback = fallback
        self.variables = {n: v for n, v in cfg.variables.items() if v.unit_kind == "currency"}
        self.search_terms = [p for v in cfg.variables.values() for p in v.search_terms]

    # ------------------------------------------------------------------ selection
    def is_relevant(self, label: str) -> bool:
        text = self.cfg.normalize_label(label).text
        return any(p.search(text) for p in self.search_terms)

    def select_tables(self, tables: Sequence[ExtractedTable], models: Dict[str, TableModel]) -> List[str]:
        chosen = []
        for t in tables:
            m = models.get(t.table_id)
            if not m or m.orientation == "no_year_axis" or not any(vc.accepted for vc in m.value_cells):
                continue
            if self.settings.table_selection == "all_year_tables":
                chosen.append(t.table_id)
                continue
            labels = {vc.raw_label for vc in m.value_cells if vc.accepted and vc.raw_label}
            for label in labels:
                lm = match_label(label, self.cfg)
                if lm.variable is None and not lm.rejected_reason and self.is_relevant(label):
                    chosen.append(t.table_id)
                    break
        return chosen[: self.settings.max_tables_per_document]

    # ------------------------------------------------------------------ request
    def build_items(self, table: ExtractedTable, model: TableModel) -> List[RowItem]:
        items: Dict[str, RowItem] = {}
        for vc in model.value_cells:
            if not vc.accepted or not vc.raw_label:
                continue
            item = items.get(vc.raw_label)
            if item is None:
                item = items[vc.raw_label] = RowItem(f"R{len(items) + 1}", vc.raw_label, vc.section,
                                                     vc.row_unit or table.unit or model.inferred_unit, [])
            if len(item.samples) < 3:
                item.samples.append(f"{vc.fy}: {vc.raw_text}")
        return list(items.values())[: self.settings.max_rows_per_table]

    def build_request(self, table: ExtractedTable, model: TableModel, items: List[RowItem]) -> Dict[str, Any]:
        years = sorted({a.fy for a in model.anchors if a.accepted})
        return {
            "table": {
                "caption": table.caption, "unit": table.unit or model.inferred_unit,
                "cited_source": table.cited_source, "orientation": model.orientation, "years": years,
                "rows": [{"row_id": it.row_id, "label": it.label, "section": it.section, "unit": it.unit,
                          "sample_values": it.samples} for it in items],
            },
            "canonical_variables": [{"name": n, "definition": v.description} for n, v in self.variables.items()],
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "row_id": {"type": "string"},
                            "label": {"type": "string"},
                            "canonical_variable": {"type": "string",
                                                   "enum": sorted(self.variables) + ["NONE", "UNKNOWN"]},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["row_id", "label", "canonical_variable", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["rows"],
            "additionalProperties": False,
        }

    # ------------------------------------------------------------------ one model call
    def _ask(self, provider: LLMProvider, request: Dict[str, Any], items: List[RowItem],
             interp: TableInterpretation) -> Dict[str, RowDecision]:
        by_id = {it.row_id: it for it in items}
        call: Dict[str, Any] = {"model": provider.model_id, "prompt_version": PROMPT_VERSION}
        decisions: Dict[str, RowDecision] = {}
        try:
            res = provider.structured_extract(request, self.schema(), INSTRUCTIONS)
            call.update(cached=res.cached, latency_s=res.latency_s, usage=res.usage)
        except LLMConfigurationError:
            raise   # missing key, no credits, unknown model: stop, do not retry per table
        except LLMError as exc:
            call["error"] = str(exc)
            interp.calls.append(call)
            return {}
        issues = []
        for row in res.data.get("rows", []) or []:
            item = by_id.get(str(row.get("row_id")))
            if item is None:
                issues.append(f"unknown row_id {row.get('row_id')!r}")
                continue
            var = row.get("canonical_variable")
            try:
                conf = max(0.0, min(1.0, float(row.get("confidence", 0))))
            except (TypeError, ValueError):
                conf = 0.0
            reason = str(row.get("reason", ""))[:300]
            if _norm(row.get("label", "")) != _norm(item.label):
                issues.append(f"{item.row_id}: echoed label {row.get('label')!r} is not the table's label")
                decisions[item.label] = RowDecision(item.row_id, item.label, "invalid", None, 0.0, "label mismatch",
                                                    provider.model_id)
            elif var == "NONE":
                decisions[item.label] = RowDecision(item.row_id, item.label, "none", None, conf, reason, provider.model_id)
            elif var in self.variables:
                decisions[item.label] = RowDecision(item.row_id, item.label, "mapped", var, conf, reason, provider.model_id)
            elif var == "UNKNOWN":
                decisions[item.label] = RowDecision(item.row_id, item.label, "unknown", None, conf, reason, provider.model_id)
            else:
                issues.append(f"{item.row_id}: variable {var!r} is not allowed")
                decisions[item.label] = RowDecision(item.row_id, item.label, "invalid", None, 0.0, "invalid variable",
                                                    provider.model_id)
        for item in items:
            if item.label not in decisions:
                decisions[item.label] = RowDecision(item.row_id, item.label, "unknown", None, 0.0,
                                                    "no answer returned for this row", provider.model_id)
        interp.hallucinations += len(issues)
        call["issues"] = issues
        interp.calls.append(call)
        return decisions

    def _needs_escalation(self, decisions: Dict[str, RowDecision], interp: TableInterpretation) -> Optional[str]:
        if not decisions:
            return "primary model failed"
        if interp.hallucinations:
            return "primary answer referenced rows or labels not in the table"
        for d in decisions.values():
            if d.status == "invalid":
                return f"invalid answer for '{d.label}'"
            if d.status == "unknown" and self.is_relevant(d.label):
                return f"primary abstained on '{d.label}'"
            if d.status == "mapped" and d.confidence < self.settings.escalate_below_confidence:
                return f"low confidence ({d.confidence:.2f}) for '{d.label}'"
        return None

    # ------------------------------------------------------------------ public
    def interpret(self, table: ExtractedTable, model: TableModel) -> TableInterpretation:
        items = self.build_items(table, model)
        interp = TableInterpretation(table.table_id, items)
        if not items:
            return interp
        request = self.build_request(table, model, items)
        interp.primary = self._ask(self.primary, request, items, interp)
        interp.final = dict(interp.primary)
        reason = self._needs_escalation(interp.primary, interp)
        if reason and self.fallback is not None:
            interp.escalated, interp.escalation_reason = True, reason
            interp.fallback = self._ask(self.fallback, request, items, interp)
            for label, f in interp.fallback.items():
                p = interp.primary.get(label)
                if f.status in ("unknown", "invalid"):
                    interp.final[label] = p if p and p.status in ("mapped", "none") and p.confidence >= \
                        self.settings.escalate_below_confidence else f
                elif p and p.status in ("mapped", "none") and p.variable != f.variable:
                    interp.final[label] = RowDecision(
                        f.row_id, label, "unknown", None, 0.0,
                        f"models disagree: {p.model} says {p.variable or 'NONE'}, {f.model} says {f.variable or 'NONE'}",
                        f"{p.model}+{f.model}")
                else:
                    interp.final[label] = f
        elif reason:
            interp.escalation_reason = f"{reason} (no fallback configured)"
        if not interp.final and interp.calls and all("error" in c for c in interp.calls):
            interp.error = "; ".join(c["error"] for c in interp.calls)
        return interp

    def label_matches(self, interp: TableInterpretation) -> Dict[str, LabelMatch]:
        """Final 'mapped' decisions as LabelMatch objects for extraction/mapper.map_table."""
        out = {}
        for label, d in interp.final.items():
            if d.status != "mapped":
                continue
            if self.settings.accept_without_review:
                conf = d.confidence * self.settings.accepted_confidence_factor
            else:
                conf = min(d.confidence, self.settings.confidence_cap)
            out[label] = LabelMatch(d.variable, method=f"llm:{d.model}", confidence=round(conf, 3),
                                    normalized=self.cfg.normalize_label(label).text, reason=d.reason)
        return out

    def unresolved(self, interp: TableInterpretation) -> List[RowDecision]:
        """Rows still ambiguous after both stages whose labels use target vocabulary -> human review."""
        return [d for d in interp.final.values()
                if d.status in ("unknown", "invalid") and self.is_relevant(d.label)
                and match_label(d.label, self.cfg).variable is None]

    def compare_with_rules(self, interp: TableInterpretation, observations: List[Observation]) -> None:
        """Record agreement, and flag disagreement, between the LLM and rule-based mappings."""
        for o in observations:
            if o.table_id != interp.table_id or o.mapping_method.startswith("llm:"):
                continue
            d = interp.final.get(o.raw_label)
            if d is None:
                continue
            if d.status == "mapped" and d.variable == o.canonical_variable:
                o.confidence_reasons.append(f"LLM {d.model} agrees ({d.confidence:.2f})")
            elif d.status in ("mapped", "none"):
                o.add_flag("llm_disagrees", "warning",
                           f"rules map '{o.raw_label}' to {o.canonical_variable}; {d.model} says "
                           f"{d.variable or 'NONE'} ({d.confidence:.2f}): {d.reason}")
