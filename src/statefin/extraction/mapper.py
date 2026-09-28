"""Map interpreted table cells to canonical variables and build observations.

A value becomes an observation only when all of the following hold:
  * its column is a year column of an accepted value type (actuals, or unqualified);
  * its row label, once normalised, equals a configured alias or fully matches a pattern
    of exactly one variable, and is not a ratio/percentage/growth label;
  * the unit of its row/table is compatible with the variable (or undeclared, which
    lowers confidence).
Everything that was considered and rejected is returned too, for candidates.json.

Confidence is computed from explicit, recorded reasons rather than a model score.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from statefin.config import Config, LabelRule, VariableSpec
from statefin.extraction.parsers import find_financial_years
from statefin.models import DocumentRecord, ExtractedTable, Observation, TableModel, ValueCell


@dataclass
class LabelMatch:
    variable: Optional[str]
    method: Optional[str] = None           # alias_exact | pattern
    confidence: float = 0.0
    note: Optional[str] = None
    normalized: str = ""
    rejected_reason: Optional[str] = None
    reason: Optional[str] = None           # model's justification, for LLM matches


# "Total (2071)", "2071-Pensions and Other Retirement Benefits", "MH 2049 Interest Payments" -
# but not a sub-head row whose label absorbed the heading ("2049 Interest Payments 05 Interest on ...").
_MAJOR_HEAD = re.compile(r"^\s*(?:total\s*\(?\s*)?(?:mh\s*)?(\d{4})\b(?!\s*[-–]\s*\d{2})", re.I)
_SUB_HEAD = re.compile(r"\s\d{2,3}\s+[A-Z]")


def match_label(raw_label: str, cfg: Config) -> LabelMatch:
    code = _MAJOR_HEAD.match(raw_label or "")
    if code and not _SUB_HEAD.search(raw_label[code.end():]):
        owners = [v for v in cfg.variables.values() if code.group(1) in v.major_heads]
        if len(owners) == 1:
            return LabelMatch(owners[0].name, method="major_head", confidence=float(cfg.get("confidence.major_head", 0.99)),
                              normalized=cfg.normalize_label(raw_label).text,
                              reason=f"major head {code.group(1)} (List of Major and Minor Heads)")
    nl = cfg.normalize_label(raw_label)
    if not nl.text:
        return LabelMatch(None, normalized="", rejected_reason="empty label")
    excluded = cfg.label_exclusion(nl.text)
    if excluded:
        return LabelMatch(None, normalized=nl.text, rejected_reason=f"ratio/percentage label ('{excluded}')")

    hits: List[Tuple[VariableSpec, LabelRule]] = []
    for var in cfg.variables.values():
        rule = var.aliases.get(nl.text)
        if rule:
            hits.append((var, rule))
    if not hits:
        for var in cfg.variables.values():
            for rule in var.patterns:
                if rule.regex is not None and rule.regex.fullmatch(nl.text):
                    hits.append((var, rule))
                    break
    hits = [(v, r) for v, r in hits if not any(p.search(nl.text) for p in v.label_excludes)]
    names = sorted({v.name for v, _ in hits})
    if not names:
        return LabelMatch(None, normalized=nl.text)
    if len(names) > 1:
        return LabelMatch(None, normalized=nl.text, rejected_reason=f"label matches several variables {names}")
    var, rule = hits[0]
    default = cfg.get("confidence.alias_exact", 0.98) if rule.kind == "alias" else cfg.get("confidence.pattern_match", 0.92)
    return LabelMatch(
        variable=var.name,
        method="alias_exact" if rule.kind == "alias" else "pattern",
        confidence=float(rule.confidence if rule.confidence is not None else default),
        note=rule.note,
        normalized=nl.text,
    )


def definition_profile(var: VariableSpec, cfg: Config, texts: Dict[str, Optional[str]]) -> Tuple[List[str], Optional[str]]:
    """(definition tags, series base year) from the texts around a value.

    texts keys: label, label_normalized, footnote, caption, section, column, cited_source."""
    tags = sorted({r.tag for r in var.definition_rules
                   if any(r.regex.search(texts.get(w) or "") for w in r.where)})
    for key in ("label", "column", "section", "caption", "footnote"):
        for rx in cfg.series_base_patterns:
            mm = rx.search(texts.get(key) or "")
            fys = find_financial_years(mm.group(1)) if mm else []
            if fys:
                return tags, fys[0][0]
    return tags, None


def confidence_tier(value: float, cfg: Config) -> str:
    tiers = cfg.get("confidence.tiers", {})
    if value >= tiers.get("exact", 0.95):
        return "exact"
    if value >= tiers.get("strong", 0.80):
        return "strong"
    if value >= tiers.get("ambiguous", 0.60):
        return "ambiguous"
    return "manual_review"


def observation_id(document_hash: str, table_id: str, vc: ValueCell, variable: str) -> str:
    key = f"{document_hash}|{table_id}|{vc.row_index}|{vc.anchor_id}|{vc.fy}|{variable}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass
class MappingResult:
    observations: List[Observation] = field(default_factory=list)
    rejected: List[Dict[str, str]] = field(default_factory=list)
    mapped_counts: Dict[str, int] = field(default_factory=dict)


def map_table(table: ExtractedTable, model: TableModel, doc: DocumentRecord, cfg: Config,
              extra_matches: Optional[Dict[str, LabelMatch]] = None, use_rules: bool = True) -> MappingResult:
    """Build observations for one table.

    extra_matches: label -> LabelMatch from another mapper (the LLM), used only for labels
    the rules leave unmapped and never for labels the rules reject (ratios, ambiguous).
    use_rules=False evaluates another mapper on its own (benchmarking); the ratio/unit
    guardrails still apply.
    """
    result = MappingResult()
    extra_matches = extra_matches or {}
    doc_spec = cfg.doc_type(doc.document_type)
    conf = cfg.pipeline.get("confidence", {})
    penalties = conf.get("parse_note_penalties", {})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    candidates: List[Tuple[ValueCell, LabelMatch]] = []
    seen_labels: Dict[str, LabelMatch] = {}
    for vc in model.value_cells:
        if not vc.raw_label:
            continue
        m = seen_labels.get(vc.raw_label)
        if m is None:
            m = match_label(vc.raw_label, cfg)
            if not use_rules and not m.rejected_reason:
                m = LabelMatch(None, normalized=m.normalized)
            if m.variable is None and not m.rejected_reason and vc.raw_label in extra_matches:
                m = extra_matches[vc.raw_label]
            seen_labels[vc.raw_label] = m
            if m.rejected_reason and m.rejected_reason != "empty label" and any(
                    p.search(m.normalized) for v in cfg.variables.values() for p in v.search_terms):
                result.rejected.append({"table_id": table.table_id, "raw_label": vc.raw_label,
                                        "reason": m.rejected_reason})
        if not m.variable or not vc.accepted or vc.value is None:
            continue
        var = cfg.variables[m.variable]
        context = f"{table.caption or ''} | {vc.section or ''}"
        ruled_out = next((p.pattern for p in var.context_excludes if p.search(context)), None)
        if ruled_out:
            result.rejected.append({"table_id": table.table_id, "raw_label": vc.raw_label,
                                    "reason": f"{m.variable} ruled out by table context '{ruled_out}'"})
            continue
        unit = vc.row_unit or table.unit or model.inferred_unit
        kind = cfg.unit_kind(unit)
        if var.unit_kind == "currency" and kind not in (None, "currency"):
            result.rejected.append({"table_id": table.table_id, "raw_label": vc.raw_label,
                                    "reason": f"unit '{unit}' incompatible with {m.variable}"})
            continue
        candidates.append((vc, m))

    # one value per (row, year): the sub-column the variable measures (revenue / total / ...)
    chosen: Dict[tuple, Tuple[ValueCell, LabelMatch]] = {}
    for vc, m in candidates:
        prefs = cfg.variables[m.variable].preferred_components + [None]
        rank = prefs.index(vc.component) if vc.component in prefs else None
        if rank is None:
            continue
        key = (vc.row_index, vc.fy, vc.value_type, m.variable)
        if key not in chosen or rank < prefs.index(chosen[key][0].component):
            chosen[key] = (vc, m)
    candidates = list(chosen.values())
    mapped_vars = {m.variable for _, m in candidates}
    table_rows = " || ".join(sorted({vc.raw_label for vc in model.value_cells if vc.raw_label}))
    for vc, m in candidates:
        var = cfg.variables[m.variable]
        nl = cfg.normalize_label(vc.raw_label)
        unit = vc.row_unit or table.unit
        c = m.confidence
        reasons = [f"label '{vc.raw_label}' -> {m.variable} by {m.method} ({m.confidence:.2f})"]
        if m.reason:
            reasons.append(f"model reason: {m.reason}")
        if unit is None:
            c *= conf.get("unit_missing", 0.70)
            reasons.append("unit not declared for table or row")
        elif vc.row_unit is None and table.unit_source == "inherited":
            c *= conf.get("unit_inherited", 0.98)
            reasons.append(f"unit inherited from {table.caption_inherited_from}")
        if vc.value_type is None:
            factor = doc_spec.unqualified_column_confidence
            if factor < 1:
                c *= factor
                reasons.append(f"year column has no Actual/BE/RE qualifier ({doc_spec.code} x{factor})")
        if not table.caption:
            c *= conf.get("caption_missing", 0.95)
            reasons.append("table has no caption")
        if len(mapped_vars) == 1:
            c *= conf.get("single_variable_table", 0.97)
            reasons.append("only one target variable in this table")
        if table.strategy != cfg.get("tables.primary_strategy", "lines"):
            c *= conf.get("text_strategy", 0.90)
            reasons.append(f"table found by fallback '{table.strategy}' strategy")
        for note in vc.parse_notes:
            if note in penalties:
                c *= penalties[note]
                reasons.append(f"value parse note: {note}")
        c = round(min(c, 1.0), 3)

        sign = None
        if var.canonical_sign_convention:
            sign = nl.sign_convention or "unspecified"
        footnote = table.footnotes.get(nl.footnote_marker) if nl.footnote_marker else None
        tags, base = definition_profile(var, cfg, {
            "label": vc.raw_label, "label_normalized": nl.text, "footnote": footnote, "caption": table.caption,
            "section": vc.section, "column": vc.column_header, "cited_source": table.cited_source,
            "table_rows": table_rows})
        if m.method == "major_head":
            tags = sorted(set(tags) | {f"major_head_{_MAJOR_HEAD.match(vc.raw_label).group(1)}"})
        variant = " ".join(t for t in tags if t in cfg.variant_tags(m.variable))

        result.observations.append(Observation(
            observation_id=observation_id(doc.sha256, table.table_id, vc, m.variable),
            state=doc.state,
            state_code=doc.state_code,
            financial_year=vc.fy,
            canonical_variable=m.variable,
            value=vc.value,
            unit=unit,
            value_type=vc.value_type or "unqualified",
            sign_convention=sign,
            raw_label=vc.raw_label,
            raw_value_text=vc.raw_text,
            column_header=vc.column_header,
            row_section=vc.section,
            label_footnote=f"[{nl.footnote_marker}] {footnote}" if footnote else None,
            source_document=doc.original_filename,
            document_id=doc.document_id,
            document_hash=doc.sha256,
            document_type=doc_spec.code,
            source_priority=doc_spec.priority,
            source_url=doc.source_url,
            document_download_date=doc.download_date,
            source_page=table.page_number,
            source_page_label=table.page_label,
            source_table=table.caption,
            table_id=table.table_id,
            table_bbox=table.bbox,
            table_cited_source=table.cited_source,
            source_text=vc.row_text,
            page_snapshot=None,
            cell_bbox=list(vc.bbox),
            definition_tags=tags,
            series_base=base,
            definition_variant=variant,
            component=vc.component,
            value_as_extracted=vc.value,
            extraction_method=f"table:{table.engine}:{table.strategy}:{model.orientation}",
            mapping_method=m.method or "",
            mapping_note=m.note,
            confidence=c,
            confidence_tier=confidence_tier(c, cfg),
            confidence_reasons=reasons,
            extracted_at=now,
        ))
        result.mapped_counts[m.variable] = result.mapped_counts.get(m.variable, 0) + 1
    return result
