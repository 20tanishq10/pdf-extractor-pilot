"""Concise, human-readable run summary (printed to the terminal and saved as summary.txt)."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from statefin.config import Config
from statefin.models import Observation
from statefin.reconciliation.sources import CellConsensus, CellKey


def _years(fys: List[str]) -> str:
    fys = sorted(set(fys))
    if not fys:
        return ""
    return fys[0] if len(fys) == 1 else f"{fys[0]} … {fys[-1]} ({len(fys)} yrs)"


def render_summary(docs, observations: List[Observation], cells: Dict[CellKey, CellConsensus],
                   panel_long: List[dict], outputs: Dict[str, Path], cfg: Config) -> str:
    out: List[str] = []
    for d in docs:
        r = d.record
        label = cfg.doc_type(r.document_type).label
        out.append(f"Document: {r.state or 'UNKNOWN STATE'} — {label}{' ' + r.report_year if r.report_year else ''}")
        out.append(f"  file      {r.original_filename}")
        reused = f"   [reused cached: {', '.join(d.reused_stages)}]" if d.reused_stages else ""
        out.append(f"  id        {r.document_id}  (state: {r.state_source}, type: {r.document_type_source}){reused}")
        if d.error:
            out.append(f"  FAILED    {d.error}")
            out.append("")
            continue
        ocr = [p.page_number for p in d.pages if p.needs_ocr]
        blank = [p.page_number for p in d.pages if p.char_count == 0 and not p.needs_ocr]
        year_tables = [t for t in d.tables if d.models.get(t.table_id) and d.models[t.table_id].orientation != "no_year_axis"]
        used = {o.table_id for o in d.observations}
        out.append(f"  Pages: {len(d.pages)}   (need OCR: {len(ocr)}, blank: {len(blank)})")
        out.append(f"  Candidate pages: {sum(c.is_candidate for c in d.candidates)}")
        out.append(f"  Tables: {len(d.tables)} detected, {len(year_tables)} year-indexed, {len(used)} yielded values")
        out.append(f"  Observations: {len(d.observations)}")
        ok = sum(v.get("ok", 0) for v in d.arithmetic.values())
        bad = sum(v.get("mismatch", 0) for v in d.arithmetic.values())
        if ok or bad:
            out.append(f"  Table arithmetic: {ok} printed formula(s) hold, {bad} do not")
        if d.interpretations or d.llm_error:
            calls = [c for i in d.interpretations.values() for c in i.calls]
            llm_obs = sum(o.mapping_method.startswith("llm:") for o in d.observations)
            disagree = sum(any(f["check"] == "llm_disagrees" for f in o.validation_flags) for o in d.observations)
            out.append(f"  LLM: {len(d.interpretations)} table(s), {len(calls)} call(s) "
                       f"({sum(c.get('cached', False) for c in calls)} cached, "
                       f"{sum(i.escalated for i in d.interpretations.values())} escalated), "
                       f"{llm_obs} new observation(s), {disagree} disagreement(s) with rules, "
                       f"{len(d.llm_unresolved)} unresolved row(s), "
                       f"{sum(i.hallucinations for i in d.interpretations.values())} hallucinated reference(s)")
            if d.llm_error:
                out.append(f"  LLM ERROR (deterministic results unaffected): {d.llm_error[:220]}")
        out.append("")

    by_state: Dict[str, List[Observation]] = defaultdict(list)
    for o in observations:
        by_state[o.state or "UNKNOWN"].append(o)
    panel_by = defaultdict(list)
    for row in panel_long:
        if row["panel_status"] in ("included", "derived"):
            panel_by[(row["state"], row["variable"])].append(row)

    for state, obs in sorted(by_state.items()):
        out.append(f"Extracted — {state}")
        out.append(f"  {'variable':<26}   {'obs':>4}  {'years':<28} {'panel':>5}  {'review':>6}")
        missing = []
        for var in cfg.core_variables + [v for v, s in cfg.variables.items() if s.role == "optional"]:
            vo = [o for o in obs if o.canonical_variable == var]
            prow = panel_by.get((state, var), [])
            review = sum(o.review_status == "NEEDS_REVIEW" for o in vo)
            if prow and not vo:
                mark, years = "~", f"derived ({' + '.join(cfg.variables[var].derive_from)})"
            elif prow:
                mark, years = "✓", _years([o.financial_year for o in vo])
            elif vo:
                mark, years = "!", _years([o.financial_year for o in vo]) + " (none in panel)"
            else:
                mark, years = "✗", "not found"
                if cfg.variables[var].role == "core":
                    missing.append(var)
            role = "" if cfg.variables[var].role == "core" else " (optional)"
            out.append(f"  {var + role:<26} {mark} {len(vo):>4}  {years:<28} {len(prow):>5}  {review:>6}")
        out.append("")
        flagged = Counter(f["check"] for o in obs if o.review_status == "NEEDS_REVIEW"
                          for f in o.validation_flags if f["severity"] in ("warning", "error"))
        low_conf = sum(1 for o in obs if o.review_status == "NEEDS_REVIEW" and not any(
            f["severity"] in ("warning", "error") for f in o.validation_flags))
        n_review = sum(o.review_status == "NEEDS_REVIEW" for o in obs)
        out.append(f"Needs review: {n_review} observation(s)" + (f", {len(missing)} missing core variable(s)" if missing else ""))
        for check, n in flagged.most_common():
            out.append(f"  {check:<32} {n}")
        if low_conf:
            out.append(f"  {'low confidence only':<32} {low_conf}")
        for var in missing:
            out.append(f"  missing: {var}")
        identity = [f for c in cells.values() if c.state == state and c.representative
                    for f in c.representative.validation_flags if f["check"] == "identity_revenue_balance"]
        if identity:
            ok = sum(f["severity"] == "info" for f in identity)
            out.append(f"Revenue identity (RR - RE = RD): holds in {ok}/{len(identity)} year(s)")
        out.append("")

    out.append("Outputs:")
    decided = sum(1 for o in observations if o.review_decision)
    if decided:
        out.append(f"Reviewer decisions applied: {decided} observation(s)")
        out.append("")
    for key in ("extractions_csv", "review_queue", "panel", "panel_long", "extractions_json", "candidates",
                "llm_label_suggestions", "human_reviews"):
        if key in outputs:
            out.append(f"  {outputs[key]}")
    return "\n".join(out)
