"""Score a mapping system against benchmark cases.

Systems:
    deterministic  rules only (config/variables.yaml)
    llm_only       the LLM decides every row; guardrails (ratio/unit/provenance) still apply
    hybrid         rules first, LLM for rows the rules leave unmapped (the production mode)

Metrics:
    variable_mapping_accuracy   share of gold rows whose predicted variable (or NONE) is right
    target rows                 correct / wrong_variable / missed / abstained (UNKNOWN)
    non-target rows             correct_none / false_positive / abstained
    row identification          precision/recall of "this row is a target variable"
    observations                precision/recall/F1 on (variable, year, value)
    fy_accuracy                 predicted values for a correctly mapped row placed in the right year
    numeric_accuracy            predicted (variable, year) cells whose value equals gold
    unit_accuracy               table unit equals gold unit
    hallucination_rate          invalid row ids/labels returned by the model per row answered
    unsupported_values          predicted values whose text is not a cell of the table (must be 0)
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from statefin.benchmark.cases import BENCHMARK_DIR, BenchmarkCase
from statefin.config import Config
from statefin.extraction.mapper import map_table
from statefin.extraction.structure import analyze_table
from statefin.models import DocumentRecord, ExtractedTable
from statefin.registry import DocumentRegistry

VALUE_TOL = 0.005


def _safe_div(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b else None


def _load_table(registry: DocumentRegistry, case: BenchmarkCase):
    record = registry.load(case.document_id)
    path = registry.processed_dir / case.document_id / "tables.json"
    if record is None or not path.exists():
        raise FileNotFoundError(f"{case.case_id}: document {case.document_file} has not been processed "
                                f"into {registry.data_dir}; run extract.py --pdf on it first")
    with open(path, encoding="utf-8") as fh:
        for t in json.load(fh):
            if t["table_id"] == case.table_id:
                return record, ExtractedTable.from_dict(t)
    raise KeyError(f"{case.case_id}: table {case.table_id} not found (table detection may have changed)")


def evaluate(cases: List[BenchmarkCase], registry: DocumentRegistry, cfg: Config, system: str,
             semantic=None) -> Dict[str, Any]:
    if system != "deterministic" and semantic is None:
        raise ValueError(f"system '{system}' needs an LLM (semantic mapper)")
    tgt, non, rows_total, correct_rows = Counter(), Counter(), 0, 0
    pred_target = true_pred_target = 0
    obs_tp = obs_pred = obs_gold = 0
    fy_ok = fy_n = num_ok = num_n = unit_ok = 0
    halluc = answered = unsupported = 0
    calls: List[Dict[str, Any]] = []
    per_case = []

    for case in cases:
        record, table = _load_table(registry, case)
        model = analyze_table(table, cfg)
        interp = None
        if system == "deterministic":
            result = map_table(table, model, record, cfg)
        else:
            interp = semantic.interpret(table, model)
            calls.extend(interp.calls)
            halluc += interp.hallucinations
            answered += len(interp.final)
            matches = semantic.label_matches(interp)
            result = map_table(table, model, record, cfg, extra_matches=matches, use_rules=(system == "hybrid"))

        predicted_var = {}
        for o in result.observations:
            predicted_var.setdefault(o.raw_label, o.canonical_variable)
        abstained = {label for label, d in (interp.final.items() if interp else [])
                     if d.status in ("unknown", "invalid") and label not in predicted_var}
        cell_texts = {c.text.replace("\n", " ").strip() for c in table.cells}

        case_correct = 0
        for g in case.rows:
            rows_total += 1
            pv = predicted_var.get(g.label)
            if g.variable:
                outcome = ("correct" if pv == g.variable else "wrong_variable" if pv
                           else "abstained" if g.label in abstained else "missed")
                tgt[outcome] += 1
            else:
                outcome = ("false_positive" if pv else "abstained" if g.label in abstained else "correct_none")
                non[outcome] += 1
            if outcome in ("correct", "correct_none"):
                correct_rows += 1
                case_correct += 1
            if pv:
                pred_target += 1
                true_pred_target += bool(g.variable)

        gold = {(g.variable, fy, round(v, 2)) for g in case.rows if g.variable for fy, v in g.values.items()}
        gold_by_cell = {(g.label, g.variable, fy): v for g in case.rows if g.variable for fy, v in g.values.items()}
        gold_rows = {g.label: g for g in case.rows}
        pred = set()
        for o in result.observations:
            pred.add((o.canonical_variable, o.financial_year, round(o.value, 2)))
            if o.raw_value_text.strip() not in cell_texts and not any(o.raw_value_text in t for t in cell_texts):
                unsupported += 1
            g = gold_rows.get(o.raw_label)
            if g and g.variable == o.canonical_variable:
                if any(abs(v - o.value) <= VALUE_TOL for v in g.values.values()):
                    fy_n += 1
                    fy_ok += abs(g.values.get(o.financial_year, float("nan")) - o.value) <= VALUE_TOL
            key = (o.raw_label, o.canonical_variable, o.financial_year)
            if key in gold_by_cell:
                num_n += 1
                num_ok += abs(gold_by_cell[key] - o.value) <= VALUE_TOL
        tp = len(pred & gold)
        obs_tp, obs_pred, obs_gold = obs_tp + tp, obs_pred + len(pred), obs_gold + len(gold)
        pred_unit = table.unit or model.inferred_unit
        unit_ok += pred_unit == case.gold_unit
        per_case.append({"case_id": case.case_id, "rows": len(case.rows), "rows_correct": case_correct,
                         "obs_tp": tp, "obs_pred": len(pred), "obs_gold": len(gold),
                         "unit_ok": pred_unit == case.gold_unit, "status": case.status})

    precision, recall = _safe_div(obs_tp, obs_pred), _safe_div(obs_tp, obs_gold)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else None
    tokens = Counter()
    for c in calls:
        for k, v in (c.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                tokens[k] += v
    return {
        "system": system,
        "model": None if semantic is None else {
            "primary": semantic.primary.model_id,
            "fallback": semantic.fallback.model_id if semantic.fallback else None},
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": len(cases),
        "unverified_cases": sum(c.status != "verified" for c in cases),
        "rows": rows_total,
        "variable_mapping_accuracy": _safe_div(correct_rows, rows_total),
        "target_rows": dict(tgt),
        "non_target_rows": dict(non),
        "row_identification": {"precision": _safe_div(true_pred_target, pred_target),
                               "recall": _safe_div(tgt["correct"] + tgt["wrong_variable"], sum(tgt.values()))},
        "observations": {"precision": precision, "recall": recall, "f1": f1,
                         "true_positive": obs_tp, "predicted": obs_pred, "gold": obs_gold},
        "fy_accuracy": _safe_div(fy_ok, fy_n),
        "numeric_accuracy": _safe_div(num_ok, num_n),
        "unit_accuracy": _safe_div(unit_ok, len(cases)),
        "hallucination_rate": _safe_div(halluc, answered) if system != "deterministic" else None,
        "unsupported_values": unsupported,
        "abstention_rate_on_targets": _safe_div(tgt["abstained"], sum(tgt.values())),
        "llm_calls": len(calls),
        "llm_calls_cached": sum(bool(c.get("cached")) for c in calls),
        "llm_escalations": sum(1 for c in calls if c.get("model") and semantic is not None and semantic.fallback
                               and c["model"] == semantic.fallback.model_id),
        "llm_errors": [c["error"] for c in calls if "error" in c],
        "tokens": dict(tokens),
        "per_case": per_case,
    }


def format_report(r: Dict[str, Any]) -> str:
    lines = [f"System: {r['system']}" + (f"  ({r['model']['primary']} -> {r['model']['fallback']})" if r["model"] else "")]
    if r["unverified_cases"]:
        lines.append(f"WARNING: {r['unverified_cases']}/{r['cases']} case(s) are UNVERIFIED drafts pre-filled by the "
                     "rules; these numbers are not evidence (rule scores are circular).")
    lines += [
        f"Cases {r['cases']}, rows {r['rows']}",
        f"  variable mapping accuracy   {r['variable_mapping_accuracy']}",
        f"  target rows                 {r['target_rows']}",
        f"  non-target rows             {r['non_target_rows']}",
        f"  row identification P/R      {r['row_identification']['precision']} / {r['row_identification']['recall']}",
        f"  observations P/R/F1         {r['observations']['precision']} / {r['observations']['recall']} / {r['observations']['f1']}"
        f"   ({r['observations']['true_positive']} of {r['observations']['gold']} gold)",
        f"  financial-year accuracy     {r['fy_accuracy']}",
        f"  numeric-value accuracy      {r['numeric_accuracy']}",
        f"  unit accuracy               {r['unit_accuracy']}",
        f"  hallucination rate          {r['hallucination_rate']}",
        f"  unsupported values          {r['unsupported_values']}",
        f"  abstention on targets       {r['abstention_rate_on_targets']}",
    ]
    if r["llm_calls"]:
        lines.append(f"  LLM calls {r['llm_calls']} ({r['llm_calls_cached']} cached, {r['llm_escalations']} to fallback), "
                     f"tokens {r['tokens']}")
    if r["llm_errors"]:
        lines.append(f"  LLM errors: {r['llm_errors'][:3]}")
    return "\n".join(lines)


def save_report(report: Dict[str, Any], directory: Path = BENCHMARK_DIR / "results") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report["evaluated_at"].replace(":", "").replace("-", "")[:15]
    tag = report["system"] + ("_" + report["model"]["primary"].replace("/", "-") if report["model"] else "")
    path = directory / f"{stamp}_{tag}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return path
