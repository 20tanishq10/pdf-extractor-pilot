"""Create draft benchmark cases from processed documents, for a person to verify."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from statefin.benchmark.cases import BENCHMARK_DIR, BenchmarkCase, GoldRow, case_path, save_case
from statefin.pdf.tables import render_snapshots


def draft_cases(pipeline, doc_result, table_ids: Optional[Sequence[str]] = None, auto_negatives: int = 5,
                directory: Path = BENCHMARK_DIR / "cases", overwrite: bool = False) -> List[Path]:
    """Draft one case per table.

    Default selection: every table that yielded observations, plus up to `auto_negatives`
    year-indexed tables that yielded none (to measure false positives and abstention).
    Every value row is listed; rows the rules mapped get that variable, others null. The
    reviewer corrects variables and values, then sets status to "verified".
    """
    rec = doc_result.record
    models = doc_result.models
    with_obs = {o.table_id for o in doc_result.observations}
    if table_ids:
        chosen = list(table_ids)
    else:
        negatives = [t.table_id for t in doc_result.tables if t.table_id not in with_obs
                     and models.get(t.table_id) and models[t.table_id].orientation != "no_year_axis"
                     and any(vc.accepted for vc in models[t.table_id].value_cells)]
        chosen = [t.table_id for t in doc_result.tables if t.table_id in with_obs] + negatives[:auto_negatives]

    tables = {t.table_id: t for t in doc_result.tables}
    snap_dir = pipeline.registry.artifact(rec.document_id, "snapshots")
    missing = [tables[t] for t in chosen if t in tables and not (snap_dir / f"{t}.png").exists()]
    if missing:
        render_snapshots(pipeline.registry.pdf_path(rec), missing, snap_dir)

    written = []
    for tid in chosen:
        if tid not in tables:
            raise KeyError(f"{tid} is not a detected table of {rec.original_filename}")
        case_id = f"{rec.document_id}_{tid}"
        if case_path(case_id, directory).exists() and not overwrite:
            continue          # never overwrite a case someone may have verified
        table, model = tables[tid], models[tid]
        mapped = {(o.raw_label): o.canonical_variable for o in doc_result.observations if o.table_id == tid}
        rows = {}
        for vc in model.value_cells:
            if not vc.accepted or not vc.raw_label or vc.value is None:
                continue
            row = rows.setdefault(vc.raw_label, GoldRow(vc.raw_label, mapped.get(vc.raw_label)))
            row.values.setdefault(vc.fy, vc.value)
        case = BenchmarkCase(
            case_id=case_id, document_file=rec.original_filename, document_sha256=rec.sha256,
            page=table.page_number, table_id=tid, caption=table.caption,
            snapshot=str((snap_dir / f"{tid}.png").relative_to(pipeline.registry.data_dir.parent)),
            gold_unit=table.unit or model.inferred_unit, rows=list(rows.values()),
            prefilled_from="rule-based pipeline output (UNVERIFIED)",
        )
        written.append(save_case(case, directory))
    return written
