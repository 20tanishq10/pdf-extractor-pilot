"""Orchestration: local PDF(s) -> validated observations -> outputs.

Per document (cached, resumable):
    DOWNLOADED -> TEXT_EXTRACTED -> CANDIDATES_FOUND -> TABLES_EXTRACTED -> VALUES_EXTRACTED
    -> [SEMANTIC_MAPPED with --llm] -> reviewer row mappings
Per run (recomputed from cached observations every time, so cross-document checks see
every document, and human decisions made meanwhile are applied):
    human decisions -> per-observation checks -> reconciliation -> identities / time series
    -> review status -> panel -> outputs

With a Store, progress is reported live and results are published after every document
(interim) and at the end (final), which is what the dashboard shows.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from statefin.config import Config, code_fingerprint, stable_hash
from statefin.export.outputs import write_outputs
from statefin.export.panel import build_panel
from statefin.export.summary import render_summary
from statefin.extraction.candidates import rank_pages
from statefin.extraction.mapper import map_table
from statefin.extraction.metadata import detect_document_type, detect_report_year, detect_state
from statefin.extraction.semantic import PROMPT_VERSION, SemanticMapper, TableInterpretation
from statefin.extraction.structure import analyze_table
from statefin.llm.base import LLMConfigurationError
from statefin.models import (PIPELINE_VERSION, CandidatePage, DocumentRecord, ExtractedTable, Observation,
                             PageText, Stage, TableModel)
from statefin.pdf.tables import extract_tables, render_snapshots
from statefin.pdf.text import extract_page_texts, load_pages, pdf_dates, save_pages
from statefin.progress import Reporter, StoreReporter
from statefin.reconciliation.sources import CellConsensus, CellKey, reconcile
from statefin.registry import DocumentRegistry
from statefin.review import ReviewBook, apply_observation_reviews, build_review_items, finalize_reviewed
from statefin.validation import finalize
from statefin.validation.accounting import check_identities
from statefin.validation.arithmetic import apply_arithmetic
from statefin.validation.anomalies import (check_observation, check_panel_eligibility, check_provenance,
                                           check_time_series, check_unit_consistency)


@dataclass
class DocumentResult:
    record: DocumentRecord
    pages: List[PageText] = field(default_factory=list)
    candidates: List[CandidatePage] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    models: Dict[str, TableModel] = field(default_factory=dict)
    observations: List[Observation] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    reused_stages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    interpretations: Dict[str, TableInterpretation] = field(default_factory=dict)
    llm_unresolved: List[dict] = field(default_factory=list)
    llm_error: Optional[str] = None
    arithmetic: Dict[str, Dict[str, int]] = field(default_factory=dict)   # table_id -> {ok, mismatch}


@dataclass
class RunResult:
    documents: List[DocumentResult]
    observations: List[Observation]
    cells: Dict[CellKey, CellConsensus]
    panel_long: List[dict]
    panel_wide: List[dict]
    outputs: Dict[str, Path]
    summary: str
    run_id: Optional[str] = None
    review_items: List[dict] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(d.error for d in self.documents)


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    tmp.replace(path)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Pipeline:
    def __init__(self, cfg: Config, data_dir: Path, force: bool = False, snapshots: bool = True,
                 log: Callable[[str], None] = print, semantic: Optional[SemanticMapper] = None,
                 store=None):
        self.cfg = cfg
        self.semantic = semantic
        self.registry = DocumentRegistry(data_dir)
        self.force = force
        self.snapshots = snapshots
        self.log = log
        self.store = store
        self.reporter: Reporter = StoreReporter(store) if store is not None else Reporter()

    # ------------------------------------------------------------------ per document
    def _apply_metadata(self, record: DocumentRecord, pages: List[PageText],
                        state: Optional[str], doc_type: Optional[str]) -> None:
        cfg = self.cfg
        if state:
            spec = cfg.state_by_name(state)
            if spec is None:
                raise ValueError(f"Unknown state '{state}'. Known: {', '.join(s.name for s in cfg.states)}")
            record.state, record.state_code, record.state_source = spec.name, spec.code, "override"
        else:
            spec, evidence = detect_state(pages, cfg, record.original_filename)
            record.state = spec.name if spec else None
            record.state_code = spec.code if spec else None
            record.state_source = "detected" if spec else "undetermined"
            record.state_evidence = evidence
        if doc_type:
            if doc_type not in cfg.doc_types:
                raise ValueError(f"Unknown document type '{doc_type}'. Known: {', '.join(cfg.doc_types)}")
            record.document_type, record.document_type_source = doc_type, "override"
        else:
            code, evidence = detect_document_type(pages, cfg, record.original_filename)
            record.document_type, record.document_type_source = code, "detected"
            record.document_type_evidence = evidence
        record.report_year = detect_report_year(pages, cfg, record.document_type)
        record.pdf_created, record.pdf_modified = pdf_dates(self.registry.pdf_path(record))

    def _stage(self, record: DocumentRecord, stage: str, params: str, artifact: str,
               load: Callable[[], Any], compute: Callable[[], Tuple[Any, Dict[str, Any]]], res: DocumentResult,
               short: str) -> Any:
        """Reuse a cached stage artifact, or compute it; report either way."""
        rep, doc_id = self.reporter, record.document_id
        if not self.force and self.registry.is_current(record, stage, params, artifact):
            value = load()
            res.reused_stages.append(short)
            rep.stage(doc_id, stage, "cached", f"{short}: reused cached result")
            return value
        rep.stage(doc_id, stage, "running")
        t0 = time.time()
        value, stats = compute()
        self.registry.complete(record, stage, params, artifact, stats)
        rep.stage(doc_id, stage, "done", f"{short}: done in {time.time() - t0:.1f}s", **stats)
        return value

    def process_document(self, pdf_path: Path, state: Optional[str] = None,
                         doc_type: Optional[str] = None) -> DocumentResult:
        cfg, reg, rep = self.cfg, self.registry, self.reporter
        record = reg.register_local_pdf(pdf_path)
        res = DocumentResult(record)
        pdf = reg.pdf_path(record)
        doc_id = record.document_id
        rep.document(doc_id, file=record.original_filename, status="running")
        stage = Stage.TEXT_EXTRACTED
        try:
            # TEXT_EXTRACTED ----------------------------------------------------------
            p_text = stable_hash({"v": PIPELINE_VERSION, "cfg": cfg.section_hash("pipeline.text"),
                                  "code": code_fingerprint("pdf/text.py")})

            def text():
                pages = extract_page_texts(pdf, cfg.get("text.min_chars_for_text_layer", 50),
                                           on_page=lambda i, n: rep.progress(doc_id, Stage.TEXT_EXTRACTED, i, n))
                save_pages(pages, reg.artifact(doc_id, "pages.jsonl"))
                return pages, {"pages": len(pages), "needs_ocr": [p.page_number for p in pages if p.needs_ocr],
                               "blank": [p.page_number for p in pages if p.char_count == 0 and not p.needs_ocr]}
            res.pages = self._stage(record, stage, p_text, "pages.jsonl",
                                    lambda: load_pages(reg.artifact(doc_id, "pages.jsonl")), text, res, "text")
            record.page_count = len(res.pages)
            self._apply_metadata(record, res.pages, state, doc_type)
            reg.save(record)
            rep.document(doc_id, pages=len(res.pages), state=record.state, document_type=record.document_type,
                         report_year=record.report_year,
                         needs_ocr=sum(p.needs_ocr for p in res.pages))
            rep.count(documents=1, pages=len(res.pages))

            # CANDIDATES_FOUND --------------------------------------------------------
            stage = Stage.CANDIDATES_FOUND
            p_cand = stable_hash({"up": p_text, "cfg": cfg.section_hash("variables", "pipeline.candidates",
                                                                        "source_rules.units"),
                                  "code": code_fingerprint("extraction/candidates.py")})

            def candidates():
                ranked = rank_pages(res.pages, cfg)
                _write_json(reg.artifact(doc_id, "candidates.json"), [c.to_dict() for c in ranked])
                return ranked, {"candidate_pages": sum(c.is_candidate for c in ranked)}
            res.candidates = self._stage(
                record, stage, p_cand, "candidates.json",
                lambda: [CandidatePage.from_dict(d) for d in _read_json(reg.artifact(doc_id, "candidates.json"))],
                candidates, res, "candidates")
            n_cand = sum(c.is_candidate for c in res.candidates)
            rep.document(doc_id, candidate_pages=n_cand)
            rep.count(candidate_pages=n_cand)

            # TABLES_EXTRACTED --------------------------------------------------------
            stage = Stage.TABLES_EXTRACTED
            cand_pages = sorted(c.page_number for c in res.candidates if c.is_candidate)
            p_tab = stable_hash({"up": p_cand, "pages": cand_pages, "cfg": cfg.section_hash(
                "pipeline.tables", "source_rules.units", "source_rules.caption_patterns",
                "source_rules.cited_source_patterns", "source_rules.column_qualifiers"),
                "code": code_fingerprint("pdf/tables.py", "pdf/wordgrid.py", "extraction/parsers.py")})

            def tables():
                labels = {p.page_number: p.page_label for p in res.pages}
                found = extract_tables(pdf, doc_id, cand_pages, cfg, labels,
                                       on_page=lambda i, n: rep.progress(doc_id, Stage.TABLES_EXTRACTED, i, n))
                _write_json(reg.artifact(doc_id, "tables.json"), [t.to_dict() for t in found])
                return found, {"tables": len(found)}
            res.tables = self._stage(
                record, stage, p_tab, "tables.json",
                lambda: [ExtractedTable.from_dict(d) for d in _read_json(reg.artifact(doc_id, "tables.json"))],
                tables, res, "tables")
            rep.count(tables=len(res.tables))

            # VALUES_EXTRACTED --------------------------------------------------------
            stage = Stage.VALUES_EXTRACTED
            p_val = stable_hash({"up": p_tab, "state": record.state, "doc_type": record.document_type,
                                 "snapshots": self.snapshots,
                                 # the review threshold and tier bounds apply at validation, not here
                                 "cfg": cfg.section_hash("variables", "source_rules"),
                                 "confidence": {k: v for k, v in (cfg.pipeline.get("confidence") or {}).items()
                                                if k not in ("review_below", "tiers")},
                                 "code": code_fingerprint("extraction/structure.py", "extraction/mapper.py",
                                                          "extraction/parsers.py", "models.py")})

            def load_values():
                data = _read_json(reg.artifact(doc_id, "observations.json"))
                res.models = {k: TableModel.from_dict(v) for k, v in data["models"].items()}
                res.rejected = data["rejected"]
                return [Observation.from_dict(d) for d in data["observations"]]

            def values():
                self._extract_values(res, pdf)
                _write_json(reg.artifact(doc_id, "observations.json"), {
                    "models": {k: v.to_dict() for k, v in res.models.items()},
                    "observations": [o.to_dict() for o in res.observations],
                    "rejected": res.rejected,
                })
                return res.observations, {"observations": len(res.observations),
                                          "tables_with_values": len({o.table_id for o in res.observations})}
            res.observations = self._stage(record, stage, p_val, "observations.json", load_values, values, res,
                                           "values")
            year_tables = sum(1 for m in res.models.values() if m.orientation != "no_year_axis")
            rep.document(doc_id, tables=len(res.tables), year_tables=year_tables,
                         tables_with_values=len({o.table_id for o in res.observations}),
                         observations=len(res.observations))
            rep.count(year_tables=year_tables, tables_with_values=len({o.table_id for o in res.observations}),
                      observations=len(res.observations))

            if self.semantic is not None:
                stage = Stage.SEMANTIC_MAPPED
                self._semantic_step(res, pdf)
            self._reviewer_mappings(res)
            rep.document(doc_id, status="done", observations=len(res.observations))
        except Exception as exc:   # fail loudly, keep other documents going
            res.error = f"{stage}: {type(exc).__name__}: {exc}"
            reg.fail(record, stage, f"{type(exc).__name__}: {exc}")
            rep.stage(doc_id, stage, "failed", f"{type(exc).__name__}: {exc}")
            self.log(f"ERROR processing {record.original_filename} at {stage}: {exc}")
            self.log(traceback.format_exc())
        return res

    def _reviewer_mappings(self, res: DocumentResult) -> None:
        """Rows a reviewer assigned to a variable in the dashboard become observations."""
        if self.store is None:
            return
        book = ReviewBook.from_store(self.store)
        tables = {t.table_id: t for t in res.tables}
        have = {(o.table_id, o.raw_label) for o in res.observations}
        added: List[Observation] = []
        for tid, table in tables.items():
            matches = {label: m for label, m in book.label_matches(res.record.document_id, tid, self.cfg).items()
                       if (tid, label) not in have}
            if matches and tid in res.models:
                mapped = map_table(table, res.models[tid], res.record, self.cfg, extra_matches=matches)
                added.extend(o for o in mapped.observations if o.mapping_method.startswith("human:"))
        if added:
            for o in added:
                o.page_snapshot = str((self.registry.artifact(res.record.document_id, "snapshots")
                                       / f"{o.table_id}.png").relative_to(self.registry.data_dir))
            need = [tables[t] for t in {o.table_id for o in added}
                    if not (self.registry.data_dir / f"processed/{res.record.document_id}/snapshots/{t}.png").exists()]
            if need and self.snapshots:
                render_snapshots(self.registry.pdf_path(res.record), need,
                                 self.registry.artifact(res.record.document_id, "snapshots"))
            res.observations.extend(added)
            self.reporter.event("review", f"{len(added)} observation(s) from reviewer row mappings",
                                document_id=res.record.document_id)

    def _semantic_step(self, res: DocumentResult, pdf: Path) -> None:
        """LLM row mapping for selected tables. Deterministic results are kept whatever happens."""
        sm, cfg, rep = self.semantic, self.cfg, self.reporter
        doc_id = res.record.document_id
        tables = {t.table_id: t for t in res.tables}
        selected = sm.select_tables(res.tables, res.models)
        rep.stage(doc_id, Stage.SEMANTIC_MAPPED, "running")
        self.log(f"  LLM: interpreting {len(selected)} table(s) with {sm.primary.model_id}"
                 + (f" (fallback {sm.fallback.model_id})" if sm.fallback else ""))
        new_obs: List[Observation] = []
        try:
            for i, tid in enumerate(selected, start=1):
                rep.progress(doc_id, Stage.SEMANTIC_MAPPED, i - 1, len(selected))
                table, model = tables[tid], res.models[tid]
                interp = sm.interpret(table, model)
                res.interpretations[tid] = interp
                for call in interp.calls:
                    rep.count(llm_calls=1, llm_cached=int(bool(call.get("cached"))),
                              llm_errors=int("error" in call))
                    rep.event("llm", f"{call['model']} on {tid} (p.{table.page_number})"
                              + (" [cached]" if call.get("cached") else "")
                              + (f" error: {call['error'][:120]}" if "error" in call else ""),
                              level="warning" if "error" in call or call.get("issues") else "info",
                              document_id=doc_id, stage=Stage.SEMANTIC_MAPPED,
                              latency_s=call.get("latency_s"), issues=call.get("issues"))
                if interp.escalated:
                    rep.count(llm_escalations=1)
                sm.compare_with_rules(interp, res.observations)
                matches = sm.label_matches(interp)
                if matches:
                    mapped = map_table(table, model, res.record, cfg, extra_matches=matches)
                    new_obs.extend(o for o in mapped.observations if o.mapping_method.startswith("llm:"))
                for d in sm.unresolved(interp):
                    res.llm_unresolved.append({"table_id": tid, "page": table.page_number, "caption": table.caption,
                                               "label": d.label, "reason": d.reason, "model": d.model})
            rep.progress(doc_id, Stage.SEMANTIC_MAPPED, len(selected), len(selected))
        except LLMConfigurationError as exc:
            res.llm_error = str(exc)
            self.semantic = None          # do not repeat the failure for every document
            self.log(f"  LLM step aborted, deterministic results kept: {exc}")
            rep.stage(doc_id, Stage.SEMANTIC_MAPPED, "failed", f"LLM step aborted (deterministic results kept): {exc}")
        if self.snapshots and new_obs:
            have = {o.table_id for o in res.observations}
            need = [tables[t] for t in {o.table_id for o in new_obs} - have]
            paths = render_snapshots(pdf, need, self.registry.artifact(doc_id, "snapshots"),
                                     cfg.get("tables.snapshot_zoom", 1.6)) if need else {}
            for o in new_obs:
                path = paths.get(o.table_id) or self.registry.artifact(doc_id, "snapshots") / f"{o.table_id}.png"
                o.page_snapshot = str(path.relative_to(self.registry.data_dir))
        res.observations.extend(new_obs)
        _write_json(self.registry.artifact(doc_id, "llm_mappings.json"), {
            "prompt_version": PROMPT_VERSION, "error": res.llm_error,
            "tables": {k: v.to_dict() for k, v in res.interpretations.items()},
            "unresolved": res.llm_unresolved,
        })
        if res.llm_error is None:
            stats = {"tables": len(res.interpretations), "new_observations": len(new_obs),
                     "escalated": sum(i.escalated for i in res.interpretations.values()),
                     "hallucinations": sum(i.hallucinations for i in res.interpretations.values()),
                     "unresolved_rows": len(res.llm_unresolved)}
            self.registry.complete(res.record, Stage.SEMANTIC_MAPPED, stable_hash({
                "prompt": PROMPT_VERSION, "primary": sm.primary.model_id,
                "fallback": sm.fallback.model_id if sm.fallback else None}), "llm_mappings.json", stats)
            rep.stage(doc_id, Stage.SEMANTIC_MAPPED, "done", f"LLM: {len(new_obs)} new observation(s)", **stats)

    def _extract_values(self, res: DocumentResult, pdf: Path) -> None:
        cfg, rep = self.cfg, self.reporter
        for i, table in enumerate(res.tables, start=1):
            rep.progress(res.record.document_id, Stage.VALUES_EXTRACTED, i, len(res.tables))
            model = analyze_table(table, cfg)
            res.models[table.table_id] = model
            mapped = map_table(table, model, res.record, cfg)
            res.observations.extend(mapped.observations)
            res.rejected.extend(mapped.rejected)
        if self.snapshots and res.observations:
            used = {o.table_id for o in res.observations}
            snap_dir = self.registry.artifact(res.record.document_id, "snapshots")
            paths = render_snapshots(pdf, [t for t in res.tables if t.table_id in used], snap_dir,
                                     cfg.get("tables.snapshot_zoom", 1.6))
            for o in res.observations:
                if o.table_id in paths:
                    o.page_snapshot = str(paths[o.table_id].relative_to(self.registry.data_dir))

    # ------------------------------------------------------------------ validation
    def _validate(self, docs: List[DocumentResult], observations: List[Observation]):
        """Human decisions, checks, reconciliation, review status, panel. Mutates `observations`."""
        cfg = self.cfg
        book = ReviewBook.from_store(self.store)
        applied = apply_observation_reviews(observations, book, cfg)
        for o in observations:
            factor = cfg.crore_factor(o.unit)
            o.value_crore = round(o.value * factor, 6) if factor is not None else None
        active = [o for o in observations if o.review_decision != "rejected"]
        report_years = {d.record.document_id: d.record.report_year for d in docs}
        doc_dates = {d.record.document_id: max(filter(None, (d.record.pdf_created, d.record.pdf_modified)), default=None)
                     for d in docs}
        for o in active:
            check_observation(o, cfg, report_years.get(o.document_id), doc_dates.get(o.document_id))
        check_unit_consistency(active)
        check_provenance(observations, {d.record.document_id: d.pages for d in docs})
        arithmetic = apply_arithmetic([d for d in docs if not d.error], active, cfg)
        for d in docs:
            d.arithmetic = arithmetic.get(d.record.document_id, {})
        cells = reconcile(active, cfg)
        check_identities(cells, cfg)
        check_time_series(cells, cfg)
        check_panel_eligibility(cells, cfg)
        finalize(observations, cfg)
        finalize_reviewed(observations)
        panel_long, panel_wide = build_panel(cells, cfg)
        items = build_review_items(docs, observations, cells, cfg)
        return cells, panel_long, panel_wide, items, applied

    def _publish(self, observations: List[Observation], items: List[dict], panel_long: List[dict]) -> None:
        self.reporter.publish([o.to_dict() for o in observations], items, panel_long)
        open_items = sum(1 for i in items if i["severity"] != "decided")
        self.reporter.set_count(observations_total=len(observations), review_items=len(items),
                                review_open=open_items,
                                panel_values=sum(1 for r in panel_long if r["panel_status"] in ("included", "derived")))

    # ------------------------------------------------------------------ run
    def run(self, pdf_paths: Sequence[Path], out_dir: Path, state: Optional[str] = None,
            doc_type: Optional[str] = None, kind: str = "extract",
            options: Optional[Dict[str, Any]] = None) -> RunResult:
        rep = self.reporter
        run_options = {"pdfs": [str(Path(p).resolve()) for p in pdf_paths], "out": str(Path(out_dir).resolve()),
                       "state": state, "doc_type": doc_type, "force": self.force, "llm": self.semantic is not None,
                       **(options or {})}
        run_id = rep.run_started(kind, run_options)
        try:
            result = self._run(pdf_paths, Path(out_dir), state, doc_type)
        except Exception as exc:
            rep.run_finished("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        result.run_id = run_id
        status = "completed_with_errors" if result.failed else "completed"
        rep.run_finished(status, result.summary, results={k: str(v) for k, v in result.outputs.items()})
        return result

    def _run(self, pdf_paths: Sequence[Path], out_dir: Path, state: Optional[str], doc_type: Optional[str]) -> RunResult:
        cfg, rep = self.cfg, self.reporter
        docs: List[DocumentResult] = []
        for i, path in enumerate(pdf_paths, start=1):
            self.log(f"Processing {Path(path).name} ...")
            rep.event("document", f"document {i}/{len(pdf_paths)}: {Path(path).name}")
            docs.append(self.process_document(Path(path), state, doc_type))
            if self.store is not None and i < len(pdf_paths):
                # Interim results for the dashboard, validated on copies so the final pass is clean.
                copies = [Observation.from_dict(o.to_dict()) for d in docs for o in d.observations]
                _, panel_long, _, items, _ = self._validate(docs, copies)
                self._publish(copies, items, panel_long)
                rep.event("validation", f"interim results: {len(copies)} observation(s), "
                                        f"{sum(1 for it in items if it['severity'] != 'decided')} open review item(s)")

        observations = [o for d in docs for o in d.observations]
        cells, panel_long, panel_wide, items, applied = self._validate(docs, observations)
        if applied:
            rep.event("review", f"applied {applied} reviewer decision(s)")

        p_valid = stable_hash({"docs": sorted(d.record.document_id for d in docs),
                               "cfg": cfg.section_hash("pipeline.validation", "pipeline.reconciliation")})
        for d in docs:
            if not d.error:
                self.registry.complete(d.record, Stage.VALIDATED, p_valid, None, {
                    "needs_review": sum(o.review_status == "NEEDS_REVIEW" for o in d.observations)})
                rep.stage(d.record.document_id, Stage.VALIDATED, "done")

        outputs = write_outputs(Path(out_dir), docs, observations, cells, panel_long, panel_wide, cfg,
                                self.registry.data_dir,
                                reviews=self.store.all_reviews() if self.store is not None else None)
        summary = render_summary(docs, observations, cells, panel_long, outputs, cfg)
        with open(outputs["summary"], "w", encoding="utf-8") as fh:
            fh.write(summary + "\n")
        self._publish(observations, items, panel_long)
        open_items = sum(1 for it in items if it["severity"] != "decided")
        rep.event("validation", f"final results: {len(observations)} observation(s), {open_items} open review item(s), "
                                f"outputs in {out_dir}")
        return RunResult(docs, observations, cells, panel_long, panel_wide, outputs, summary, review_items=items)
