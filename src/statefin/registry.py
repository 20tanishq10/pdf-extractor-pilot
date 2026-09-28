"""Document registry: content-addressed cache and per-stage status for every document.

Layout (all plain files, inspectable by hand):

    data/raw/<document_id>.pdf                 cached copy of the source PDF
    data/processed/<document_id>/manifest.json metadata + status of every stage
    data/processed/<document_id>/pages.jsonl   TEXT_EXTRACTED
    data/processed/<document_id>/candidates.json CANDIDATES_FOUND
    data/processed/<document_id>/tables.json   TABLES_EXTRACTED
    data/processed/<document_id>/observations.json VALUES_EXTRACTED (pre-validation)
    data/processed/<document_id>/snapshots/*.png  page crops of source tables

document_id is the first 16 hex digits of the file's SHA-256, so the same PDF uploaded
twice (or downloaded under another name) is recognised and processed once. A stage is
reused only if it completed with the same parameter hash (code version + the config
sections it depends on) and its artifact still exists; otherwise it re-runs. A failure is
recorded on the stage and the next run resumes from there.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from statefin.models import DocumentRecord, Stage


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentRegistry:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ records
    def doc_dir(self, document_id: str) -> Path:
        d = self.processed_dir / document_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def artifact(self, document_id: str, name: str) -> Path:
        return self.doc_dir(document_id) / name

    def load(self, document_id: str) -> Optional[DocumentRecord]:
        path = self.processed_dir / document_id / "manifest.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return DocumentRecord.from_dict(json.load(fh))

    def save(self, record: DocumentRecord) -> None:
        path = self.doc_dir(record.document_id) / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, indent=2, ensure_ascii=False)
        tmp.replace(path)

    def list_documents(self) -> List[DocumentRecord]:
        out = []
        for manifest in sorted(self.processed_dir.glob("*/manifest.json")):
            with open(manifest, encoding="utf-8") as fh:
                out.append(DocumentRecord.from_dict(json.load(fh)))
        return out

    def register_local_pdf(self, pdf_path: Path, copy: bool = True) -> DocumentRecord:
        """Mode B entry point: register a user-supplied PDF (idempotent)."""
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        with open(pdf_path, "rb") as fh:
            if fh.read(5) != b"%PDF-":
                raise ValueError(f"Not a PDF file: {pdf_path}")
        digest = sha256_file(pdf_path)
        document_id = digest[:16]
        record = self.load(document_id)
        cached = self.raw_dir / f"{document_id}.pdf"
        if copy and not cached.exists():
            shutil.copy2(pdf_path, cached)
        if record is None:
            record = DocumentRecord(
                document_id=document_id,
                sha256=digest,
                original_filename=pdf_path.name,
                original_path=str(pdf_path),
                cached_path=str(cached.relative_to(self.data_dir)) if cached.exists() else None,
                source_kind="local_upload",
                registered_at=_now(),
            )
            record.stages[Stage.DOWNLOADED] = {"completed_at": _now(), "note": "local file registered"}
            self.save(record)
        return record

    def pdf_path(self, record: DocumentRecord) -> Path:
        if record.cached_path and (self.data_dir / record.cached_path).exists():
            return self.data_dir / record.cached_path
        return Path(record.original_path)

    # ------------------------------------------------------------------ stages
    def is_current(self, record: DocumentRecord, stage: str, params_hash: str, artifact: str) -> bool:
        info = record.stages.get(stage)
        return bool(info and info.get("status") == "done" and info.get("params_hash") == params_hash
                    and (self.processed_dir / record.document_id / artifact).exists())

    def complete(self, record: DocumentRecord, stage: str, params_hash: str, artifact: Optional[str],
                 stats: Optional[Dict[str, Any]] = None) -> None:
        record.stages[stage] = {"status": "done", "completed_at": _now(), "params_hash": params_hash,
                                "artifact": artifact, "stats": stats or {}}
        if Stage.ORDER.index(stage) >= Stage.ORDER.index(record.status):
            record.status = stage
        record.last_error = None
        self.save(record)

    def fail(self, record: DocumentRecord, stage: str, error: str) -> None:
        record.stages[stage] = {"status": "failed", "failed_at": _now(), "error": error}
        record.last_error = f"{stage}: {error}"
        self.save(record)
