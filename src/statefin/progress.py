"""Progress reporting hooks used by the pipeline.

`Reporter` is a no-op, so the pipeline runs unchanged without a database (tests, library
use). `StoreReporter` writes events, per-document stage progress and counters to the
SQLite store that the dashboard streams from. Progress writes are throttled.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from statefin.store import Store


class Reporter:
    run_id: Optional[str] = None

    def run_started(self, kind: str, options: Dict[str, Any]) -> Optional[str]:
        return None

    def run_finished(self, status: str, summary: str = "", error: Optional[str] = None,
                     results: Optional[Dict[str, Any]] = None) -> None:
        pass

    def event(self, kind: str, message: str, level: str = "info", document_id: Optional[str] = None,
              stage: Optional[str] = None, **data: Any) -> None:
        pass

    def document(self, document_id: str, **fields: Any) -> None:
        pass

    def stage(self, document_id: str, stage: str, status: str, message: str = "", **stats: Any) -> None:
        pass

    def progress(self, document_id: str, stage: str, done: int, total: int) -> None:
        pass

    def count(self, **increments: int) -> None:
        pass

    def set_count(self, **values: Any) -> None:
        pass

    def publish(self, observations, review_items, panel_cells) -> None:
        pass


class StoreReporter(Reporter):
    def __init__(self, store: Store, min_interval_s: float = 0.25):
        self.store = store
        self.min_interval_s = min_interval_s
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.order: Dict[str, int] = {}
        self.counters: Dict[str, Any] = {}
        self._last_write: Dict[str, float] = {}

    def _due(self, key: str, force: bool = False) -> bool:
        now = time.time()
        if force or now - self._last_write.get(key, 0) >= self.min_interval_s:
            self._last_write[key] = now
            return True
        return False

    # ------------------------------------------------------------------ run
    def run_started(self, kind: str, options: Dict[str, Any]) -> str:
        self.run_id = self.store.start_run(kind, options)
        self.store.add_event(self.run_id, "run", f"{kind} run started", data={"options": options})
        return self.run_id

    def run_finished(self, status: str, summary: str = "", error: Optional[str] = None,
                     results: Optional[Dict[str, Any]] = None) -> None:
        if not self.run_id:
            return
        self.store.set_counters(self.run_id, self.counters)
        level = "error" if status == "failed" else "info"
        self.store.add_event(self.run_id, "run", f"run {status}" + (f": {error}" if error else ""), level=level)
        self.store.finish_run(self.run_id, status, summary, error, results)

    # ------------------------------------------------------------------ events
    def event(self, kind: str, message: str, level: str = "info", document_id: Optional[str] = None,
              stage: Optional[str] = None, **data: Any) -> None:
        self.store.add_event(self.run_id, kind, message, level, document_id, stage, data)
        if self._due("counters"):
            self.store.set_counters(self.run_id, self.counters)

    def count(self, **increments: int) -> None:
        for k, v in increments.items():
            self.counters[k] = self.counters.get(k, 0) + v
        if self.run_id and self._due("counters"):
            self.store.set_counters(self.run_id, self.counters)

    def set_count(self, **values: Any) -> None:
        self.counters.update(values)
        if self.run_id and self._due("counters", force=True):
            self.store.set_counters(self.run_id, self.counters)

    # ------------------------------------------------------------------ documents
    def _write_doc(self, document_id: str, force: bool = False) -> None:
        if self.run_id and self._due(f"doc:{document_id}", force):
            self.store.put_doc_progress(self.run_id, document_id, self.order[document_id], self.docs[document_id])

    def document(self, document_id: str, **fields: Any) -> None:
        if document_id not in self.docs:
            self.order[document_id] = len(self.order)
            self.docs[document_id] = {"document_id": document_id, "stages": {}, "current_stage": None,
                                      "status": "running"}
        self.docs[document_id].update(fields)
        self._write_doc(document_id, force=True)

    def stage(self, document_id: str, stage: str, status: str, message: str = "", **stats: Any) -> None:
        doc = self.docs.setdefault(document_id, {"document_id": document_id, "stages": {}})
        self.order.setdefault(document_id, len(self.order))
        entry = doc["stages"].setdefault(stage, {})
        entry.update({"status": status, "t": time.time()}, **({"stats": stats} if stats else {}))
        if status == "running":
            entry.update(done=0, total=None, started=time.time())
            doc["current_stage"] = stage
        if status == "failed":
            doc["status"] = "failed"
            doc["error"] = message
        self._write_doc(document_id, force=True)
        if message or status in ("done", "cached", "failed"):
            text = message or f"{stage.lower().replace('_', ' ')} {status}"
            self.store.add_event(self.run_id, "stage", text, "error" if status == "failed" else "info",
                                 document_id, stage, {"status": status, **stats})

    def progress(self, document_id: str, stage: str, done: int, total: int) -> None:
        doc = self.docs.get(document_id)
        if doc is None:
            return
        entry = doc["stages"].setdefault(stage, {"status": "running"})
        entry.update(done=done, total=total)
        self._write_doc(document_id, force=(done >= total))
        if self.run_id and self._due("counters"):
            self.store.set_counters(self.run_id, self.counters)

    # ------------------------------------------------------------------ results
    def publish(self, observations, review_items, panel_cells) -> None:
        if self.run_id:
            self.store.publish(self.run_id, observations, review_items, panel_cells)
