"""SQLite store for the live dashboard and the human-review ledger (data/statefin.db).

Tables
    runs          one row per extraction / rebuild run, with live counters and heartbeat
    events        append-only progress log (what the dashboard streams)
    doc_progress  current stage and progress of each document in a run
    observations  observations published by a run (interim during a run, final at the end)
    review_items  review queue published by a run
    panel_cells   panel coverage published by a run
    reviews       APPEND-ONLY human decisions. Never updated or deleted; the latest row per
                  target wins, and a "reset" row withdraws a decision. This is the audit trail.

WAL journaling lets the dashboard read while an extraction (possibly in another process,
e.g. extract.py in a terminal) writes. Every call opens its own short-lived connection, so
the store is safe to use from the HTTP server's threads.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, kind TEXT, status TEXT, started_at TEXT, finished_at TEXT,
    heartbeat REAL, options TEXT, counters TEXT, summary TEXT, error TEXT, results TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts TEXT, level TEXT, kind TEXT,
    document_id TEXT, stage TEXT, message TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
CREATE TABLE IF NOT EXISTS doc_progress (
    run_id TEXT, document_id TEXT, position INTEGER, data TEXT, updated_at TEXT,
    PRIMARY KEY (run_id, document_id));
CREATE TABLE IF NOT EXISTS observations (
    run_id TEXT, observation_id TEXT, state TEXT, financial_year TEXT, variable TEXT, value REAL,
    review_status TEXT, data TEXT, PRIMARY KEY (run_id, observation_id));
CREATE TABLE IF NOT EXISTS review_items (
    run_id TEXT, item_id TEXT, position INTEGER, data TEXT, PRIMARY KEY (run_id, item_id));
CREATE TABLE IF NOT EXISTS panel_cells (
    run_id TEXT, state TEXT, financial_year TEXT, variable TEXT, data TEXT,
    PRIMARY KEY (run_id, state, financial_year, variable));
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, target_type TEXT, target_id TEXT, decision TEXT,
    value REAL, variable TEXT, note TEXT, reviewer TEXT, created_at TEXT, context TEXT);
CREATE INDEX IF NOT EXISTS reviews_target ON reviews(target_type, target_id, id);
CREATE TABLE IF NOT EXISTS config_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, reviewer TEXT, kind TEXT, key TEXT,
    old_value TEXT, new_value TEXT, note TEXT);
"""

DECISIONS = {
    "observation": {"accepted", "rejected", "corrected", "reset"},
    "row_label": {"mapped", "none", "reset"},
    "missing": {"acknowledged", "reset"},
}
STALE_AFTER_S = 300


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _loads(value: Optional[str], default=None):
    return json.loads(value) if value else default


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs
    def start_run(self, kind: str, options: Dict[str, Any]) -> str:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self._conn() as c:
            c.execute("INSERT INTO runs (run_id, kind, status, started_at, heartbeat, options, counters) "
                      "VALUES (?, ?, 'running', ?, ?, ?, '{}')",
                      (run_id, kind, now_iso(), time.time(), json.dumps(options, default=str)))
        return run_id

    def finish_run(self, run_id: str, status: str, summary: str = "", error: Optional[str] = None,
                   results: Optional[Dict[str, Any]] = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET status=?, finished_at=?, heartbeat=?, summary=?, error=?, results=? "
                      "WHERE run_id=?", (status, now_iso(), time.time(), summary, error,
                                         json.dumps(results or {}, default=str), run_id))

    def set_counters(self, run_id: str, counters: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET counters=?, heartbeat=? WHERE run_id=?",
                      (json.dumps(counters), time.time(), run_id))

    def mark_stale_runs(self) -> int:
        with self._conn() as c:
            cur = c.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running' "
                            "AND heartbeat < ?", (now_iso(), time.time() - STALE_AFTER_S))
            return cur.rowcount

    def _run_row(self, row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        for k in ("options", "counters", "results"):
            d[k] = _loads(d.get(k), {})
        return d

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            return self._run_row(c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())

    def latest_run(self, kinds: Tuple[str, ...] = ("extract", "rebuild")) -> Optional[Dict[str, Any]]:
        q = f"SELECT * FROM runs WHERE kind IN ({','.join('?' * len(kinds))}) ORDER BY started_at DESC, rowid DESC LIMIT 1"
        with self._conn() as c:
            return self._run_row(c.execute(q, kinds).fetchone())

    def active_run(self) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE status='running' AND heartbeat >= ? "
                            "ORDER BY started_at DESC LIMIT 1", (time.time() - STALE_AFTER_S,)).fetchone()
            return self._run_row(row)

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [self._run_row(r) for r in c.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()]

    def results_run_id(self) -> Optional[str]:
        """Most recent run that has published observations (interim or final)."""
        with self._conn() as c:
            row = c.execute("SELECT o.run_id FROM observations o JOIN runs r ON r.run_id = o.run_id "
                            "ORDER BY r.started_at DESC, r.rowid DESC LIMIT 1").fetchone()
            return row["run_id"] if row else None

    # ------------------------------------------------------------------ events
    def add_event(self, run_id: Optional[str], kind: str, message: str, level: str = "info",
                  document_id: Optional[str] = None, stage: Optional[str] = None,
                  data: Optional[Dict[str, Any]] = None) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO events (run_id, ts, level, kind, document_id, stage, message, data) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (run_id, now_iso(), level, kind, document_id, stage, message,
                             json.dumps(data or {}, default=str)))
            if run_id:
                c.execute("UPDATE runs SET heartbeat=? WHERE run_id=?", (time.time(), run_id))
            return int(cur.lastrowid)

    def events_after(self, after_id: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?", (after_id, limit)).fetchall()
        return [{**dict(r), "data": _loads(r["data"], {})} for r in rows]

    def recent_events(self, limit: int = 300) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "data": _loads(r["data"], {})} for r in reversed(rows)]

    def last_event_id(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT MAX(id) AS m FROM events").fetchone()
            return int(row["m"] or 0)

    # ------------------------------------------------------------------ document progress
    def put_doc_progress(self, run_id: str, document_id: str, position: int, data: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO doc_progress (run_id, document_id, position, data, updated_at) VALUES (?,?,?,?,?) "
                      "ON CONFLICT(run_id, document_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                      (run_id, document_id, position, json.dumps(data, default=str), now_iso()))
            c.execute("UPDATE runs SET heartbeat=? WHERE run_id=?", (time.time(), run_id))

    def doc_progress(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM doc_progress WHERE run_id=? ORDER BY position", (run_id,)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    # ------------------------------------------------------------------ published results
    def publish(self, run_id: str, observations: Iterable[Dict[str, Any]], review_items: Iterable[Dict[str, Any]],
                panel_cells: Iterable[Dict[str, Any]]) -> None:
        with self._conn() as c:
            for table in ("observations", "review_items", "panel_cells"):
                c.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
            c.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?)", [
                (run_id, o["observation_id"], o.get("state"), o["financial_year"], o["canonical_variable"],
                 o["value"], o["review_status"], json.dumps(o, default=str)) for o in observations])
            c.executemany("INSERT INTO review_items VALUES (?,?,?,?)", [
                (run_id, it["item_id"], i, json.dumps(it, default=str)) for i, it in enumerate(review_items)])
            c.executemany("INSERT OR REPLACE INTO panel_cells VALUES (?,?,?,?,?)", [
                (run_id, p["state"], p["financial_year"], p["variable"], json.dumps(p, default=str))
                for p in panel_cells])

    def observations(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM observations WHERE run_id=? ORDER BY state, variable, financial_year",
                             (run_id,)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def observation(self, run_id: str, observation_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT data FROM observations WHERE run_id=? AND observation_id=?",
                            (run_id, observation_id)).fetchone()
        return json.loads(row["data"]) if row else None

    def cell_observations(self, run_id: str, state: str, fy: str, variable: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM observations WHERE run_id=? AND state IS ? AND financial_year=? "
                             "AND variable=?", (run_id, state, fy, variable)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def review_items(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM review_items WHERE run_id=? ORDER BY position", (run_id,)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def panel_cells(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT data FROM panel_cells WHERE run_id=?", (run_id,)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    # ------------------------------------------------------------------ review ledger
    def add_review(self, target_type: str, target_id: str, decision: str, value: Optional[float] = None,
                   variable: Optional[str] = None, note: Optional[str] = None, reviewer: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if target_type not in DECISIONS:
            raise ValueError(f"unknown review target type {target_type!r}")
        if decision not in DECISIONS[target_type]:
            raise ValueError(f"decision {decision!r} not valid for {target_type}; use one of {sorted(DECISIONS[target_type])}")
        if decision == "corrected" and value is None:
            raise ValueError("a correction needs a value")
        if decision == "mapped" and not variable:
            raise ValueError("a row mapping needs a variable")
        with self._conn() as c:
            cur = c.execute("INSERT INTO reviews (target_type, target_id, decision, value, variable, note, reviewer, "
                            "created_at, context) VALUES (?,?,?,?,?,?,?,?,?)",
                            (target_type, target_id, decision, value, variable, note or "", reviewer or "reviewer",
                             now_iso(), json.dumps(context or {}, default=str)))
            row = c.execute("SELECT * FROM reviews WHERE id=?", (cur.lastrowid,)).fetchone()
        return {**dict(row), "context": _loads(row["context"], {})}

    def latest_reviews(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Effective decision per target (latest row wins; 'reset' removes the decision)."""
        with self._conn() as c:
            rows = c.execute("SELECT r.* FROM reviews r JOIN (SELECT target_type, target_id, MAX(id) AS id "
                             "FROM reviews GROUP BY target_type, target_id) m ON r.id = m.id").fetchall()
        return {(r["target_type"], r["target_id"]): {**dict(r), "context": _loads(r["context"], {})}
                for r in rows if r["decision"] != "reset"}

    def review_history(self, target_type: str, target_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM reviews WHERE target_type=? AND target_id=? ORDER BY id DESC",
                             (target_type, target_id)).fetchall()
        return [{**dict(r), "context": _loads(r["context"], {})} for r in rows]

    def all_reviews(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM reviews ORDER BY id").fetchall()
        return [{**dict(r), "context": _loads(r["context"], {})} for r in rows]

    # ------------------------------------------------------------------ configuration changes
    def log_config_change(self, kind: str, key: str, old: Any, new: Any, reviewer: str = "reviewer",
                          note: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO config_changes (created_at, reviewer, kind, key, old_value, new_value, note) "
                      "VALUES (?,?,?,?,?,?,?)", (now_iso(), reviewer, kind, key, json.dumps(old, default=str),
                                                 json.dumps(new, default=str), note or ""))

    def config_changes(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM config_changes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "old_value": _loads(r["old_value"]), "new_value": _loads(r["new_value"])} for r in rows]

    def reviews_since(self, iso_time: Optional[str]) -> int:
        with self._conn() as c:
            if not iso_time:
                return int(c.execute("SELECT COUNT(*) AS n FROM reviews").fetchone()["n"])
            return int(c.execute("SELECT COUNT(*) AS n FROM reviews WHERE created_at > ?", (iso_time,)).fetchone()["n"])
