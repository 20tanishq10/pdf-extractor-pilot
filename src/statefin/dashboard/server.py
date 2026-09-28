"""Local real-time dashboard: live extraction progress + human review (standard library only).

    python dashboard.py            -> http://127.0.0.1:8765

Live updates use Server-Sent Events (/api/stream), fed from the SQLite store that every
extraction writes to, whether it was started here or with extract.py in a terminal.

Security: binds to 127.0.0.1 only; requests must carry a localhost Host header; every POST
must send `X-Statefin: 1` with a JSON body and, if an Origin is present, a same-origin one.
Browsers cannot add that header cross-origin without a CORS preflight this server never
grants, so other web pages cannot submit reviews or start runs.
"""
from __future__ import annotations

import csv
import io
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from statefin.config import Config
from statefin.dashboard.images import render_observation
from statefin.pdf.tables import render_snapshots
from statefin.registry import DocumentRegistry
from statefin.store import Store

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 1 << 20


# ---------------------------------------------------------------------- runs

class RunManager:
    """One extraction/rebuild at a time, in a background thread; debounced auto-rebuild."""

    def __init__(self, cfg: Config, data_dir: Path, store: Store, log: Callable[[str], None] = print):
        self.cfg, self.data_dir, self.store, self.log = cfg, data_dir, store, log
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.rebuild_due: Optional[float] = None
        threading.Thread(target=self._ticker, daemon=True).start()

    def busy(self) -> bool:
        return bool(self._thread and self._thread.is_alive()) or self.store.active_run() is not None

    def start(self, pdfs: Sequence[str], llm: bool = False, force: bool = False, kind: str = "extract",
              out: Optional[str] = None, state: Optional[str] = None, doc_type: Optional[str] = None) -> None:
        with self._lock:
            if self.busy():
                raise RuntimeError("an extraction is already running")
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            args=(list(pdfs), llm, force, kind, out, state, doc_type))
            self._thread.start()

    def _run(self, pdfs, llm, force, kind, out, state, doc_type) -> None:
        from statefin.cli import make_semantic_mapper
        from statefin.llm.base import LLMConfigurationError
        from statefin.pipeline import Pipeline

        semantic = None
        if llm:
            try:
                semantic = make_semantic_mapper(self.cfg, self.data_dir)
            except LLMConfigurationError as exc:
                self.store.add_event(None, "run", f"LLM not usable, run not started: {exc}", "error")
                return
        from statefin.config import Config
        self.cfg = Config(self.cfg.config_dir)          # settings may have changed since start-up
        pipe = Pipeline(self.cfg, self.data_dir, force=force, snapshots=True, log=self.log,
                        semantic=semantic, store=self.store)
        try:
            pipe.run([Path(p) for p in pdfs], Path(out) if out else self.data_dir / "output", state, doc_type,
                     kind=kind)
        except Exception:   # already recorded as a failed run with its error
            self.log(traceback.format_exc())

    def schedule_rebuild(self, delay: float = 1.5) -> None:
        self.rebuild_due = time.time() + delay

    def _ticker(self) -> None:
        while True:
            time.sleep(0.5)
            if self.rebuild_due and time.time() >= self.rebuild_due and not self.busy():
                self.rebuild_due = None
                last = self.store.latest_run(("extract", "rebuild"))
                opts = (last or {}).get("options") or {}
                if opts.get("pdfs"):
                    try:
                        self.start(opts["pdfs"], llm=bool(opts.get("llm")), force=False, kind="rebuild",
                                   out=opts.get("out"), state=opts.get("state"), doc_type=opts.get("doc_type"))
                    except RuntimeError:
                        self.rebuild_due = time.time() + 2


# ---------------------------------------------------------------------- application

class App:
    def __init__(self, cfg: Config, data_dir: Path, store: Store, docs_dirs: Sequence[Path] = (),
                 log: Callable[[str], None] = print, port: int = 8765):
        self.cfg, self.data_dir, self.store = cfg, Path(data_dir), store
        self.registry = DocumentRegistry(self.data_dir)
        self.docs_dirs = [Path(d) for d in docs_dirs]
        self.runs = RunManager(cfg, self.data_dir, store, log)
        self.port = port

    # -------------------------------------------------------------- read models
    def _results_run(self) -> Optional[Dict[str, Any]]:
        rid = self.store.results_run_id()
        return self.store.get_run(rid) if rid else None

    def overview(self) -> Dict[str, Any]:
        active = self.store.active_run()
        shown = active or self.store.latest_run(("extract", "rebuild"))
        results = self._results_run()
        items = self.store.review_items(results["run_id"]) if results else []
        latest = self.store.latest_reviews()
        open_n = sum(1 for it in items if it["severity"] != "decided"
                     and (it["target_type"], it["target_id"]) not in latest)
        decided = sum(1 for it in items if (it["target_type"], it["target_id"]) in latest or it["severity"] == "decided")
        if shown:
            shown = {**{k: shown[k] for k in ("run_id", "kind", "status", "started_at", "finished_at", "counters",
                                              "options", "error")},
                     "heartbeat_age_s": round(time.time() - (shown.get("heartbeat") or time.time()), 1)}
        return {
            "run": shown,
            "active": active is not None,
            "documents": self.store.doc_progress(shown["run_id"]) if shown else [],
            "results_run_id": results["run_id"] if results else None,
            "results_status": results["status"] if results else None,
            "review": {"open": open_n, "decided": decided, "total": len(items),
                       "pending_rebuild": self.store.reviews_since(results["started_at"]) if results else 0},
            "rebuild_scheduled": self.runs.rebuild_due is not None,
            "last_event_id": self.store.last_event_id(),
            "variables": {n: {"role": v.role, "description": v.description} for n, v in self.cfg.variables.items()},
        }

    def review_items(self, q: Dict[str, str]) -> Dict[str, Any]:
        results = self._results_run()
        if not results:
            return {"items": [], "run_id": None}
        latest = self.store.latest_reviews()
        out = []
        for it in self.store.review_items(results["run_id"]):
            decision = latest.get((it["target_type"], it["target_id"]))
            it["decision"] = decision
            it["open"] = decision is None and it["severity"] != "decided"
            status = q.get("status", "open")
            if status == "open" and not it["open"] or status == "decided" and it["open"]:
                continue
            if q.get("type") and it["item_type"] != q["type"]:
                continue
            if q.get("severity") and it["severity"] != q["severity"]:
                continue
            if q.get("variable") and it.get("variable") != q["variable"]:
                continue
            text = q.get("q", "").lower()
            if text and text not in json.dumps([it.get("raw_label"), it.get("variable"), it.get("financial_year"),
                                                it.get("source_table"), it.get("checks")]).lower():
                continue
            out.append(it)
        return {"items": out, "run_id": results["run_id"]}

    def observation_detail(self, obs_id: str) -> Dict[str, Any]:
        results = self._results_run()
        obs = self.store.observation(results["run_id"], obs_id) if results else None
        if obs is None:
            raise KeyError(f"observation {obs_id} not in the latest results")
        group = self.store.cell_observations(results["run_id"], obs["state"], obs["financial_year"],
                                             obs["canonical_variable"])
        latest = self.store.latest_reviews()
        for g in group:
            g["pending_decision"] = latest.get(("observation", g["observation_id"]))
        return {"observation": obs, "group": group,
                "decision": latest.get(("observation", obs_id)),
                "history": self.store.review_history("observation", obs_id)}

    def row_detail(self, document_id: str, table_id: str, label: str) -> Dict[str, Any]:
        path = self.data_dir / "processed" / document_id / "observations.json"
        tables = self.data_dir / "processed" / document_id / "tables.json"
        if not path.exists() or not tables.exists():
            raise KeyError("document not processed")
        model = json.loads(path.read_text())["models"].get(table_id, {})
        table = next((t for t in json.loads(tables.read_text()) if t["table_id"] == table_id), {})
        cells = [{"fy": v["fy"], "raw_text": v["raw_text"], "value": v["value"], "accepted": v["accepted"],
                  "column_header": v["column_header"]} for v in model.get("value_cells", []) if v["raw_label"] == label]
        tid = f"{document_id}|{table_id}|{label}"
        return {"label": label, "cells": cells, "caption": table.get("caption"), "unit": table.get("unit"),
                "page": table.get("page_number"), "decision": self.store.latest_reviews().get(("row_label", tid)),
                "history": self.store.review_history("row_label", tid)}

    def panel(self) -> Dict[str, Any]:
        results = self._results_run()
        cells = self.store.panel_cells(results["run_id"]) if results else []
        latest = self.store.latest_reviews()
        for c in cells:
            rid = c.get("representative_observation_id")
            c["pending_decision"] = latest.get(("observation", rid)) if rid else None
        variables = [v for v in self.cfg.core_variables] + [n for n, v in self.cfg.variables.items()
                                                           if v.role == "optional"]
        return {"cells": cells, "variables": variables,
                "states": sorted({c["state"] for c in cells}),
                "years": sorted({c["financial_year"] for c in cells}, reverse=True)}

    def observations(self, q: Dict[str, str]) -> Dict[str, Any]:
        results = self._results_run()
        rows = self.store.observations(results["run_id"]) if results else []
        keep = ("observation_id", "state", "financial_year", "canonical_variable", "value", "value_as_extracted",
                "unit", "raw_label", "source_document", "source_page", "source_table", "confidence",
                "confidence_tier", "review_status", "review_decision", "mapping_method", "reconciliation_status")
        text = q.get("q", "").lower()
        out = []
        for o in rows:
            if q.get("variable") and o["canonical_variable"] != q["variable"]:
                continue
            if q.get("review_status") and o["review_status"] != q["review_status"]:
                continue
            if text and text not in f"{o['raw_label']} {o['canonical_variable']} {o['financial_year']} " \
                                    f"{o.get('source_table')} {o['source_document']}".lower():
                continue
            out.append({k: o.get(k) for k in keep})
        return {"observations": out[:3000], "total": len(out)}

    def documents(self) -> List[Dict[str, Any]]:
        out = []
        for r in self.registry.list_documents():
            out.append({"document_id": r.document_id, "file": r.original_filename, "state": r.state,
                        "document_type": r.document_type, "report_year": r.report_year, "pages": r.page_count,
                        "status": r.status, "last_error": r.last_error,
                        "stages": {k: {"status": v.get("status"), "at": v.get("completed_at") or v.get("failed_at"),
                                       "stats": v.get("stats", {})} for k, v in r.stages.items()}})
        return out

    def files(self) -> List[Dict[str, Any]]:
        registered = {r.original_path: r.document_id for r in self.registry.list_documents()}
        seen, out = set(), []
        for d in self.docs_dirs:
            if d.is_dir():
                for p in sorted(d.glob("*.pdf")):
                    seen.add(str(p.resolve()))
                    out.append({"path": str(p.resolve()), "name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1),
                                "document_id": registered.get(str(p.resolve()))})
        for path, doc_id in registered.items():
            if path not in seen and Path(path).exists():
                out.append({"path": path, "name": Path(path).name,
                            "size_mb": round(Path(path).stat().st_size / 1e6, 1), "document_id": doc_id})
        return out

    def image(self, q: Dict[str, str]) -> Path:
        if q.get("obs"):
            results = self._results_run()
            obs = self.store.observation(results["run_id"], q["obs"]) if results else None
            if obs is None:
                raise KeyError("unknown observation")
            record = self.registry.load(obs["document_id"])
            kind = "page" if q.get("kind") == "page" else "cell"
            out = self.data_dir / "processed" / obs["document_id"] / "review_images" / f"{obs['observation_id']}_{kind}.png"
            return render_observation(self.registry.pdf_path(record), obs, out, whole_page=(kind == "page"))
        doc_id, table_id = q.get("doc", ""), q.get("table", "")
        if not doc_id.isalnum() or not table_id.replace("_", "").isalnum():
            raise KeyError("bad image request")
        path = self.data_dir / "processed" / doc_id / "snapshots" / f"{table_id}.png"
        if not path.exists():
            record = self.registry.load(doc_id)
            tables = json.loads((self.data_dir / "processed" / doc_id / "tables.json").read_text())
            from statefin.models import ExtractedTable
            table = next(ExtractedTable.from_dict(t) for t in tables if t["table_id"] == table_id)
            render_snapshots(self.registry.pdf_path(record), [table], path.parent)
        return path

    # -------------------------------------------------------------- writes
    def add_review(self, body: Dict[str, Any]) -> Dict[str, Any]:
        value = body.get("value")
        if value not in (None, ""):
            value = float(str(value).replace(",", ""))
        else:
            value = None
        review = self.store.add_review(body["target_type"], str(body["target_id"]), body["decision"], value=value,
                                       variable=body.get("variable") or None, note=body.get("note"),
                                       reviewer=(body.get("reviewer") or "reviewer")[:80],
                                       context=body.get("context") or {})
        self.store.add_event(None, "review", f"{review['reviewer']}: {review['decision']} {review['target_type']} "
                                             f"{str(review['target_id'])[:60]}", data={"review_id": review["id"]})
        if body.get("rebuild", True):
            self.runs.schedule_rebuild()
        return review

    def pick(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        chosen, others = str(body["chosen_id"]), [str(x) for x in body.get("others", []) if str(x) != str(body["chosen_id"])]
        note = body.get("note") or "picked as the correct value among conflicting observations"
        out = [self.add_review({"target_type": "observation", "target_id": chosen, "decision": "accepted",
                                "note": note, "reviewer": body.get("reviewer"), "rebuild": False})]
        for oid in others:
            out.append(self.add_review({"target_type": "observation", "target_id": oid, "decision": "rejected",
                                        "note": f"another observation ({chosen}) was picked as correct",
                                        "reviewer": body.get("reviewer"), "rebuild": False}))
        self.runs.schedule_rebuild()
        return out

    def start_run(self, body: Dict[str, Any]) -> Dict[str, Any]:
        pdfs = [str(Path(p).expanduser().resolve()) for p in body.get("pdfs", []) if str(p).strip()]
        if not pdfs:
            raise ValueError("choose at least one PDF")
        for p in pdfs:
            path = Path(p)
            if not path.is_file() or path.suffix.lower() != ".pdf":
                raise ValueError(f"not a PDF file: {p}")
        self.runs.start(pdfs, llm=bool(body.get("llm")), force=bool(body.get("force")))
        return {"started": True, "pdfs": pdfs}

    # -------------------------------------------------------------- settings and learning
    def reload_config(self) -> None:
        self.cfg = Config(self.cfg.config_dir)
        self.runs.cfg = self.cfg

    def settings(self) -> Dict[str, Any]:
        from statefin import settings as S
        return {**S.describe(self.cfg), "history": self.store.config_changes(100)}

    def settings_preview(self, value: float) -> Dict[str, Any]:
        """How many observations would need review at a different accuracy threshold."""
        results = self._results_run()
        obs = self.store.observations(results["run_id"]) if results else []
        def needs(o, t):
            if o.get("review_decision"):
                return False
            blocking = any(f["severity"] in ("warning", "error") for f in o.get("validation_flags", []))
            return blocking or o["confidence"] < t
        now = float(self.cfg.get("confidence.review_below", 0.8))
        return {"total": len(obs), "threshold_now": now, "needs_review_now": sum(needs(o, now) for o in obs),
                "threshold": value, "needs_review_at": sum(needs(o, value) for o in obs),
                "low_confidence_at": sum(1 for o in obs if not o.get("review_decision") and o["confidence"] < value)}

    def save_settings(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from statefin import settings as S
        done = S.apply_changes(self.cfg, body.get("changes") or {}, body.get("reset") or [], self.store,
                               (body.get("reviewer") or "reviewer")[:80], body.get("note") or "")
        if done:
            self.reload_config()
            self.store.add_event(None, "settings", "settings changed: " + ", ".join(
                f"{d['key']} {d['old']} -> {d['new']}" for d in done))
            self.runs.schedule_rebuild(0.5)
        return {"changed": done}

    def learn(self) -> Dict[str, Any]:
        from statefin import learn as L
        results = self._results_run()
        obs = self.store.observations(results["run_id"]) if results else []
        min_support = int(self.cfg.get("learn.min_support", 2))
        return {"aliases": L.alias_candidates(self.cfg, obs, self.store.latest_reviews()),
                "rules": L.rule_suggestions(self.cfg, obs, min_support),
                "rejections": sum(1 for o in obs if o.get("review_decision") == "rejected"),
                "overrides": self.cfg.overrides, "min_support": min_support}

    def learn_apply(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from statefin import settings as S
        kind, variable, value = body["kind"], body.get("variable"), body["value"]
        added = S.add_rule(self.cfg, kind, variable, value, self.store, (body.get("reviewer") or "reviewer")[:80],
                           body.get("note") or "")
        if added:
            self.reload_config()
            self.store.add_event(None, "learn", f"config rule added: {kind} {variable or ''} {value}")
            self.runs.schedule_rebuild(0.5)
        return {"added": added}

    def reviews_csv(self) -> str:
        buf = io.StringIO()
        cols = ["id", "created_at", "reviewer", "target_type", "target_id", "decision", "value", "variable", "note",
                "context"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in self.store.all_reviews():
            w.writerow({**r, "context": json.dumps(r.get("context") or {}, ensure_ascii=False)})
        return "﻿" + buf.getvalue()


# ---------------------------------------------------------------------- HTTP

def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "statefin-dashboard/1"

        def log_message(self, *_args):
            pass

        # ---------------------------------------------------------- helpers
        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ("127.0.0.1", "localhost", "::1")

        def _send(self, status: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = 200) -> None:
            self._send(status, json.dumps(obj, default=str, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        # ---------------------------------------------------------- GET
        def do_GET(self):
            if not self._host_ok():
                return self._error(403, "dashboard only answers on localhost")
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8",
                                      {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; "
                                       "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                                       "connect-src 'self'; frame-ancestors 'none'"})
                if url.path == "/api/stream":
                    resume = self.headers.get("Last-Event-ID") or q.get("after") or "0"
                    return self._stream(int(resume) if str(resume).isdigit() else 0)
                if url.path == "/api/overview":
                    return self._json(app.overview())
                if url.path == "/api/events":
                    return self._json(app.store.events_after(int(q.get("after") or 0), int(q.get("limit") or 500))
                                      if q.get("after") else app.store.recent_events(300))
                if url.path == "/api/review/items":
                    return self._json(app.review_items(q))
                if url.path == "/api/observation":
                    return self._json(app.observation_detail(q["id"]))
                if url.path == "/api/row":
                    return self._json(app.row_detail(q["doc"], q["table"], q["label"]))
                if url.path == "/api/panel":
                    return self._json(app.panel())
                if url.path == "/api/observations":
                    return self._json(app.observations(q))
                if url.path == "/api/documents":
                    return self._json(app.documents())
                if url.path == "/api/runs":
                    return self._json(app.store.list_runs(30))
                if url.path == "/api/files":
                    return self._json(app.files())
                if url.path == "/api/image":
                    return self._send(200, app.image(q).read_bytes(), "image/png")
                if url.path == "/api/settings":
                    return self._json(app.settings())
                if url.path == "/api/settings/preview":
                    return self._json(app.settings_preview(float(q["value"])))
                if url.path == "/api/learn":
                    return self._json(app.learn())
                if url.path == "/api/reviews.csv":
                    return self._send(200, app.reviews_csv().encode("utf-8"), "text/csv; charset=utf-8",
                                      {"Content-Disposition": "attachment; filename=human_reviews.csv"})
                return self._error(404, "not found")
            except (KeyError, StopIteration) as exc:
                return self._error(404, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as exc:
                traceback.print_exc()
                return self._error(500, f"{type(exc).__name__}: {exc}")

        def _stream(self, after: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            last = after
            last_overview, last_ping, last_sent = "", time.time(), 0.0
            try:
                if not after:
                    recent = app.store.recent_events(300)
                    if recent:
                        last = recent[-1]["id"]
                        self._sse("log", recent, last)
                while True:
                    events = app.store.events_after(last, 300)
                    if events:
                        last = events[-1]["id"]
                        self._sse("log", events, last)
                    if events or time.time() - last_sent > 1.0:
                        ov = json.dumps(app.overview(), default=str)
                        if ov != last_overview:
                            self.wfile.write(f"event: overview\ndata: {ov}\n\n".encode())
                            self.wfile.flush()
                            last_overview = ov
                        last_sent = time.time()
                    if time.time() - last_ping > 15:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = time.time()
                    time.sleep(0.4)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _sse(self, event: str, data: Any, event_id: int) -> None:
            self.wfile.write(f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode())
            self.wfile.flush()

        # ---------------------------------------------------------- POST
        def do_POST(self):
            if not self._host_ok():
                return self._error(403, "dashboard only answers on localhost")
            if self.headers.get("X-Statefin") != "1":
                return self._error(403, "missing X-Statefin header")
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
                return self._error(403, "cross-origin request refused")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._error(415, "JSON body required")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._error(413, "body too large")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/review":
                    return self._json({"review": app.add_review(body)})
                if path == "/api/review/pick":
                    return self._json({"reviews": app.pick(body)})
                if path == "/api/run":
                    return self._json(app.start_run(body), 202)
                if path == "/api/settings":
                    return self._json(app.save_settings(body))
                if path == "/api/learn/apply":
                    return self._json(app.learn_apply(body))
                if path == "/api/rebuild":
                    app.runs.schedule_rebuild(0)
                    return self._json({"scheduled": True}, 202)
                return self._error(404, "not found")
            except (ValueError, KeyError, TypeError) as exc:
                return self._error(400, str(exc))
            except RuntimeError as exc:
                return self._error(409, str(exc))
            except Exception as exc:
                traceback.print_exc()
                return self._error(500, f"{type(exc).__name__}: {exc}")

    return Handler


def serve(cfg: Config, data_dir: Path, port: int = 8765, docs_dirs: Sequence[Path] = (),
          log: Callable[[str], None] = print, store: Optional[Store] = None) -> ThreadingHTTPServer:
    """Start the dashboard in a background thread; returns the server (call .shutdown() to stop)."""
    store = store or Store(Path(data_dir) / "statefin.db")
    stale = store.mark_stale_runs()
    if stale:
        log(f"marked {stale} stale run(s) as interrupted")
    app = App(cfg, data_dir, store, docs_dirs, log, port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    httpd.daemon_threads = True
    httpd.app = app  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
