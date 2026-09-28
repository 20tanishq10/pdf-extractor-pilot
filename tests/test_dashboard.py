"""Store, review application, live reporting and the dashboard HTTP API."""
import json
import socket
import time
import urllib.error
import urllib.request

import pytest

from statefin.dashboard.server import serve
from statefin.pipeline import Pipeline
from statefin.store import Store


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "data" / "statefin.db")


def run(cfg, pdf, tmp_path, store, **kw):
    return Pipeline(cfg, tmp_path / "data", log=lambda _m: None, store=store, **kw).run([pdf], tmp_path / "out")


def obs_of(result, var, fy):
    return next(o for o in result.observations if o.canonical_variable == var and o.financial_year == fy)


def panel_value(result, var, fy):
    row = next(r for r in result.panel_long if r["variable"] == var and r["financial_year"] == fy)
    return row["value"], row["panel_status"]


# ----------------------------------------------------------------------------- store

def test_review_ledger_is_append_only_and_latest_wins(store):
    store.add_review("observation", "abc", "accepted", reviewer="a")
    store.add_review("observation", "abc", "corrected", value=12.5, note="typo in PDF", reviewer="b")
    assert store.latest_reviews()[("observation", "abc")]["decision"] == "corrected"
    store.add_review("observation", "abc", "reset", reviewer="b")
    assert ("observation", "abc") not in store.latest_reviews()
    assert [r["decision"] for r in store.review_history("observation", "abc")] == ["reset", "corrected", "accepted"]


@pytest.mark.parametrize("kwargs", [
    {"target_type": "observation", "target_id": "x", "decision": "mapped"},
    {"target_type": "observation", "target_id": "x", "decision": "corrected"},          # no value
    {"target_type": "row_label", "target_id": "x", "decision": "mapped"},                # no variable
    {"target_type": "nonsense", "target_id": "x", "decision": "accepted"},
])
def test_invalid_reviews_are_refused(store, kwargs):
    with pytest.raises(ValueError):
        store.add_review(**kwargs)


# ----------------------------------------------------------------------------- live reporting

def test_run_reports_progress_and_publishes_results(cfg, synthetic_pdf, tmp_path, store):
    result = run(cfg, synthetic_pdf, tmp_path, store)
    r = store.get_run(result.run_id)
    assert r["status"] == "completed" and r["counters"]["pages"] == 4 and r["counters"]["observations_total"] > 0
    docs = store.doc_progress(result.run_id)
    assert docs[0]["status"] == "done"
    assert {"TEXT_EXTRACTED", "TABLES_EXTRACTED", "VALUES_EXTRACTED", "VALIDATED"} <= set(docs[0]["stages"])
    kinds = {e["kind"] for e in store.recent_events()}
    assert {"run", "stage", "validation"} <= kinds
    assert len(store.observations(result.run_id)) == len(result.observations)
    assert store.results_run_id() == result.run_id


# ----------------------------------------------------------------------------- decisions applied

def test_rejected_observation_leaves_the_panel(cfg, synthetic_pdf, tmp_path, store):
    first = run(cfg, synthetic_pdf, tmp_path, store)
    target = obs_of(first, "pension_expenditure", "2024-25")
    store.add_review("observation", target.observation_id, "rejected", note="test")
    second = run(cfg, synthetic_pdf, tmp_path, store)
    o = obs_of(second, "pension_expenditure", "2024-25")
    assert o.review_status == "HUMAN_REJECTED" and o.value == 9543.85        # kept in the raw layer
    assert not any(r["variable"] == "pension_expenditure" and r["financial_year"] == "2024-25"
                   for r in second.panel_long)


def test_correction_keeps_the_extracted_value(cfg, synthetic_pdf, tmp_path, store):
    first = run(cfg, synthetic_pdf, tmp_path, store)
    target = obs_of(first, "salary_wages", "2024-25")
    store.add_review("observation", target.observation_id, "corrected", value=17461.05, note="digits transposed",
                     reviewer="tg")
    second = run(cfg, synthetic_pdf, tmp_path, store)
    o = obs_of(second, "salary_wages", "2024-25")
    assert (o.value, o.value_as_extracted, o.raw_value_text) == (17461.05, 17461.50, "17,461.50")
    assert o.review_status == "HUMAN_CORRECTED" and o.reviewed_by == "tg"
    assert any(f["check"] == "human_corrected" for f in o.validation_flags)
    assert panel_value(second, "salary_wages", "2024-25") == (17461.05, "included")
    assert not any(f["check"] == "provenance_text_mismatch" for f in o.validation_flags)


def test_accepting_an_error_flagged_value_admits_it_to_the_panel(cfg, synthetic_pdf, tmp_path, store):
    from statefin.export.panel import build_panel
    result = run(cfg, synthetic_pdf, tmp_path, store)
    key = next(k for k in result.cells if k[1:] == ("2024-25", "gsdp"))
    rep = result.cells[key].representative
    rep.add_flag("fy_not_closed_at_document_date", "error", "synthetic error flag")
    status = lambda: next(r["panel_status"] for r in build_panel(result.cells, cfg)[0]
                          if r["variable"] == "gsdp" and r["financial_year"] == "2024-25")
    assert status() == "excluded"
    rep.review_decision = "accepted"          # what a reviewer's "Accept" sets on the next run
    assert status() == "included"


def test_reviewer_row_mapping_creates_observations_from_cells(cfg, synthetic_pdf, tmp_path, store):
    first = run(cfg, synthetic_pdf, tmp_path, store)
    doc_id = first.documents[0].record.document_id
    table_id = next(o.table_id for o in first.observations if o.canonical_variable == "pension_expenditure")
    store.add_review("row_label", f"{doc_id}|{table_id}|Total", "mapped", variable="revenue_expenditure",
                     reviewer="tg", note="mechanism test")
    second = run(cfg, synthetic_pdf, tmp_path, store)
    human = [o for o in second.observations if o.mapping_method == "human:tg"]
    assert sorted(o.value for o in human) == [31699.69, 31985.04, 33266.28]   # the row's printed cells
    assert all(o.review_status == "HUMAN_ACCEPTED" for o in human)


# ----------------------------------------------------------------------------- HTTP API

@pytest.fixture()
def server(cfg, synthetic_pdf, tmp_path, store):
    run(cfg, synthetic_pdf, tmp_path, store)
    port = free_port()
    httpd = serve(cfg, tmp_path / "data", port, [synthetic_pdf.parent], log=lambda _m: None, store=store)
    yield f"http://127.0.0.1:{port}", httpd, store
    httpd.shutdown()


def get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_api_overview_panel_and_images(server):
    base, _, _ = server
    status, ctype, body = get(f"{base}/api/overview")
    ov = json.loads(body)
    assert status == 200 and ov["results_run_id"] and ov["run"]["status"] == "completed"
    panel = json.loads(get(f"{base}/api/panel")[2])
    assert any(c["variable"] == "pension_expenditure" for c in panel["cells"])
    oid = next(c["representative_observation_id"] for c in panel["cells"] if c["variable"] == "pension_expenditure")
    detail = json.loads(get(f"{base}/api/observation?id={oid}")[2])
    assert detail["observation"]["cell_bbox"]
    status, ctype, png = get(f"{base}/api/image?obs={oid}&kind=cell")
    assert status == 200 and ctype == "image/png" and png[:4] == b"\x89PNG"
    assert get(f"{base}/")[2].startswith(b"<!doctype html>")


def test_api_refuses_unsafe_writes(server):
    base, _, _ = server
    body = {"target_type": "observation", "target_id": "x", "decision": "accepted"}
    assert post(f"{base}/api/review", body)[0] == 403                                   # no X-Statefin
    assert post(f"{base}/api/review", body, {"X-Statefin": "1", "Origin": "https://evil.example"})[0] == 403
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(f"{base}/api/overview", {"Host": "evil.example"})                            # DNS rebinding
    assert exc.value.code == 403
    assert post(f"{base}/api/review", {**body, "decision": "bogus"}, {"X-Statefin": "1"})[0] == 400


def test_api_review_and_pick_are_saved(server):
    base, httpd, store = server
    items = json.loads(get(f"{base}/api/review/items?status=all")[2])["items"]
    ids = [o["observation_id"] for o in store.observations(store.results_run_id())][:2]
    status, data = post(f"{base}/api/review", {"target_type": "observation", "target_id": ids[0],
                                               "decision": "corrected", "value": "1,234.5", "note": "n",
                                               "reviewer": "tg"}, {"X-Statefin": "1"})
    assert status == 200 and data["review"]["value"] == 1234.5
    status, data = post(f"{base}/api/review/pick", {"chosen_id": ids[0], "others": ids, "reviewer": "tg"},
                        {"X-Statefin": "1"})
    assert status == 200 and [r["decision"] for r in data["reviews"]] == ["accepted", "rejected"]
    latest = store.latest_reviews()
    assert latest[("observation", ids[0])]["decision"] == "accepted"
    assert latest[("observation", ids[1])]["decision"] == "rejected"
    assert httpd.app.runs.rebuild_due is not None                                        # auto-rebuild queued
    assert isinstance(items, list)


def test_api_stream_sends_live_events(server):
    base, _, _ = server
    with urllib.request.urlopen(f"{base}/api/stream", timeout=10) as r:
        assert r.headers.get("Content-Type") == "text/event-stream"
        chunk = b""
        deadline = time.time() + 5
        while b"event: overview" not in chunk and time.time() < deadline:
            chunk += r.read1(4096)
    assert b"event: log" in chunk and b"event: overview" in chunk
