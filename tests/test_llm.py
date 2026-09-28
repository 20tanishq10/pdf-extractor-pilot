"""LLM layer tests. No network: a scripted provider, and a local HTTP server that mimics
the Gemini / Ollama / OpenAI-compatible wire formats."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from statefin.extraction.mapper import map_table
from statefin.extraction.semantic import SemanticMapper
from statefin.extraction.structure import analyze_table
from statefin.llm.base import LLMConfigurationError, LLMProvider
from statefin.llm.cache import CachedProvider, ResponseCache
from statefin.llm.providers import GeminiProvider, OllamaProvider, OpenAICompatibleProvider, to_gemini_schema
from statefin.llm.settings import load_llm_settings
from statefin.pipeline import Pipeline

from test_tables_and_mapping import doc_record, grid_table


class ScriptedProvider(LLMProvider):
    """Answers with a function of the request; records every call."""
    name = "fake"

    def __init__(self, model, answer):
        super().__init__(model, {})
        self.answer = answer
        self.calls = []

    def _generate(self, instructions, input_text, schema):
        request = json.loads(input_text)
        self.calls.append(request)
        return json.dumps({"rows": self.answer(request["table"]["rows"])}), {"input_tokens": 10}


def answer_with(mapping, confidence=0.95):
    """mapping: label -> variable/NONE/UNKNOWN (default NONE)."""
    def fn(rows):
        return [{"row_id": r["row_id"], "label": r["label"], "canonical_variable": mapping.get(r["label"], "NONE"),
                 "confidence": confidence, "reason": "test"} for r in rows]
    return fn


def settings(**policy):
    raw = {"primary": {"provider": "gemini", "model": "small"}, "policy": policy}
    return load_llm_settings(raw, env={})


TABLE = [
    ["Components", "2023-24", "2024-25"],
    ["Pensionary Charges and Retirement Payouts", "9,212.64", "9,543.85"],
    ["Salaries & Wages", "17,124.03", "17,461.50"],
    ["Interest Payment/Revenue Receipts", "14.42", "15.32"],
]


def test_llm_classifies_label_and_python_supplies_the_value(cfg):
    table = grid_table(TABLE)
    model = analyze_table(table, cfg)
    small = ScriptedProvider("small", answer_with({"Pensionary Charges and Retirement Payouts": "pension_expenditure",
                                                   "Salaries & Wages": "salary_wages"}))
    sm = SemanticMapper(cfg, settings(), small, None)
    interp = sm.interpret(table, model)
    request = small.calls[0]
    assert {r["label"] for r in request["table"]["rows"]} >= {"Pensionary Charges and Retirement Payouts"}
    assert "rows" in sm.schema()["properties"] and "value" not in json.dumps(sm.schema())

    obs = map_table(table, model, doc_record(), cfg, extra_matches=sm.label_matches(interp)).observations
    llm = [o for o in obs if o.mapping_method.startswith("llm:")]
    assert sorted(o.value for o in llm) == [9212.64, 9543.85]           # values from the cells
    assert all(o.canonical_variable == "pension_expenditure" and o.confidence <= 0.79 for o in llm)
    rules = [o for o in obs if o.canonical_variable == "salary_wages"]
    assert all(o.mapping_method == "alias_exact" for o in rules)       # rules keep precedence


def test_llm_cannot_turn_a_ratio_row_into_a_level(cfg):
    table = grid_table(TABLE)
    model = analyze_table(table, cfg)
    small = ScriptedProvider("small", answer_with({"Interest Payment/Revenue Receipts": "interest_payments"}))
    sm = SemanticMapper(cfg, settings(), small, None)
    obs = map_table(table, model, doc_record(), cfg, extra_matches=sm.label_matches(sm.interpret(table, model))).observations
    assert not any(o.raw_label.startswith("Interest Payment/") for o in obs)


def test_hallucinated_rows_are_discarded_and_escalated(cfg):
    table = grid_table(TABLE)
    model = analyze_table(table, cfg)

    def bad(rows):
        return [{"row_id": "R99", "label": "Invented row", "canonical_variable": "pension_expenditure",
                 "confidence": 0.99, "reason": "made up"},
                {"row_id": rows[0]["row_id"], "label": "Pension (paraphrased)", "canonical_variable": "pension_expenditure",
                 "confidence": 0.99, "reason": "wrong label"}]

    small = ScriptedProvider("small", bad)
    large = ScriptedProvider("large", answer_with({"Pensionary Charges and Retirement Payouts": "pension_expenditure"}))
    sm = SemanticMapper(cfg, settings(), small, large)
    interp = sm.interpret(table, model)
    assert interp.hallucinations == 2 and interp.escalated and len(large.calls) == 1
    assert interp.final["Pensionary Charges and Retirement Payouts"].variable == "pension_expenditure"
    assert interp.final["Pensionary Charges and Retirement Payouts"].model == "fake/large"


def test_fallback_is_not_called_when_primary_is_confident(cfg):
    table = grid_table(TABLE)
    small = ScriptedProvider("small", answer_with({"Pensionary Charges and Retirement Payouts": "pension_expenditure",
                                                   "Salaries & Wages": "salary_wages"}))
    large = ScriptedProvider("large", answer_with({}))
    interp = SemanticMapper(cfg, settings(), small, large).interpret(table, analyze_table(table, cfg))
    assert not interp.escalated and large.calls == []


def test_disagreement_between_models_becomes_review(cfg):
    table = grid_table(TABLE)
    small = ScriptedProvider("small", answer_with({"Pensionary Charges and Retirement Payouts": "pension_expenditure"},
                                                  confidence=0.5))
    large = ScriptedProvider("large", answer_with({"Pensionary Charges and Retirement Payouts": "salary_wages"}))
    sm = SemanticMapper(cfg, settings(), small, large)
    interp = sm.interpret(table, analyze_table(table, cfg))
    d = interp.final["Pensionary Charges and Retirement Payouts"]
    assert d.status == "unknown" and "disagree" in d.reason
    assert sm.label_matches(interp) == {}
    assert [u.label for u in sm.unresolved(interp)] == ["Pensionary Charges and Retirement Payouts"]


def test_selection_sends_only_tables_with_unmapped_relevant_rows(cfg):
    mapped_only = grid_table([["Components", "2023-24", "2024-25"], ["Pension", "1", "2"]])
    mapped_only.table_id = "t_mapped"
    unmapped = grid_table(TABLE)
    unmapped.table_id = "t_unmapped"
    models = {t.table_id: analyze_table(t, cfg) for t in (mapped_only, unmapped)}
    sm = SemanticMapper(cfg, settings(), ScriptedProvider("small", answer_with({})), None)
    assert sm.select_tables([mapped_only, unmapped], models) == ["t_unmapped"]
    sm_all = SemanticMapper(cfg, settings(table_selection="all_year_tables"), ScriptedProvider("s", answer_with({})), None)
    assert sm_all.select_tables([mapped_only, unmapped], models) == ["t_mapped", "t_unmapped"]


def test_response_cache(tmp_path):
    inner = ScriptedProvider("small", answer_with({}))
    cached = CachedProvider(inner, ResponseCache(tmp_path))
    rows = {"table": {"rows": [{"row_id": "R1", "label": "Pension"}]}}
    first = cached.structured_extract(rows, {"type": "object"}, "instr")
    second = cached.structured_extract(rows, {"type": "object"}, "instr")
    assert not first.cached and second.cached and len(inner.calls) == 1
    assert second.data == first.data


def test_settings_overrides():
    raw = {"primary": {"provider": "gemini", "model": "a"}, "fallback": {"provider": "gemini", "model": "b"},
           "providers": {"gemini": {"api_key_env": "X"}}}
    s = load_llm_settings(raw, env={"STATEFIN_LLM_MODEL": "c", "STATEFIN_LLM_FALLBACK_MODEL": "none"})
    assert s.primary.model == "c" and s.fallback is None and s.primary.options["api_key_env"] == "X"
    s = load_llm_settings(raw, env={}, overrides={"primary_provider": "ollama", "primary_model": "qwen2.5:7b"})
    assert s.primary.model_id == "ollama/qwen2.5:7b" and s.fallback.model_id == "gemini/b"
    assert s.confidence_cap < 0.80 and not s.accept_without_review


def test_end_to_end_with_llm_cross_check(cfg, synthetic_pdf, tmp_path):
    small = ScriptedProvider("small", answer_with({"Salaries & Wages4": "salary_wages", "Pension": "salary_wages"}))
    sm = SemanticMapper(cfg, settings(table_selection="all_year_tables"), small, None)
    result = Pipeline(cfg, tmp_path / "data", log=lambda _m: None, semantic=sm).run([synthetic_pdf], tmp_path / "out")
    obs = result.observations
    salary = [o for o in obs if o.canonical_variable == "salary_wages"]
    assert any("agrees" in r for o in salary for r in o.confidence_reasons)
    pension = [o for o in obs if o.canonical_variable == "pension_expenditure"]
    assert pension and all(any(f["check"] == "llm_disagrees" for f in o.validation_flags) for o in pension)
    assert all(o.review_status == "NEEDS_REVIEW" for o in pension)
    assert (tmp_path / "out" / "llm_label_suggestions.csv").exists()


# ----------------------------------------------------------------------------- wire formats

class _Server:
    def __init__(self, status=200, body=None):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": json.loads(self.rfile.read(length))})
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


SCHEMA = {"type": "object", "properties": {"rows": {"type": "array", "items": {
    "type": "object", "properties": {"row_id": {"type": "string"}}, "required": ["row_id"],
    "additionalProperties": False}}}, "required": ["rows"], "additionalProperties": False}


def test_gemini_wire_format(monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "k-123")
    server = _Server(body={"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"rows": []}'}]}}],
                           "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7}})
    try:
        p = GeminiProvider("gemini-test", {"api_key_env": "TEST_GEMINI_KEY", "base_url": server.url, "max_retries": 0})
        res = p.structured_extract({"x": 1}, SCHEMA, "instructions")
    finally:
        server.close()
    req = server.requests[0]
    assert req["path"] == "/models/gemini-test:generateContent"
    assert req["headers"]["x-goog-api-key"] == "k-123"                 # key in a header, not the URL
    gen = req["body"]["generationConfig"]
    assert gen["temperature"] == 0 and gen["responseMimeType"] == "application/json"
    assert gen["responseSchema"]["type"] == "OBJECT" and "additionalProperties" not in json.dumps(gen["responseSchema"])
    assert req["body"]["systemInstruction"]["parts"][0]["text"] == "instructions"
    assert res.data == {"rows": []} and res.usage["total_tokens"] == 7


def test_gemini_billing_error_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "k")
    server = _Server(status=402, body={"error": {"message": "credits depleted"}})
    try:
        p = GeminiProvider("m", {"api_key_env": "TEST_GEMINI_KEY", "base_url": server.url, "max_retries": 0})
        with pytest.raises(LLMConfigurationError, match="billing"):
            p.structured_extract({}, SCHEMA, "")
    finally:
        server.close()


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(LLMConfigurationError, match="NO_SUCH_KEY"):
        GeminiProvider("m", {"api_key_env": "NO_SUCH_KEY"})


def test_ollama_wire_format():
    server = _Server(body={"message": {"content": '{"rows": [{"row_id": "R1"}]}'}, "eval_count": 3})
    try:
        res = OllamaProvider("qwen2.5:7b", {"base_url": server.url}).structured_extract({}, SCHEMA, "sys")
    finally:
        server.close()
    body = server.requests[0]["body"]
    assert server.requests[0]["path"] == "/api/chat" and body["format"] == SCHEMA and body["stream"] is False
    assert res.data["rows"][0]["row_id"] == "R1"


def test_openai_compatible_wire_format(monkeypatch):
    monkeypatch.setenv("TEST_OAI_KEY", "sk-x")
    server = _Server(body={"choices": [{"message": {"content": '```json\n{"rows": []}\n```'}}]})
    try:
        p = OpenAICompatibleProvider("local-model", {"base_url": server.url, "api_key_env": "TEST_OAI_KEY"})
        res = p.structured_extract({}, SCHEMA, "sys")
    finally:
        server.close()
    req = server.requests[0]
    assert req["path"] == "/chat/completions" and req["headers"]["authorization"] == "Bearer sk-x"
    assert req["body"]["response_format"]["json_schema"]["schema"] == SCHEMA
    assert res.data == {"rows": []}


def test_gemini_schema_conversion():
    g = to_gemini_schema(SCHEMA)
    assert g["type"] == "OBJECT" and g["properties"]["rows"]["items"]["type"] == "OBJECT"
    assert g["propertyOrdering"] == ["rows"]
