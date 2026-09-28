"""Concrete providers: Gemini API, Ollama (local), and any OpenAI-compatible server.

All three receive the same instructions, input and JSON Schema, so they can be compared
on the same benchmark. Temperature defaults to 0 for reproducibility.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from statefin.llm.base import LLMConfigurationError, LLMError, LLMProvider

_GEMINI_SCHEMA_KEYS = {"type", "format", "description", "nullable", "enum", "properties", "required", "items",
                       "minItems", "maxItems", "minimum", "maximum", "propertyOrdering"}


def to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """JSON Schema -> the OpenAPI subset accepted by Gemini's responseSchema (upper-case types)."""
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_KEYS:
            continue
        if key == "type":
            out["type"] = str(value).upper()
        elif key == "properties":
            out["properties"] = {k: to_gemini_schema(v) for k, v in value.items()}
            out["propertyOrdering"] = list(value.keys())
        elif key == "items":
            out["items"] = to_gemini_schema(value)
        else:
            out[key] = value
    return out


def _require_key(options: Dict[str, Any], provider: str) -> str:
    env = options.get("api_key_env")
    key = os.environ.get(env, "") if env else ""
    if not key:
        raise LLMConfigurationError(
            f"{provider}: no API key. Set {env} in the environment or in the project's .env file.")
    return key


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, options: Optional[Dict[str, Any]] = None):
        super().__init__(model, options)
        self._key = _require_key(self.options, self.name)
        self.base_url = self.options.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def _generate(self, instructions: str, input_text: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        gen: Dict[str, Any] = {
            "temperature": self.options.get("temperature", 0),
            "responseMimeType": "application/json",
            "responseSchema": to_gemini_schema(schema),
        }
        if self.options.get("thinking_budget") is not None:
            gen["thinkingConfig"] = {"thinkingBudget": int(self.options["thinking_budget"])}
        payload = {
            "systemInstruction": {"parts": [{"text": instructions}]},
            "contents": [{"role": "user", "parts": [{"text": input_text}]}],
            "generationConfig": gen,
        }
        resp = self._post_json(f"{self.base_url}/models/{self.model}:generateContent", payload,
                               headers={"x-goog-api-key": self._key})
        candidates = resp.get("candidates") or []
        if not candidates:
            raise LLMError(f"{self.model_id}: no candidates (promptFeedback: {resp.get('promptFeedback')})")
        cand = candidates[0]
        if cand.get("finishReason") not in (None, "STOP"):
            raise LLMError(f"{self.model_id}: finishReason {cand.get('finishReason')}")
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        usage = resp.get("usageMetadata", {})
        return text, {"input_tokens": usage.get("promptTokenCount"), "output_tokens": usage.get("candidatesTokenCount"),
                      "thinking_tokens": usage.get("thoughtsTokenCount"), "total_tokens": usage.get("totalTokenCount")}


class OllamaProvider(LLMProvider):
    """Local models (Qwen, Gemma, Llama, ...) served by Ollama; structured output via `format`."""
    name = "ollama"

    def __init__(self, model: str, options: Optional[Dict[str, Any]] = None):
        super().__init__(model, options)
        self.base_url = (os.environ.get(self.options.get("base_url_env", ""), "")
                         or self.options.get("base_url", "http://localhost:11434")).rstrip("/")

    def _generate(self, instructions: str, input_text: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": input_text}],
            "format": schema,
            "stream": False,
            "options": {"temperature": self.options.get("temperature", 0),
                        **{k: self.options[k] for k in ("seed", "num_ctx", "top_k", "top_p") if k in self.options}},
            **({"keep_alive": self.options["keep_alive"]} if "keep_alive" in self.options else {}),
        }
        resp = self._post_json(f"{self.base_url}/api/chat", payload)
        return resp.get("message", {}).get("content", ""), {
            "input_tokens": resp.get("prompt_eval_count"), "output_tokens": resp.get("eval_count")}


class OpenAICompatibleProvider(LLMProvider):
    """Any /v1/chat/completions server: vLLM, llama.cpp server, LM Studio, OpenRouter, ..."""
    name = "openai_compatible"

    def __init__(self, model: str, options: Optional[Dict[str, Any]] = None):
        super().__init__(model, options)
        self.base_url = (os.environ.get(self.options.get("base_url_env", ""), "")
                         or self.options.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        env = self.options.get("api_key_env")
        self._key = os.environ.get(env, "") if env else ""

    def _generate(self, instructions: str, input_text: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if self.options.get("json_mode", "schema") == "schema":
            response_format = {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}}
        else:   # servers without JSON-schema support: JSON mode + schema in the prompt
            response_format = {"type": "json_object"}
            instructions = f"{instructions}\n\nReturn JSON matching this JSON Schema:\n{schema}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": input_text}],
            "temperature": self.options.get("temperature", 0),
            "response_format": response_format,
        }
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        resp = self._post_json(f"{self.base_url}/chat/completions", payload, headers)
        choices = resp.get("choices") or []
        if not choices:
            raise LLMError(f"{self.model_id}: no choices in response")
        usage = resp.get("usage", {})
        return choices[0].get("message", {}).get("content", ""), {
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}


PROVIDERS = {cls.name: cls for cls in (GeminiProvider, OllamaProvider, OpenAICompatibleProvider)}
