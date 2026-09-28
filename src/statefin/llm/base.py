"""Provider-independent LLM interface.

Everything outside this package depends only on `LLMProvider.structured_extract`, never
on a vendor SDK. Providers talk HTTP with the standard library, so no provider adds a
dependency, and a new backend (another hosted API, a local server) is one small subclass.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union


class LLMError(RuntimeError):
    """Any failure to obtain a usable structured response."""


class LLMConfigurationError(LLMError):
    """Missing API key, unknown provider, unreachable local server, ..."""


@dataclass
class LLMResult:
    data: Dict[str, Any]              # parsed JSON matching the requested schema
    raw_text: str
    provider: str
    model: str
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False

    @property
    def model_id(self) -> str:
        return f"{self.provider}/{self.model}"


def parse_json_text(text: str) -> Dict[str, Any]:
    """Parse a model's JSON answer, tolerating ```json fences; anything else is an error."""
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    if m:
        t = m.group(1)
    try:
        data = json.loads(t)
    except json.JSONDecodeError as exc:
        raise LLMError(f"response is not valid JSON: {exc}; first 200 chars: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError("response JSON is not an object")
    return data


class LLMProvider(ABC):
    """One model behind one API. Subclasses implement `_generate` only."""

    name = "base"

    def __init__(self, model: str, options: Optional[Dict[str, Any]] = None):
        self.model = model
        self.options = dict(options or {})
        self._last_call = 0.0

    @property
    def model_id(self) -> str:
        return f"{self.name}/{self.model}"

    @abstractmethod
    def _generate(self, instructions: str, input_text: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Return (raw JSON text, usage metadata)."""

    def structured_extract(self, input_data: Union[str, Dict[str, Any]], schema: Dict[str, Any],
                           instructions: str = "") -> LLMResult:
        """Send `input_data` with `instructions`; return JSON conforming to `schema` (JSON Schema)."""
        text = input_data if isinstance(input_data, str) else json.dumps(input_data, ensure_ascii=False)
        self._respect_rate_limit()
        t0 = time.time()
        raw, usage = self._generate(instructions, text, schema)
        return LLMResult(parse_json_text(raw), raw, self.name, self.model, usage, round(time.time() - t0, 2))

    # ------------------------------------------------------------------ shared HTTP plumbing
    def _respect_rate_limit(self) -> None:
        wait = float(self.options.get("min_interval_s", 0)) - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """POST JSON with retries on 429/5xx (exponential backoff, honouring Retry-After)."""
        body = json.dumps(payload).encode()
        retries = int(self.options.get("max_retries", 3))
        timeout = float(self.options.get("timeout_s", 60))
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=body, method="POST",
                                         headers={"Content-Type": "application/json", **(headers or {})})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1))
                    continue
                if exc.code in (401, 402, 403, 404):
                    what = {401: "authentication failed", 402: "billing/credits problem",
                            403: "permission denied", 404: "model or endpoint not found"}[exc.code]
                    raise LLMConfigurationError(f"{self.model_id}: {what} (HTTP {exc.code}): {detail}") from exc
                raise LLMError(f"{self.model_id}: HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise LLMConfigurationError(f"{self.model_id}: cannot reach {url.split('?')[0]}: {exc.reason}") from exc
        raise LLMError(f"{self.model_id}: retries exhausted")
