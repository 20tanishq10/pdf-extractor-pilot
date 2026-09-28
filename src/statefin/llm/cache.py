"""Content-addressed cache of LLM responses.

The key is a hash of (provider, model, prompt version, instructions, input, schema), so a
rerun with identical inputs reuses the stored answer: no API cost, and the result is
reproducible even though hosted models change over time. Each entry keeps the full
request and raw response for audit.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from statefin.llm.base import LLMProvider, LLMResult


def request_key(provider: str, model: str, instructions: str, input_text: str, schema: Dict[str, Any]) -> str:
    blob = json.dumps({"provider": provider, "model": model, "instructions": instructions,
                       "input": input_text, "schema": schema}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=1, ensure_ascii=False)
        tmp.replace(path)


class CachedProvider(LLMProvider):
    """Decorator adding the response cache to any provider."""

    def __init__(self, inner: LLMProvider, cache: ResponseCache):
        super().__init__(inner.model, inner.options)
        self.inner = inner
        self.name = inner.name
        self.cache = cache

    def _generate(self, instructions, input_text, schema):  # pragma: no cover - not used directly
        return self.inner._generate(instructions, input_text, schema)

    def structured_extract(self, input_data: Union[str, Dict[str, Any]], schema: Dict[str, Any],
                           instructions: str = "") -> LLMResult:
        text = input_data if isinstance(input_data, str) else json.dumps(input_data, ensure_ascii=False)
        key = request_key(self.name, self.model, instructions, text, schema)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResult(hit["data"], hit["raw_text"], self.name, self.model, hit.get("usage", {}),
                             hit.get("latency_s", 0.0), cached=True)
        result = self.inner.structured_extract(text, schema, instructions)
        self.cache.put(key, {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": self.name, "model": self.model, "instructions": instructions, "input": text,
            "schema": schema, "raw_text": result.raw_text, "data": result.data, "usage": result.usage,
            "latency_s": result.latency_s,
        })
        return result
