"""LLM settings: config/llm.yaml, overridable by environment variables and CLI flags.

Environment overrides (useful for comparing models on the benchmark without editing files):
    STATEFIN_LLM_PROVIDER, STATEFIN_LLM_MODEL
    STATEFIN_LLM_FALLBACK_PROVIDER, STATEFIN_LLM_FALLBACK_MODEL   (model "none" disables the fallback)
Secrets are read only from the environment (e.g. GEMINI_API_KEY), which may be populated
from a git-ignored .env file in the project root. Keys are never written to outputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from statefin.llm.base import LLMConfigurationError, LLMProvider
from statefin.llm.cache import CachedProvider, ResponseCache
from statefin.llm.providers import PROVIDERS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


@dataclass
class ModelSpec:
    provider: str
    model: str
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class LLMSettings:
    primary: ModelSpec
    fallback: Optional[ModelSpec]
    table_selection: str = "unmapped_relevant"
    max_tables_per_document: int = 40
    max_rows_per_table: int = 80
    escalate_below_confidence: float = 0.75
    accept_without_review: bool = False
    confidence_cap: float = 0.79
    accepted_confidence_factor: float = 0.95
    cache_dir: str = "llm_cache"


def _spec(raw: Mapping[str, Any], providers: Mapping[str, Any]) -> ModelSpec:
    provider = raw["provider"]
    if provider not in PROVIDERS:
        raise LLMConfigurationError(f"unknown LLM provider '{provider}'; known: {', '.join(PROVIDERS)}")
    options = {**(providers.get(provider) or {}), **{k: v for k, v in raw.items() if k not in ("provider", "model")}}
    return ModelSpec(provider, str(raw["model"]), options)


def load_llm_settings(raw: Mapping[str, Any], env: Mapping[str, str] = os.environ,
                      overrides: Optional[Dict[str, Optional[str]]] = None) -> LLMSettings:
    """Build settings from the parsed llm.yaml, then env vars, then explicit overrides (CLI)."""
    raw = dict(raw or {})
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    providers = raw.get("providers", {}) or {}
    primary = dict(raw.get("primary") or {"provider": "gemini", "model": "gemini-2.5-flash-lite"})
    fallback = dict(raw["fallback"]) if raw.get("fallback") else None

    for key, env_name in (("provider", "STATEFIN_LLM_PROVIDER"), ("model", "STATEFIN_LLM_MODEL")):
        value = overrides.get(f"primary_{key}") or env.get(env_name)
        if value:
            primary[key] = value
    for key, env_name in (("provider", "STATEFIN_LLM_FALLBACK_PROVIDER"), ("model", "STATEFIN_LLM_FALLBACK_MODEL")):
        value = overrides.get(f"fallback_{key}") or env.get(env_name)
        if value:
            fallback = dict(fallback or {"provider": primary["provider"]})
            fallback[key] = value
    if fallback and (str(fallback.get("model", "")).lower() == "none" or overrides.get("no_fallback")):
        fallback = None

    policy = raw.get("policy", {}) or {}
    return LLMSettings(
        primary=_spec(primary, providers),
        fallback=_spec(fallback, providers) if fallback else None,
        table_selection=overrides.get("table_selection") or policy.get("table_selection", "unmapped_relevant"),
        max_tables_per_document=int(policy.get("max_tables_per_document", 40)),
        max_rows_per_table=int(policy.get("max_rows_per_table", 80)),
        escalate_below_confidence=float(policy.get("escalate_below_confidence", 0.75)),
        accept_without_review=bool(policy.get("accept_llm_mappings_without_review", False)),
        confidence_cap=float(policy.get("llm_confidence_cap", 0.79)),
        accepted_confidence_factor=float(policy.get("accepted_confidence_factor", 0.95)),
        cache_dir=str(raw.get("cache_dir", "llm_cache")),
    )


def build_provider(spec: ModelSpec, cache: Optional[ResponseCache] = None) -> LLMProvider:
    provider = PROVIDERS[spec.provider](spec.model, spec.options)
    return CachedProvider(provider, cache) if cache else provider
