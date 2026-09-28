"""User-adjustable settings (dashboard Settings tab) and their persistence in config/overrides.yaml.

Only settings listed in CATALOG can be changed from the dashboard; each has a type and a
valid range, so a typo cannot silently break a run. Every change is written to
config/overrides.yaml and logged in the store's config_changes table.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from statefin.config import Config, load_overrides, save_overrides

# key: dotted path into the merged config ("pipeline.x.y" -> pipeline.yaml, "llm.x" -> llm.yaml)
CATALOG: List[Dict[str, Any]] = [
    {"group": "Review and accuracy", "key": "pipeline.confidence.review_below", "type": "number",
     "min": 0.5, "max": 0.999, "step": 0.01, "label": "Accuracy threshold (auto-accept confidence)",
     "help": "Observations with confidence below this go to human review. Raise it for stricter review; "
             "values with warning or error flags always go to review regardless."},
    {"group": "Review and accuracy", "key": "pipeline.confidence.tiers.exact", "type": "number",
     "min": 0.5, "max": 1.0, "step": 0.01, "label": "Tier: exact match from",
     "help": "Confidence at or above this is labelled 'exact'."},
    {"group": "Review and accuracy", "key": "pipeline.confidence.tiers.strong", "type": "number",
     "min": 0.4, "max": 1.0, "step": 0.01, "label": "Tier: strong match from", "help": ""},
    {"group": "Review and accuracy", "key": "pipeline.confidence.tiers.ambiguous", "type": "number",
     "min": 0.1, "max": 1.0, "step": 0.01, "label": "Tier: ambiguous from",
     "help": "Below this an observation is labelled 'manual review'."},
    {"group": "Review and accuracy", "key": "pipeline.confidence.unit_missing", "type": "number",
     "min": 0.1, "max": 1.0, "step": 0.05, "label": "Confidence factor when no unit is declared", "help": ""},
    {"group": "Validation", "key": "pipeline.validation.yoy_max_change", "type": "number",
     "min": 0.05, "max": 5.0, "step": 0.05, "label": "Year-on-year change that is flagged (share)",
     "help": "0.5 flags changes above +/-50 %. Flags only; values are never changed."},
    {"group": "Validation", "key": "pipeline.validation.yoy_magnitude_jump", "type": "number",
     "min": 2, "max": 1000, "step": 1, "label": "Jump treated as a possible unit error (x)", "help": ""},
    {"group": "Validation", "key": "pipeline.validation.identity_abs_tolerance", "type": "number",
     "min": 0, "max": 100, "step": 0.5, "label": "Revenue identity tolerance (crore)", "help": ""},
    {"group": "Validation", "key": "pipeline.reconciliation.agreement_abs_tolerance", "type": "number",
     "min": 0, "max": 10, "step": 0.01, "label": "Minimum agreement tolerance between sources (crore)",
     "help": "Sources also agree within one unit of their last printed digit."},
    {"group": "Page and table detection", "key": "pipeline.candidates.min_score", "type": "number",
     "min": 0, "max": 200, "step": 1, "label": "Minimum page score to look for tables", "help": ""},
    {"group": "Page and table detection", "key": "pipeline.candidates.max_pages", "type": "number",
     "min": 1, "max": 2000, "step": 1, "label": "Maximum candidate pages per document", "help": ""},
    {"group": "Page and table detection", "key": "pipeline.tables.fallback_strategy", "type": "select",
     "options": ["wordgrid", "text", ""], "label": "Reader for unruled / scanned tables",
     "help": "wordgrid = word-position reader (recommended); text = PyMuPDF text strategy; empty = none."},
    {"group": "LLM", "key": "llm.primary.provider", "type": "select",
     "options": ["ollama", "gemini", "openai_compatible"], "label": "Primary provider", "help": ""},
    {"group": "LLM", "key": "llm.primary.model", "type": "text", "label": "Primary model", "help": ""},
    {"group": "LLM", "key": "llm.fallback.provider", "type": "select",
     "options": ["ollama", "gemini", "openai_compatible"], "label": "Fallback provider", "help": ""},
    {"group": "LLM", "key": "llm.fallback.model", "type": "text", "label": "Fallback model ('none' = no fallback)",
     "help": ""},
    {"group": "LLM", "key": "llm.policy.table_selection", "type": "select",
     "options": ["unmapped_relevant", "all_year_tables"], "label": "Which tables the LLM sees", "help": ""},
    {"group": "LLM", "key": "llm.policy.escalate_below_confidence", "type": "number",
     "min": 0, "max": 1, "step": 0.05, "label": "Escalate to fallback below confidence", "help": ""},
    {"group": "LLM", "key": "llm.policy.accept_llm_mappings_without_review", "type": "bool",
     "label": "Accept LLM-only mappings without review",
     "help": "Keep off until the benchmark shows LLM mappings are reliable."},
    {"group": "LLM", "key": "llm.policy.llm_confidence_cap", "type": "number",
     "min": 0.1, "max": 0.99, "step": 0.01, "label": "Confidence cap for LLM-only mappings", "help": ""},
]
_BY_KEY = {s["key"]: s for s in CATALOG}


def _get(d: Dict[str, Any], dotted: str) -> Any:
    node: Any = d
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _unset(d: Dict[str, Any], dotted: str) -> None:
    keys = dotted.split(".")
    stack = [d]
    for k in keys[:-1]:
        if k not in stack[-1]:
            return
        stack.append(stack[-1][k])
    stack[-1].pop(keys[-1], None)
    for node, k in zip(reversed(stack[:-1]), reversed(keys[:-1])):    # prune empty parents
        if node.get(k) == {}:
            node.pop(k)


def describe(cfg: Config) -> Dict[str, Any]:
    """Every setting with its default, current value and whether it is overridden."""
    import yaml                              # defaults = base files, without overrides
    defaults: Dict[str, Any] = {}
    for name in ("pipeline", "llm"):
        path = cfg.config_dir / f"{name}.yaml"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                defaults[name] = yaml.safe_load(fh) or {}
    overrides = load_overrides(cfg.config_dir)
    items = []
    for s in CATALOG:
        items.append({**s, "default": _get(defaults, s["key"]), "current": _get(cfg.raw, s["key"]),
                      "overridden": _get(overrides, s["key"]) is not None})
    return {"settings": items, "overrides_file": str(cfg.config_dir / "overrides.yaml")}


def validate(key: str, value: Any) -> Any:
    spec = _BY_KEY.get(key)
    if spec is None:
        raise ValueError(f"unknown setting {key!r}")
    if spec["type"] == "number":
        v = float(value)
        if not (spec["min"] <= v <= spec["max"]):
            raise ValueError(f"{spec['label']}: {v} is outside [{spec['min']}, {spec['max']}]")
        return int(v) if float(v).is_integer() and spec.get("step", 1) >= 1 else v
    if spec["type"] == "bool":
        return value in (True, "true", "1", 1, "on")
    if spec["type"] == "select":
        if value not in spec["options"]:
            raise ValueError(f"{spec['label']}: choose one of {spec['options']}")
        return value
    v = str(value).strip()
    if not v:
        raise ValueError(f"{spec['label']} cannot be empty")
    return v


def apply_changes(cfg: Config, changes: Dict[str, Any], resets: List[str], store=None,
                  reviewer: str = "reviewer", note: str = "") -> List[Dict[str, Any]]:
    """Validate and persist setting changes; returns [{key, old, new}]."""
    overrides = copy.deepcopy(load_overrides(cfg.config_dir))
    done = []
    for key, value in changes.items():
        new = validate(key, value)
        old = _get(cfg.raw, key)
        if new == old:
            continue
        _set(overrides, key, new)
        done.append({"key": key, "old": old, "new": new})
    for key in resets:
        if key not in _BY_KEY:
            raise ValueError(f"unknown setting {key!r}")
        if _get(overrides, key) is not None:
            done.append({"key": key, "old": _get(overrides, key), "new": "(default)"})
            _unset(overrides, key)
    t = cfg.raw.get("pipeline", {}).get("confidence", {}).get("tiers", {})
    tiers = {k: _get(overrides, f"pipeline.confidence.tiers.{k}") or t.get(k) for k in ("exact", "strong", "ambiguous")}
    if tiers["exact"] and tiers["strong"] and tiers["ambiguous"] and not (tiers["exact"] >= tiers["strong"] >= tiers["ambiguous"]):
        raise ValueError("tiers must satisfy exact >= strong >= ambiguous")
    if done:
        save_overrides(cfg.config_dir, overrides)
        if store is not None:
            for d in done:
                store.log_config_change("setting", d["key"], d["old"], d["new"], reviewer, note)
    return done


def add_rule(cfg: Config, kind: str, variable: Optional[str], value: str, store=None,
             reviewer: str = "reviewer", note: str = "") -> bool:
    """Append a learned rule (alias, label/context exclude, column/caption pattern) to overrides."""
    overrides = copy.deepcopy(load_overrides(cfg.config_dir))
    if kind in ("add_aliases", "add_label_excludes", "add_context_excludes"):
        if variable not in cfg.variables:
            raise ValueError(f"unknown variable {variable!r}")
        bucket = overrides.setdefault("variables", {}).setdefault(kind, {}).setdefault(variable, [])
    elif kind == "add_global_label_excludes":
        bucket = overrides.setdefault("variables", {}).setdefault(kind, [])
    elif kind in ("add_column_exclude_patterns", "add_projection_caption_patterns"):
        bucket = overrides.setdefault("source_rules", {}).setdefault(kind, [])
    else:
        raise ValueError(f"unknown rule kind {kind!r}")
    if kind != "add_aliases":
        import re
        re.compile(value)                     # refuse an invalid regular expression
    if value in bucket:
        return False
    bucket.append(value)
    save_overrides(cfg.config_dir, overrides)
    if store is not None:
        store.log_config_change("rule", f"{kind}:{variable or '*'}", None, value, reviewer, note)
    return True
