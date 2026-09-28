"""Load and compile the YAML configuration.

All document conventions (vocabulary, units, qualifiers, source hierarchy, thresholds) live
in config/*.yaml. This module compiles them once and exposes small helpers. Each stage of
the pipeline hashes only the config sections it depends on, so editing, for example, the
label aliases re-runs mapping without re-parsing any PDF.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

import yaml

from statefin.extraction.parsers import NormalizedLabel, UnitMatch, detect_unit, detect_value_types, normalize_label

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _rx(pattern: str) -> Pattern:
    return re.compile(pattern, re.I)


OVERRIDES_FILE = "overrides.yaml"
OVERRIDES_HEADER = """# Managed by the dashboard (Settings and Learn tabs); every change is also logged in
# data/statefin.db (config_changes). Merged over pipeline.yaml, llm.yaml, variables.yaml and
# source_rules.yaml, which are never rewritten. Delete an entry to return to the default.
"""


def load_overrides(config_dir: Path) -> Dict[str, Any]:
    path = Path(config_dir) / OVERRIDES_FILE
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_overrides(config_dir: Path, data: Dict[str, Any]) -> Path:
    path = Path(config_dir) / OVERRIDES_FILE
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(OVERRIDES_HEADER)
        yaml.safe_dump(data, fh, sort_keys=True, allow_unicode=True)
    tmp.replace(path)
    return path


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def apply_overrides(raw: Dict[str, Dict[str, Any]], ov: Dict[str, Any]) -> None:
    """Settings are deep-merged; learned rules are appended (never replacing base rules)."""
    _deep_merge(raw.setdefault("pipeline", {}), ov.get("pipeline") or {})
    _deep_merge(raw.setdefault("llm", {}), ov.get("llm") or {})
    variables = raw.get("variables", {}).get("variables", {})
    vov = ov.get("variables") or {}
    for key, field_name in (("add_aliases", "aliases"), ("add_label_excludes", "label_excludes"),
                            ("add_context_excludes", "context_excludes")):
        for var, items in (vov.get(key) or {}).items():
            if var in variables:
                existing = variables[var].setdefault(field_name, []) or []
                variables[var][field_name] = existing + [i for i in items if i not in existing]
    raw["variables"]["global_label_excludes"] = (raw["variables"].get("global_label_excludes") or []) + list(
        vov.get("add_global_label_excludes") or [])
    sov = ov.get("source_rules") or {}
    for key in ("column_exclude_patterns", "projection_caption_patterns"):
        extra = sov.get(f"add_{key}") or []
        if extra:
            raw["source_rules"][key] = (raw["source_rules"].get(key) or []) + list(extra)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


_PACKAGE_DIR = Path(__file__).resolve().parent


def code_fingerprint(*modules: str) -> str:
    """Hash of the source of the given modules (paths relative to the package), so a code
    change invalidates exactly the cached stages that depend on it."""
    h = hashlib.sha256()
    for rel in sorted(modules):
        h.update(rel.encode())
        h.update((_PACKAGE_DIR / rel).read_bytes())
    return h.hexdigest()[:16]


@dataclass
class LabelRule:
    kind: str                         # alias | pattern
    source: str                       # alias text or regex as written in YAML
    confidence: Optional[float] = None
    note: Optional[str] = None
    regex: Optional[Pattern] = None


@dataclass
class DefinitionRule:
    tag: str
    regex: Pattern
    where: Tuple[str, ...]
    substantive: bool = False     # compared across years for structural breaks
    variant: bool = False         # measures a different quantity: reconciled separately


@dataclass
class VariableSpec:
    name: str
    role: str
    description: str
    unit_kind: str
    allow_negative: bool
    canonical_sign_convention: Optional[str]
    search_terms: List[Pattern]
    aliases: Dict[str, LabelRule]     # normalised alias -> rule
    patterns: List[LabelRule]
    plausible_range: Optional[Tuple[float, float]]
    derive_from: List[str] = field(default_factory=list)
    context_excludes: List[Pattern] = field(default_factory=list)   # caption/section text that rules the variable out
    label_excludes: List[Pattern] = field(default_factory=list)     # normalised labels never mapped to this variable
    definition_rules: List[DefinitionRule] = field(default_factory=list)
    major_heads: List[str] = field(default_factory=list)                 # e.g. ["2071"]
    preferred_components: List[str] = field(default_factory=lambda: ["total"])


@dataclass
class DocTypeSpec:
    code: str
    priority: int
    label: str
    title_patterns: List[Pattern]
    report_year_patterns: List[Pattern]
    unqualified_column_confidence: float


@dataclass
class StateSpec:
    name: str
    code: str
    aliases: List[str]
    first_fy: Optional[str]
    notes: Optional[str]
    pattern: Pattern
    breaks: List[Dict[str, str]] = field(default_factory=list)   # [{fy, reason}] reorganisations


class Config:
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.raw: Dict[str, Dict[str, Any]] = {}
        for name in ("variables", "source_rules", "pipeline", "states"):
            path = self.config_dir / f"{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Missing configuration file: {path}")
            with open(path, encoding="utf-8") as fh:
                self.raw[name] = yaml.safe_load(fh) or {}

        llm_path = self.config_dir / "llm.yaml"          # optional: only needed with --llm
        if llm_path.exists():
            with open(llm_path, encoding="utf-8") as fh:
                self.raw["llm"] = yaml.safe_load(fh) or {}
        else:
            self.raw["llm"] = {}
        self.overrides = load_overrides(self.config_dir)
        apply_overrides(self.raw, self.overrides)

        rules = self.raw["source_rules"]
        self.unit_rules: List[Tuple[str, str, List[Pattern]]] = [
            (u["unit"], u.get("kind", "other"), [_rx(p) for p in u["patterns"]]) for u in rules["units"]
        ]
        self.qualifier_rules: Dict[str, List[Pattern]] = {
            k: [_rx(p) for p in v] for k, v in rules["column_qualifiers"].items()
        }
        self.all_qualifier_patterns: List[Pattern] = [p for ps in self.qualifier_rules.values() for p in ps]
        self.accepted_value_types = set(rules.get("accepted_value_types", ["actual"]))
        self.column_exclude_patterns = [_rx(p) for p in rules.get("column_exclude_patterns", [])]
        self.caption_patterns = [_rx(p) for p in rules.get("caption_patterns", [])]
        self.cited_source_patterns = [_rx(p) for p in rules.get("cited_source_patterns", [])]
        self.series_base_patterns = [_rx(p) for p in rules.get("series_base_patterns", [])]
        self.component_rules: Dict[str, List[Pattern]] = {
            k: [_rx(p) for p in v] for k, v in (rules.get("column_components") or {}).items()}
        self.projection_caption_patterns = [_rx(p) for p in rules.get("projection_caption_patterns", [])]
        self.sign_rules: Dict[str, List[Pattern]] = {
            k: [_rx(p) for p in v] for k, v in rules.get("sign_conventions", {}).items()
        }
        self.doc_types: Dict[str, DocTypeSpec] = {}
        for code, spec in rules["document_types"].items():
            self.doc_types[code] = DocTypeSpec(
                code=code,
                priority=int(spec["priority"]),
                label=spec.get("label", code),
                title_patterns=[_rx(p) for p in spec.get("title_patterns", [])],
                report_year_patterns=[_rx(p) for p in spec.get("report_year_patterns", [])],
                unqualified_column_confidence=float(spec.get("unqualified_column_confidence", 0.85)),
            )

        vdoc = self.raw["variables"]
        self.global_label_excludes = [_rx(p) for p in vdoc.get("global_label_excludes", [])]
        self.variables: Dict[str, VariableSpec] = {}
        for name, spec in vdoc["variables"].items():
            aliases: Dict[str, LabelRule] = {}
            for a in spec.get("aliases", []) or []:
                entry = a if isinstance(a, dict) else {"label": a}
                norm = self.normalize_label(entry["label"]).text
                aliases[norm] = LabelRule("alias", entry["label"], entry.get("confidence"), entry.get("note"))
            patterns = []
            for p in spec.get("patterns", []) or []:
                entry = p if isinstance(p, dict) else {"regex": p}
                patterns.append(LabelRule("pattern", entry["regex"], entry.get("confidence"),
                                          entry.get("note"), re.compile(entry["regex"], re.I)))
            rng = spec.get("plausible_range_crore")
            self.variables[name] = VariableSpec(
                name=name,
                role=spec.get("role", "core"),
                description=spec.get("description", ""),
                unit_kind=spec.get("unit_kind", "currency"),
                allow_negative=bool(spec.get("allow_negative", False)),
                canonical_sign_convention=spec.get("canonical_sign_convention"),
                search_terms=[re.compile(r"\b" + t + r"\b", re.I) for t in spec.get("search_terms", []) or []],
                aliases=aliases,
                patterns=patterns,
                plausible_range=(float(rng[0]), float(rng[1])) if rng else None,
                derive_from=list(spec.get("derive_from", []) or []),
                context_excludes=[re.compile(p, re.I) for p in spec.get("context_excludes", []) or []],
                label_excludes=[re.compile(p, re.I) for p in spec.get("label_excludes", []) or []],
                definition_rules=[DefinitionRule(r["tag"], re.compile(r["pattern"], re.I), tuple(r.get("where", ["label"])),
                                                 bool(r.get("substantive")), bool(r.get("variant")))
                                  for r in spec.get("definition_rules", []) or []],
                major_heads=[str(m) for m in spec.get("major_heads", []) or []],
                preferred_components=list(spec.get("preferred_components", []) or ["total"]),
            )

        self.states: List[StateSpec] = []
        for st in self.raw["states"]["states"]:
            names = [st["name"]] + list(st.get("aliases", []) or [])
            alt = "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in sorted(names, key=len, reverse=True))
            self.states.append(StateSpec(
                name=st["name"], code=st["code"], aliases=list(st.get("aliases", []) or []),
                first_fy=st.get("first_fy"), notes=st.get("notes"),
                pattern=re.compile(r"\b(?:" + alt + r")\b", re.I),
                breaks=[{"fy": str(b["fy"]), "reason": b.get("reason", "")} for b in st.get("breaks", []) or []],
            ))

        self.pipeline: Dict[str, Any] = self.raw["pipeline"]

    # ------------------------------------------------------------------ helpers
    def section_hash(self, *keys: str) -> str:
        """Hash of named config sections, e.g. section_hash('variables', 'pipeline.confidence')."""
        parts = {}
        for key in keys:
            node: Any = self.raw
            for k in key.split("."):
                node = node.get(k, {}) if isinstance(node, dict) else {}
            parts[key] = node
        return stable_hash(parts)

    def normalize_label(self, raw: str) -> NormalizedLabel:
        return normalize_label(raw, getattr(self, "sign_rules", None), getattr(self, "unit_rules", None))

    def detect_unit(self, text: str) -> Optional[UnitMatch]:
        return detect_unit(text, self.unit_rules)

    CRORE_FACTORS = {"INR crore": 1.0, "INR lakh": 0.01, "INR thousand": 0.0001}

    def crore_factor(self, unit: Optional[str]) -> Optional[float]:
        """Multiplier from `unit` to INR crore (1 crore = 100 lakh = 10,000 thousand)."""
        return self.CRORE_FACTORS.get(unit or "")

    def unit_kind(self, unit: Optional[str]) -> Optional[str]:
        for name, kind, _ in self.unit_rules:
            if name == unit:
                return kind
        return None

    def value_types(self, text: str) -> List[str]:
        return detect_value_types(text, self.qualifier_rules)

    def column_component(self, text: str) -> Optional[str]:
        t = re.sub(r"\s+", " ", re.sub(r"[()\[\]:*]", " ", text or "")).strip().lower()
        for name, pats in getattr(self, "component_rules", {}).items():
            if any(p.search(t) for p in pats):
                return name
        return None

    def variant_tags(self, variable: str) -> set:
        return {r.tag for r in self.variables[variable].definition_rules if r.variant}

    def substantive_tags(self, variable: str) -> set:
        return {r.tag for r in self.variables[variable].definition_rules if r.substantive}

    def label_exclusion(self, normalized_text: str) -> Optional[str]:
        for pat in self.global_label_excludes:
            if pat.search(normalized_text):
                return pat.pattern
        return None

    def doc_type(self, code: Optional[str]) -> DocTypeSpec:
        return self.doc_types.get(code or "OTHER", self.doc_types["OTHER"])

    def state_by_name(self, name: str) -> Optional[StateSpec]:
        for st in self.states:
            if st.pattern.fullmatch(name.strip()):
                return st
        return None

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.pipeline
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def core_variables(self) -> List[str]:
        return [n for n, v in self.variables.items() if v.role == "core"]

    @property
    def panel_variables(self) -> List[str]:
        return [n for n, v in self.variables.items() if v.role in ("core", "optional")]
