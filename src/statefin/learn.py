"""Turn review decisions into proposed configuration changes (dashboard Learn tab).

1. Alias candidates: labels whose mapping a person has confirmed but the rules cannot yet
   make - rows mapped by a reviewer, and LLM mappings a reviewer accepted. Promoting one
   adds it to variables.aliases (via config/overrides.yaml), so from then on the
   deterministic rules map it and the LLM is not needed for it.

2. Rule suggestions: rejected observations grouped by pattern -
     * the same normalised label rejected for the same variable -> a label exclude;
     * a word shared by the captions / section headings / column headers of rejected
       values and (almost) absent from accepted ones -> a context exclude for that variable
       or a column-exclude pattern.
   Each suggestion carries its evidence (supporting rejections, with examples) and its
   counter-evidence (accepted observations it would also remove), so a person decides.

Nothing is applied automatically.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Dict, List

from statefin.config import Config
from statefin.extraction.mapper import match_label

STOPWORDS = set("""a an and as at by for from in into of on or the to with table statement appendix annexure
chart contd continued concld figures crore lakh rupees rs inr state government total year years during
per cent percent report page source details particulars components""".split())


def _sid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", (text or "").lower()) if w not in STOPWORDS}


def _example(o: Dict[str, Any]) -> Dict[str, Any]:
    return {"observation_id": o["observation_id"], "label": o["raw_label"], "value": o["value"],
            "financial_year": o["financial_year"], "document": o["source_document"], "page": o["source_page"],
            "table": o.get("source_table")}


def alias_candidates(cfg: Config, observations: List[Dict[str, Any]], latest_reviews: Dict) -> List[Dict[str, Any]]:
    found: Dict[tuple, Dict[str, Any]] = {}
    for (ttype, tid), r in latest_reviews.items():
        if ttype == "row_label" and r["decision"] == "mapped":
            label = tid.split("|", 2)[2]
            key = (label, r["variable"])
            found.setdefault(key, {"label": label, "variable": r["variable"], "source": "reviewer mapped this row",
                                   "confirmed": True, "reviewer": r.get("reviewer"), "examples": []})
    for o in observations:
        if not o.get("mapping_method", "").startswith("llm:"):
            continue
        key = (o["raw_label"], o["canonical_variable"])
        confirmed = o.get("review_decision") in ("accepted", "corrected")
        entry = found.setdefault(key, {"label": o["raw_label"], "variable": o["canonical_variable"],
                                       "source": f"LLM mapping ({o['mapping_method'][4:]})",
                                       "confirmed": confirmed, "reviewer": o.get("reviewed_by"), "examples": []})
        entry["confirmed"] = entry["confirmed"] or confirmed
        if len(entry["examples"]) < 3:
            entry["examples"].append(_example(o))
    out = []
    for (label, var), e in found.items():
        if match_label(label, cfg).variable == var:
            continue                                  # the rules already map it
        e["normalized"] = cfg.normalize_label(label).text
        e["id"] = _sid("alias", var, e["normalized"])
        out.append(e)
    return sorted(out, key=lambda e: (not e["confirmed"], e["variable"], e["label"]))


def rule_suggestions(cfg: Config, observations: List[Dict[str, Any]], min_support: int = 2) -> List[Dict[str, Any]]:
    rejected = [o for o in observations if o.get("review_decision") == "rejected"]
    kept = [o for o in observations if o.get("review_decision") != "rejected"]
    suggestions: List[Dict[str, Any]] = []

    # 1. the same label rejected repeatedly for the same variable
    by_label = defaultdict(list)
    for o in rejected:
        by_label[(cfg.normalize_label(o["raw_label"]).text, o["canonical_variable"])].append(o)
    for (norm, var), group in by_label.items():
        if len(group) < min_support or not norm:
            continue
        pattern = "^" + re.escape(norm) + "$"
        if any(p.pattern == pattern for p in cfg.variables[var].label_excludes):
            continue
        counter = [o for o in kept if o["canonical_variable"] == var and cfg.normalize_label(o["raw_label"]).text == norm]
        suggestions.append({
            "id": _sid("label", var, pattern), "kind": "add_label_excludes", "variable": var, "value": pattern,
            "title": f"Stop mapping the label '{norm}' to {var}",
            "support": len(group), "counter": len(counter),
            "examples": [_example(o) for o in group[:4]], "counter_examples": [_example(o) for o in counter[:3]],
        })

    # 2. context words shared by rejected values of a variable
    for scope, field, kind in (("caption", "source_table", "add_context_excludes"),
                               ("section", "row_section", "add_context_excludes"),
                               ("column", "column_header", "add_column_exclude_patterns")):
        counts: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for o in rejected:
            for w in _words(o.get(field) or ""):
                counts[(w, o["canonical_variable"] if kind == "add_context_excludes" else "*")].append(o)
        for (word, var), group in counts.items():
            docs_tables = {(o["document_id"], o["table_id"]) for o in group}
            if len(group) < min_support or len(docs_tables) < 1:
                continue
            counter = [o for o in kept if word in _words(o.get(field) or "")
                       and (var == "*" or o["canonical_variable"] == var)]
            if len(counter) > len(group) // 4:       # would remove too many accepted values
                continue
            pattern = r"\b" + re.escape(word) + r"\b"
            existing = cfg.column_exclude_patterns if var == "*" else cfg.variables[var].context_excludes
            if any(p.pattern == pattern for p in existing):
                continue
            target = "any variable" if var == "*" else var
            suggestions.append({
                "id": _sid(scope, var, pattern), "kind": kind, "variable": None if var == "*" else var, "value": pattern,
                "title": (f"Skip year columns whose header contains '{word}'" if kind == "add_column_exclude_patterns"
                          else f"Do not take {target} from tables whose {scope} contains '{word}'"),
                "support": len(group), "counter": len(counter),
                "examples": [_example(o) for o in group[:4]], "counter_examples": [_example(o) for o in counter[:3]],
            })
    # strongest evidence first; drop word-level duplicates of a label rule on the same rows
    suggestions.sort(key=lambda s: (-(s["support"] - 2 * s["counter"]), s["kind"] != "add_label_excludes", s["title"]))
    return suggestions[:25]
