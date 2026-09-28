"""Identify a document's state, type and reference year from its text.

Detection is evidence-based and recorded: the manifest stores the match counts that led
to each decision. If the evidence is weak the field stays unknown rather than guessed; the
CLI's --state / --doc-type overrides always win and are recorded as overrides.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from statefin.config import Config, StateSpec
from statefin.extraction.parsers import canonical_fy, find_financial_years
from statefin.models import PageText
from statefin.pdf.text import normalize_for_search

TITLE_PAGES = 25       # document-type patterns are counted in the first N pages ...
FRONT_PAGES = 3        # ... and matches on the first few (title) pages count extra
FRONT_WEIGHT = 5


def detect_state(pages: List[PageText], cfg: Config, filename: str = "") -> Tuple[Optional[StateSpec], Dict[str, int]]:
    counts: Counter = Counter()
    text = filename + "\n" + "\n".join(p.text for p in pages)
    for st in cfg.states:
        n = len(st.pattern.findall(text))
        if n:
            counts[st.name] = n
    if not counts:
        return None, {}
    ranked = counts.most_common()
    top_name, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    # Require a clear margin: reports routinely mention other states in comparisons.
    if top >= 3 and top >= 3 * runner_up:
        return cfg.state_by_name(top_name), dict(ranked[:5])
    return None, dict(ranked[:5])


def detect_document_type(pages: List[PageText], cfg: Config, filename: str = "") -> Tuple[str, Dict[str, int]]:
    front = normalize_for_search(filename + "\n" + "\n".join(p.text for p in pages[:FRONT_PAGES]))
    rest = normalize_for_search("\n".join(p.text for p in pages[FRONT_PAGES:TITLE_PAGES]))
    scores = {code: sum(FRONT_WEIGHT * len(p.findall(front)) + len(p.findall(rest)) for p in spec.title_patterns)
              for code, spec in cfg.doc_types.items() if spec.title_patterns}
    scores = {k: v for k, v in scores.items() if v}
    if not scores:
        return "OTHER", {}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], cfg.doc_types[kv[0]].priority))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "OTHER", scores
    return ranked[0][0], scores


def detect_report_year(pages: List[PageText], cfg: Config, doc_type: str) -> Optional[str]:
    text = "\n".join(p.text for p in pages[:TITLE_PAGES])
    for pat in cfg.doc_type(doc_type).report_year_patterns:
        m = pat.search(text)
        if not m:
            continue
        token = m.group(1)
        fys = find_financial_years(token)
        if fys:
            return fys[0][0]
        if token.isdigit() and len(token) == 4:          # "year ended 31 March 2025"
            return canonical_fy(int(token) - 1)
    return None
