"""Find and rank pages likely to hold the target financial tables.

Scoring is deliberately simple and transparent: pages are rewarded for the number of
*distinct* variables they mention (so "pension + salary + revenue expenditure" beats many
repetitions of one generic term), and for looking like a table (year headers, a unit
declaration, a table caption, many numbers). Every page's score breakdown is saved in
candidates.json.
"""
from __future__ import annotations

import re
from typing import List

from statefin.config import Config
from statefin.extraction.parsers import find_financial_years
from statefin.models import CandidatePage, PageText
from statefin.pdf.text import normalize_for_search

_NUMERIC_TOKEN = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.,])-?\d+\.\d{1,2}(?![\d])")
_PLAIN_INT = re.compile(r"(?:\(-\))?-?\d{3,7}")      # scanned reports print "61575", "(-)14285"
_YEAR = re.compile(r"(?:19|20)\d{2}")
_CAPTION = re.compile(r"\b(?:table|appendix|annexure|statement)\s*[-–]?\s*\d+(?:\.\d+)*\s*[:.\-–]", re.I)


def score_page(page: PageText, cfg: Config) -> CandidatePage:
    text = normalize_for_search(page.text)
    w = cfg.get("candidates.weights", {})

    matched = {}
    for name, var in cfg.variables.items():
        hits = sum(len(p.findall(text)) for p in var.search_terms)
        if hits:
            matched[name] = hits
    core = [n for n in matched if cfg.variables[n].role == "core"]
    aux = [n for n in matched if cfg.variables[n].role != "core"]
    term_hits = sum(matched.values())
    fy_tokens = len(find_financial_years(page.text))
    numeric_tokens = len(_NUMERIC_TOKEN.findall(page.text)) + sum(
        1 for tok in page.text.split() if _PLAIN_INT.fullmatch(tok.strip("|][;:")) and not _YEAR.fullmatch(tok.strip("|][;:")))
    unit = cfg.detect_unit(page.text)
    unit_declared = unit.unit if unit and unit.kind == "currency" else None
    captions = [m.group(0).strip() for m in _CAPTION.finditer(page.text)]

    score = (
        len(core) * w.get("w_core", 10)
        + len(aux) * w.get("w_aux", 3)
        + min(term_hits, w.get("term_hits_cap", 20)) * w.get("w_term_hit", 0.5)
        + (w.get("w_years", 5) if fy_tokens >= 3 else 0)
        + (w.get("w_unit", 3) if unit_declared else 0)
        + min(len(captions), 3) * w.get("w_caption", 2)
        + (w.get("w_numbers", 4) if numeric_tokens >= 20 else 0)
    )
    return CandidatePage(
        page_number=page.page_number,
        page_label=page.page_label,
        score=round(float(score), 2),
        rank=0,
        is_candidate=False,
        matched_variables=matched,
        term_hits=term_hits,
        fy_tokens=fy_tokens,
        numeric_tokens=numeric_tokens,
        unit_declared=unit_declared,
        captions=captions,
    )


def rank_pages(pages: List[PageText], cfg: Config) -> List[CandidatePage]:
    """Score every page; return pages with any signal, best first, with candidates marked."""
    min_score = cfg.get("candidates.min_score", 12)
    min_numbers = cfg.get("candidates.min_numeric_tokens", 10)
    min_years = cfg.get("candidates.min_fy_tokens", 2)
    max_pages = cfg.get("candidates.max_pages", 80)

    scored = [score_page(p, cfg) for p in pages]
    scored = [c for c in scored if c.score > 0]
    scored.sort(key=lambda c: (-c.score, c.page_number))
    selected = 0
    for rank, cand in enumerate(scored, start=1):
        cand.rank = rank
        if (cand.score >= min_score and cand.numeric_tokens >= min_numbers
                and cand.fy_tokens >= min_years and selected < max_pages):
            cand.is_candidate = True
            selected += 1
    return scored
