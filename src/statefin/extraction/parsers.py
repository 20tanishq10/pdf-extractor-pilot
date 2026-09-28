"""Deterministic parsers: numbers, financial years, units, column qualifiers, row labels.

These are pure functions with no I/O. Anything that is a document *convention* (unit
phrases, BE/RE qualifiers, sign-convention markers) is passed in from the compiled
configuration so that it stays editable in YAML.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Financial years
# --------------------------------------------------------------------------------------

_FY_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-–—/]\s*(\d{4}|\d{2})(?!\d)")


def canonical_fy(start_year: int) -> str:
    """2024 -> '2024-25'; 1999 -> '1999-00'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def fy_start_year(fy: str) -> int:
    return int(fy[:4])


def find_financial_years(text: str) -> List[Tuple[str, int, int]]:
    """Every valid financial-year token in `text` as (canonical, start, end).

    Accepts 2024-25, 2024–25, 2024-2025, 2024/25, "2016-\\n17". The second part must be
    the following year, so ranges such as "2015-2020" or dates are rejected.
    """
    out: List[Tuple[str, int, int]] = []
    for m in _FY_RE.finditer(text or ""):
        start = int(m.group(1))
        tail = m.group(2)
        ok = int(tail) == (start + 1) % 100 if len(tail) == 2 else int(tail) == start + 1
        if ok:
            out.append((canonical_fy(start), m.start(), m.end()))
    return out


def parse_financial_year(text: str) -> Optional[str]:
    """First financial year in `text`, canonicalised to 'YYYY-YY', or None."""
    found = find_financial_years(text)
    return found[0][0] if found else None


_FY_FILLER = re.compile(r"\b(?:fy|f\.y\.|year|years|financial|in|for|the|total|grand|revenue|capital|charged|voted"
                        r"|plan|non-plan|accounts?)\b", re.I)


def dominant_financial_year(text: str, qualifier_patterns: Iterable[Pattern] = ()) -> Optional[str]:
    """The FY if it is essentially the whole content of a header cell.

    "2024-25", "2024-25 (Actuals)", "FY 2023-24" -> FY.  "Change over 2005-06 for" or a
    chart legend containing several years -> None. This stops descriptive parent headers
    from being mistaken for year columns.
    """
    found = find_financial_years(text)
    if len(found) != 1:
        return None
    fy, s, e = found[0]
    rest = (text[:s] + " " + text[e:]).lower()
    for pat in qualifier_patterns:
        rest = pat.sub(" ", rest)
    rest = _FY_FILLER.sub(" ", rest)
    rest = re.sub(r"[\s()\[\]{}.,:;*#@\-–—/]+", "", rest)
    return fy if len(rest) <= 2 else None


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

MISSING_TOKENS = {
    "", "-", "--", "---", "—", "–", "−", "‐", "na", "n.a", "n.a.", "n/a", "nil", "..", "...",
    "…", "neg", "neg.", "negl", "negl.", "x", "*", "nr", "n.r.",
}

_NUM_BODY = re.compile(
    r"^(?:"
    r"\d{1,3}(?:,\d{3})+"            # western grouping 1,234,567
    r"|\d{1,2}(?:,\d{2})*,\d{3}"     # Indian (lakh) grouping 1,02,187
    r"|\d+"                          # ungrouped
    r")(?:\.\d+)?$"
    r"|^\.\d+$"
)
_UNIT_SUFFIX = re.compile(r"\s*\b(crores?|cr\.?|lakhs?|lacs?)\s*$", re.I)
_CURRENCY_PREFIX = re.compile(r"^\s*(?:₹|rs\.?|inr|rupees)\s*", re.I)
_FOOTNOTE_SUFFIX = re.compile(r"\s*[*#@$†‡^°®¹²³⁴⁵⁶⁷⁸⁹]+$")
_FOOTNOTE_LETTER = re.compile(r"(?<=\d)\s+\(?[a-z]\)?$")     # Finance Accounts: "40.79 a", "20,096.40 b"
_PAREN_SIGN = re.compile(r"^\(\s*([-–—−+])\s*\)\s*")
_MINUS_CHARS = "-–—−‐"
_SIGN_ONLY_TOKENS = {"-", "–", "—", "−", "‐", "(-)", "(–)", "(−)", "(+)"}


@dataclass
class ParsedNumber:
    raw: str
    value: Optional[float] = None
    status: str = "invalid"          # ok | missing | invalid | multi
    notes: List[str] = field(default_factory=list)
    unit_hint: Optional[str] = None  # "INR crore", "INR lakh", "INR", "percent"
    parts: List["ParsedNumber"] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _parse_single(s: str, raw: str) -> ParsedNumber:
    notes: List[str] = []
    unit_hint: Optional[str] = None

    m = _UNIT_SUFFIX.search(s)
    if m:
        unit_hint = "INR lakh" if m.group(1).lower().startswith("la") else "INR crore"
        s = s[: m.start()].strip()
    m = _CURRENCY_PREFIX.match(s)
    if m:
        unit_hint = unit_hint or "INR"
        s = s[m.end():].strip()
    if s.endswith("%"):
        unit_hint = "percent"
        s = s[:-1].strip()
    m = _FOOTNOTE_SUFFIX.search(s) or _FOOTNOTE_LETTER.search(s)
    if m and m.start() > 0:
        notes.append("footnote_marker")
        s = s[: m.start()].strip()

    negative = False
    m = _PAREN_SIGN.match(s)
    if m:  # "(-)6.04" / "(+)5.88" notation
        negative = m.group(1) != "+"
        s = s[m.end():]
    elif s[:1] in _MINUS_CHARS:
        negative = True
        rest = s[1:]
        if rest[:1].isspace():
            notes.append("split_sign")   # "- 11,265.73" or "-\n12,379.84"
        s = rest.strip()
    elif s[:1] == "+":
        s = s[1:].strip()
    elif len(s) > 2 and s[0] == "(" and s[-1] == ")":
        negative = True
        notes.append("parentheses_negative")
        s = s[1:-1].strip()

    if not _NUM_BODY.match(s):
        return ParsedNumber(raw=raw, status="invalid", notes=notes, unit_hint=unit_hint)
    value = float(s.replace(",", ""))
    return ParsedNumber(raw=raw, value=-value if negative else value, status="ok",
                        notes=notes, unit_hint=unit_hint)


def parse_number(raw: Optional[str]) -> ParsedNumber:
    """Parse one table cell into a number without guessing.

    Handles Indian lakh grouping (1,02,187.62), western grouping, ₹/Rs prefixes, crore/lakh
    suffixes, "(-)" notation, minus signs separated from digits, and trailing footnote
    markers. Placeholders ("--", "NA", "Nil") are `missing`, never zero. A cell holding
    several numbers is `multi` (the caller decides whether it can be split).
    """
    raw = raw if raw is not None else ""
    s = unicodedata.normalize("NFKC", raw).strip()
    s = re.sub(r"\s+", " ", s)
    if s.lower() in MISSING_TOKENS or s.lower().rstrip(".") in MISSING_TOKENS:
        return ParsedNumber(raw=raw, status="missing")

    single = _parse_single(s, raw)
    if single.ok:
        return single

    # "- 11,265.73" is one number; "21,210.01 25,224.89" is two.
    tokens = s.split(" ")
    if len(tokens) > 1:
        merged: List[str] = []
        for tok in tokens:
            if merged and merged[-1] in _SIGN_ONLY_TOKENS:
                merged[-1] = merged[-1] + " " + tok
            else:
                merged.append(tok)
        parts = [_parse_single(t, t) for t in merged]
        if len(parts) > 1 and all(p.ok for p in parts):
            return ParsedNumber(raw=raw, status="multi", parts=parts)
    return single


def is_number(text: Optional[str]) -> bool:
    return parse_number(text).ok


# --------------------------------------------------------------------------------------
# Units and column qualifiers
# --------------------------------------------------------------------------------------

@dataclass
class UnitMatch:
    unit: str
    kind: str      # currency | percent | other
    matched_text: str


def detect_unit(text: str, unit_rules: Sequence[Tuple[str, str, Sequence[Pattern]]]) -> Optional[UnitMatch]:
    """Earliest unit declaration in `text` according to the configured rules."""
    best: Optional[Tuple[int, UnitMatch]] = None
    for unit, kind, patterns in unit_rules:
        for pat in patterns:
            m = pat.search(text or "")
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), UnitMatch(unit=unit, kind=kind, matched_text=m.group(0).strip()))
    return best[1] if best else None


def detect_value_types(text: str, qualifier_rules: Dict[str, Sequence[Pattern]]) -> List[str]:
    """Column qualifiers (actual / budget_estimate / revised_estimate / ...) present in text."""
    found: List[str] = []
    for value_type, patterns in qualifier_rules.items():
        if any(p.search(text or "") for p in patterns):
            found.append(value_type)
    return found


# --------------------------------------------------------------------------------------
# Row labels
# --------------------------------------------------------------------------------------

_SERIAL_CELL = re.compile(r"^\(?\s*(?:\d{1,3}[a-z]?|[ivxlc]{1,5}|[a-z])\s*[.)]?\s*\)?$", re.I)
_LEADING_ENUM = re.compile(
    r"^\s*(?:\d{1,3}[a-z]?[.)]|\d{1,3}\s+(?=[A-Za-z])|\(\s*[ivxlc]{1,5}\s*\)|[ivxlc]{1,5}[.)]"
    r"|\(\s*[a-z0-9]{1,3}\s*\)|[a-z][.)])\s*",
    re.I,
)
_SIGN_MARKER = re.compile(r"\(\s*[-–—−+]\s*\)")
_INNER_GROUP = re.compile(r"[({]([^(){}]*)[)}]")
_FORMULA_CONTENT = re.compile(r"^[0-9ivxlcab-h\s.,]*[+\-–−*/][0-9ivxlcab-h\s+\-–−*/.,]*$|^\s*\d+\s*$", re.I)
_UNIT_PAREN = re.compile(r"\([^()]*?(?:crores?|lakhs?|per\s*cent|%|days)[^()]*\)", re.I)
# "Wages4", "payments#", and OCR renderings of superscripts: "Revenue!", "Excise?", "Services®"
_FOOTNOTE_MARKER = re.compile(r"(?<=[A-Za-z)])(\d{1,2}|[*#@$†‡^!?®°’'¹²³⁴⁵⁶⁷⁸⁹]+)\s*$")


def is_serial_cell(text: str) -> bool:
    """'1', '1.', '(i)', '6a', '(a)' - a row number rather than a label."""
    return bool(_SERIAL_CELL.match((text or "").strip()))


@dataclass
class NormalizedLabel:
    raw: str
    text: str                                # normalised form used for matching
    serial: Optional[str] = None             # leading "10." / "(i)" if present
    footnote_marker: Optional[str] = None    # "4" in "Salaries & Wages4", "#" in "Interest payments#"
    sign_convention: Optional[str] = None    # surplus_positive | deficit_positive | ambiguous
    unit: Optional[UnitMatch] = None         # unit stated inside the label, e.g. "(₹ in crore)"


def _strip_abbreviation_parens(s: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        inner = m.group(1)
        return " " if sum(ch.isupper() for ch in inner) >= 2 else m.group(0)
    return re.sub(r"\(\s*([A-Za-z]{2,6})\s*\)", repl, s)


def normalize_label(
    raw: str,
    sign_rules: Optional[Dict[str, Sequence[Pattern]]] = None,
    unit_rules: Optional[Sequence[Tuple[str, str, Sequence[Pattern]]]] = None,
) -> NormalizedLabel:
    """Normalise an accounting row label for matching, recording what was removed.

    "10. Revenue Deficit (-)/ Surplus (+) (4-10)" -> "revenue deficit / surplus"
    (serial "10.", sign_convention surplus_positive). The raw label is always kept.
    """
    s = unicodedata.normalize("NFKC", raw or "")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", s)     # "Grants-in-\naid"
    s = re.sub(r"\s+", " ", s).strip()
    out = NormalizedLabel(raw=raw or "", text="")

    if sign_rules:
        conventions = {name for name, pats in sign_rules.items() if any(p.search(s) for p in pats)}
        if len(conventions) == 1:
            out.sign_convention = conventions.pop()
        elif len(conventions) > 1:
            out.sign_convention = "ambiguous"

    if unit_rules:
        m = _UNIT_PAREN.search(s)
        if m:
            out.unit = detect_unit(m.group(0), unit_rules)
    s = _UNIT_PAREN.sub(" ", s)

    m = _LEADING_ENUM.match(s)
    while m and m.end() > 0 and m.end() < len(s):
        out.serial = out.serial or m.group(0).strip()
        s = s[m.end():]
        m = _LEADING_ENUM.match(s)

    s = _SIGN_MARKER.sub(" ", s)
    # Finance Accounts cross-references "(Ref. Statement 3 & 14)" and footnote letters "(a)"
    s = re.sub(r"\(\s*(?:ref|refer|see|vide)\b[^()]*\)", " ", s, flags=re.I)
    s = re.sub(r"\s\(\s*[a-z]\s*\)(?=\s|$)", " ", s)
    # Remove formula references innermost-first: "{(4+5+6)-14}", "(1+2+3)", "(i+ii)".
    for _ in range(5):
        new = _INNER_GROUP.sub(lambda g: " " if _FORMULA_CONTENT.match(g.group(1)) else g.group(0), s)
        if new == s:
            break
        s = new
    s = _strip_abbreviation_parens(s)
    s = re.sub(r"\s+", " ", s).strip()

    m = _FOOTNOTE_MARKER.search(s)
    if m:
        out.footnote_marker = m.group(1)
        s = s[: m.start()].strip()

    s = s.replace("&", " and ")
    s = s.lower().replace("’", "").replace("'", "").replace("‘", "")
    s = re.sub(r"[^\w/%\s]", " ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s+", " ", s).strip()
    out.text = s
    return out
