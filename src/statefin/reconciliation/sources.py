"""Group observations of the same state x year x variable and reconcile them.

Rules (never averaging, never silently choosing):
  * values within `agreement_abs_tolerance` of each other agree (rounding noise);
  * if all observations agree, the cell is `consistent` and each observation records the
    others as corroboration (a small confidence bonus when it comes from another table);
  * if they disagree and exactly one agreeing group contains the highest-priority source
    (source_rules.yaml), the cell is `resolved_by_priority`; the lower-priority
    observations are flagged as a discrepancy and go to review;
  * otherwise the cell is `conflict`: every observation is flagged and the analytical
    panel leaves the cell empty until a human decides.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from statefin.config import Config
from statefin.extraction.mapper import confidence_tier
from statefin.models import Observation

CellKey = Tuple[str, str, str]   # (state, financial_year, canonical_variable)


@dataclass
class CellConsensus:
    state: str
    state_code: Optional[str]
    financial_year: str
    variable: str
    status: str                                  # single | consistent | resolved_by_priority | conflict
    value: Optional[float]
    representative: Optional[Observation]
    observations: List[Observation]
    agreeing: List[Observation] = field(default_factory=list)
    note: str = ""
    variant: str = ""                            # definition variant of this cell ("" = canonical)
    other_variants: Dict[str, "CellConsensus"] = field(default_factory=dict)


def cell_key(o: Observation) -> CellKey:
    return (o.state or "UNKNOWN", o.financial_year, o.canonical_variable)


_OPPOSITE = {"surplus_positive": "deficit_positive", "deficit_positive": "surplus_positive"}


def _num(o: Observation) -> float:
    """Value in INR crore when the unit is known (lakh / 100), else as printed."""
    return o.value_crore if o.value_crore is not None else o.value


def _comparable(o: Observation, canonical: Optional[str]) -> float:
    """Crore value in the canonical sign convention, for comparison only (never stored)."""
    if canonical and o.sign_convention == _OPPOSITE.get(canonical):
        return -_num(o)
    return _num(o)


def _infer_sign_conventions(obs: List[Observation], canonical: Optional[str], tol: float) -> None:
    """Deficits printed without a sign marker take the convention implied by marked observations.

    Same magnitude and same sign as a marked value -> same convention; same magnitude and
    opposite sign (and no same-sign match) -> the opposite convention. Values are untouched.
    """
    if not canonical:
        return
    marked = [o for o in obs if o.sign_convention in _OPPOSITE]
    for o in obs:
        if o.sign_convention != "unspecified" or abs(_num(o)) <= tol:
            continue
        same = [p for p in marked if _agree_values(_comparable(p, canonical), _num(o), p, o, tol)]
        opposite = [p for p in marked if _agree_values(_comparable(p, canonical), -_num(o), p, o, tol)]
        if same and not opposite:
            o.sign_convention = canonical
            o.add_flag("sign_convention_inferred", "info",
                       f"no sign marker in label; equals {same[0].observation_id} ({same[0].sign_convention}), "
                       f"so reported as '{canonical}'")
        elif opposite and not same:
            o.sign_convention = _OPPOSITE[canonical]
            o.add_flag("sign_convention_inferred", "info",
                       f"no sign marker in label; same magnitude as {opposite[0].observation_id} with opposite "
                       f"sign, so reported as '{o.sign_convention}'. Value kept as printed")


def printed_decimals(raw_value_text: str) -> int:
    digits = "".join(ch for ch in raw_value_text if ch.isdigit() or ch == ".")
    return len(digits.rsplit(".", 1)[1]) if "." in digits else 0


def _half_unit(o: Observation) -> float:
    """One unit of the last printed digit, in crore: publications round or truncate
    (18,152 vs 18,153), and a lakh figure's 0.01 is 0.0001 crore."""
    scale = (o.value_crore / o.value) if (o.value_crore is not None and o.value) else 1.0
    return 1.0 * 10 ** -printed_decimals(o.raw_value_text) * abs(scale)


def _agree_values(x: float, y: float, a: Observation, b: Observation, tol: float) -> bool:
    return abs(x - y) <= max(tol, _half_unit(a), _half_unit(b)) + 1e-9


def _agree(a: Observation, b: Observation, tol: float, canonical: Optional[str]) -> bool:
    """Equal within rounding: tolerance is the config value or half a unit of the last digit
    printed by the less precise of the two (4,754 agrees with 4,753.92; 4,754 does not with 4,755.2).
    Values are compared in crore, so a Finance Accounts lakh figure meets an SFAR crore figure."""
    return _agree_values(_comparable(a, canonical), _comparable(b, canonical), a, b, tol)


def _clusters(obs: List[Observation], tol: float, canonical: Optional[str]) -> List[List[Observation]]:
    groups: List[List[Observation]] = []
    for o in sorted(obs, key=lambda o: _comparable(o, canonical)):
        if groups and all(_agree(o, m, tol, canonical) for m in groups[-1]):
            groups[-1].append(o)
        else:
            groups.append([o])
    return groups


def _representative(group: List[Observation], canonical: Optional[str]) -> Observation:
    # Within an agreeing group prefer an observation printed in the canonical convention,
    # then the preferred source, then confidence.
    return sorted(group, key=lambda o: (bool(canonical) and o.sign_convention != canonical, o.source_priority,
                                        -o.confidence, o.source_page, o.observation_id))[0]


def _fmt(o: Observation) -> str:
    unit = "" if o.unit in (None, "INR crore") else f" {o.unit} (= {_num(o):,.2f} crore)"
    return f"{o.value:,.2f}{unit} ({o.source_document} p.{o.source_page} {o.source_table or o.table_id}; id {o.observation_id})"


def _reconcile_group(key: CellKey, obs: List[Observation], cfg: Config, tol: float, bonus: float) -> CellConsensus:
    canonical = cfg.variables[key[2]].canonical_sign_convention
    _infer_sign_conventions(obs, canonical, tol)
    clusters = _clusters(obs, tol, canonical)
    chosen: Optional[List[Observation]] = None
    if len(clusters) == 1:
        chosen = clusters[0]
        status = "single" if len(obs) == 1 else "consistent"
    else:
        top = min(o.source_priority for o in obs)
        top_clusters = [c for c in clusters if min(o.source_priority for o in c) == top]
        if len(top_clusters) == 1:
            chosen, status = top_clusters[0], "resolved_by_priority"
        else:
            status = "conflict"

    chosen_ids = {id(o) for o in chosen or []}
    for o in obs:
        o.reconciliation_status = status
        same = next(c for c in clusters if any(o is p for p in c))
        same_ids = {id(p) for p in same}
        o.corroborated_by = [p.observation_id for p in same if p is not o]
        o.conflicts_with = [p.observation_id for p in obs if id(p) not in same_ids]
        if any(p.table_id != o.table_id or p.document_id != o.document_id for p in same if p is not o):
            o.confidence = round(min(1.0, o.confidence + bonus), 3)
            o.confidence_tier = confidence_tier(o.confidence, cfg)
            o.confidence_reasons.append(f"corroborated by {len(same) - 1} other observation(s)")

    rep = _representative(chosen, canonical) if chosen else None
    if status == "resolved_by_priority" and rep:
        for o in obs:
            if id(o) not in chosen_ids:
                o.add_flag("source_discrepancy", "warning",
                           f"value {o.value:,.2f} differs from higher-priority source value {_fmt(rep)}; "
                           "panel uses the higher-priority source")
            else:
                o.add_flag("source_discrepancy", "info",
                           "lower-priority source(s) report a different value: "
                           + "; ".join(_fmt(p) for p in obs if id(p) not in chosen_ids))
    elif status == "conflict":
        for o in obs:
            others = "; ".join(_fmt(p) for p in obs if p.observation_id in o.conflicts_with)
            why = ""
            profiles = {(tuple(p.definition_tags), p.series_base) for p in obs}
            if len(profiles) > 1:
                why = ("; the values use different definitions/series: "
                       + " vs ".join(f"[{', '.join(t) or 'no tags'}{'; base ' + b if b else ''}]" for t, b in sorted(profiles, key=str)))
            o.add_flag("conflict", "warning",
                       f"same-priority sources disagree: this {o.value:,.2f} vs {others}; not resolved automatically{why}")

    return CellConsensus(
        state=key[0], state_code=obs[0].state_code, financial_year=key[1], variable=key[2],
        status=status, value=_num(rep) if rep else None, representative=rep,
        observations=obs, agreeing=list(chosen or []),
    )


def reconcile(observations: List[Observation], cfg: Config) -> Dict[CellKey, CellConsensus]:
    tol = float(cfg.get("reconciliation.agreement_abs_tolerance", 0.05))
    bonus = float(cfg.get("confidence.corroboration_bonus", 0.03))
    groups: Dict[CellKey, List[Observation]] = defaultdict(list)
    for o in observations:
        groups[cell_key(o)].append(o)

    cells: Dict[CellKey, CellConsensus] = {}
    for key, obs in sorted(groups.items()):
        # Observations measuring a different quantity (salaries only; object-head pensions)
        # are reconciled separately; the panel cell uses the canonical definition if present.
        by_variant: Dict[str, List[Observation]] = defaultdict(list)
        for o in obs:
            by_variant[o.definition_variant or ""].append(o)
        results = {v: _reconcile_group(key, group, cfg, tol, bonus) for v, group in by_variant.items()}
        selected = "" if "" in results else min(results, key=lambda v: (min(o.source_priority for o in by_variant[v]), v))
        cell = results[selected]
        cell.variant = selected
        cell.other_variants = {v: c for v, c in results.items() if v != selected}
        if selected and cell.representative is not None:
            cell.representative.add_flag("definition_variant_in_panel", "warning",
                                         f"no observation with the canonical definition; the panel uses variant '{selected}'")
        for v, other in cell.other_variants.items():
            for o in other.observations:
                o.add_flag("definition_variant", "info",
                           f"measures '{v}', a different definition; the panel uses "
                           + (f"'{selected}'" if selected else "the canonical definition")
                           + (f" ({_fmt(cell.representative)})" if cell.representative else ""))
        cells[key] = cell
    return cells
