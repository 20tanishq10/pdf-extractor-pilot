"""Accounting identities and cross-variable ratio checks, run on reconciled cell values.

Failures are flagged on the observation that feeds the panel (the cell's representative)
and never alter a value. When the revenue identity holds under the canonical convention
for a deficit whose label had no sign marker, the convention is recorded as inferred.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from statefin.config import Config
from statefin.reconciliation.sources import CellConsensus, CellKey


def _cell(by_year: Dict[str, CellConsensus], var: str) -> Optional[CellConsensus]:
    cell = by_year.get(var)
    return cell if cell and cell.value is not None and cell.representative else None


def check_identities(cells: Dict[CellKey, CellConsensus], cfg: Config) -> None:
    abs_tol = float(cfg.get("validation.identity_abs_tolerance", 1.0))
    rel_tol = float(cfg.get("validation.identity_rel_tolerance", 0.001))
    by_state_year: Dict[tuple, Dict[str, CellConsensus]] = defaultdict(dict)
    for (state, fy, var), cell in cells.items():
        by_state_year[(state, fy)][var] = cell

    for (state, fy), by_var in by_state_year.items():
        rr, re_, rd = (_cell(by_var, v) for v in ("revenue_receipts", "revenue_expenditure", "revenue_deficit_surplus"))
        if rr and re_ and rd:
            expected = rr.value - re_.value
            tol = max(abs_tol, rel_tol * abs(re_.value))
            rep = rd.representative
            fits_sp = abs(rd.value - expected) <= tol
            fits_dp = abs(-rd.value - expected) <= tol
            conv = rep.sign_convention or "unspecified"
            detail = (f"RR {rr.value:,.2f} - RE {re_.value:,.2f} = {expected:,.2f}; reported {rd.value:,.2f} "
                      f"(tolerance {tol:,.2f})")
            if conv == "surplus_positive":
                ok = fits_sp
            elif conv == "deficit_positive":
                ok = fits_dp
                detail += "; compared with sign reversed because the label is deficit-positive"
            else:
                ok = fits_sp or fits_dp
                if fits_sp and not fits_dp:
                    for o in rd.agreeing:
                        if o.sign_convention == "unspecified":
                            o.sign_convention = "surplus_positive"
                            o.add_flag("sign_convention_inferred", "info",
                                       "convention 'surplus_positive' inferred from the revenue identity")
            if ok:
                rep.add_flag("identity_revenue_balance", "info", "holds: " + detail)
            else:
                rep.add_flag("identity_revenue_balance", "warning",
                             "FAILS: " + detail + f" [RR id {rr.representative.observation_id}, "
                             f"RE id {re_.representative.observation_id}]")

        if re_:
            parts = [(_cell(by_var, v), v) for v in ("salary_wages", "pension_expenditure", "interest_payments")]
            present = [(c, v) for c, v in parts if c]
            total = sum(c.value for c, _ in present)
            if len(present) >= 2 and total > re_.value * (1 + rel_tol) + abs_tol:
                for c, v in present:
                    c.representative.add_flag(
                        "components_exceed_total", "warning",
                        f"{' + '.join(v for _, v in present)} = {total:,.2f} exceeds revenue expenditure "
                        f"{re_.value:,.2f} (possible total picked up as a component)")

        for rc in cfg.get("validation.ratio_checks", []) or []:
            num, den = _cell(by_var, rc["numerator"]), _cell(by_var, rc["denominator"])
            if not num or not den or den.value <= 0:
                continue
            ratio = num.value / den.value
            if rc.get("abs"):
                ratio = abs(ratio)
            if not (rc["min"] <= ratio <= rc["max"]):
                num.representative.add_flag(
                    "ratio_out_of_range", "warning",
                    f"{rc['numerator']}/{rc['denominator']} = {ratio:.3f} outside [{rc['min']}, {rc['max']}] "
                    f"({num.value:,.2f} / {den.value:,.2f}); possible unit, column or definition error")
