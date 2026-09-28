from statefin.models import Observation
from statefin.reconciliation.sources import reconcile
from statefin.validation import finalize
from statefin.validation.accounting import check_identities
from statefin.validation.anomalies import check_observation, check_time_series


def obs(var, fy, value, *, oid=None, table="t1", doc="d1", priority=2, sign=None, raw=None, label="x",
        unit="INR crore") -> Observation:
    return Observation(
        observation_id=oid or f"{var}-{fy}-{table}-{doc}", state="Himachal Pradesh", state_code="HP",
        financial_year=fy, canonical_variable=var, value=value, unit=unit, value_type="actual",
        sign_convention=sign, raw_label=label, raw_value_text=raw or f"{value:,.2f}", column_header=fy,
        row_section=None, label_footnote=None, source_document=f"{doc}.pdf", document_id=doc, document_hash=doc,
        document_type="CAG_SFAR", source_priority=priority, source_url=None, document_download_date=None,
        source_page=1, source_page_label=None, source_table="Table", table_id=table, table_bbox=[0, 0, 1, 1],
        table_cited_source=None, source_text="row", page_snapshot=None, extraction_method="table",
        mapping_method="alias_exact", mapping_note=None, confidence=0.98, confidence_tier="exact",
        confidence_reasons=["test"])


def test_agreeing_observations_corroborate(cfg):
    a, b = obs("pension_expenditure", "2024-25", 9543.85, table="t1"), obs("pension_expenditure", "2024-25", 9543.85, table="t2")
    cells = reconcile([a, b], cfg)
    cell = cells[("Himachal Pradesh", "2024-25", "pension_expenditure")]
    assert cell.status == "consistent" and cell.value == 9543.85
    assert a.corroborated_by == [b.observation_id]


def test_rounded_value_agrees_within_printed_precision(cfg):
    a = obs("union_tax_share", "2024-25", 10681.24, table="t1")
    b = obs("union_tax_share", "2024-25", 10681.0, table="t2", raw="10,681")
    assert reconcile([a, b], cfg)[("Himachal Pradesh", "2024-25", "union_tax_share")].status == "consistent"


def test_same_priority_disagreement_is_a_conflict_not_an_average(cfg):
    a = obs("salary_wages", "2015-16", 6806.64, table="t1")
    b = obs("salary_wages", "2015-16", 6806.84, table="t2")
    cell = reconcile([a, b], cfg)[("Himachal Pradesh", "2015-16", "salary_wages")]
    assert cell.status == "conflict" and cell.value is None
    assert all(any(f["check"] == "conflict" for f in o.validation_flags) for o in (a, b))
    assert (a.value, b.value) == (6806.64, 6806.84)       # untouched


def test_higher_priority_source_wins_and_discrepancy_is_flagged(cfg):
    cag = obs("interest_payments", "2016-17", 3358.91, doc="cag", priority=2)
    fc = obs("interest_payments", "2016-17", 3360.00, doc="fc", priority=4)
    cell = reconcile([cag, fc], cfg)[("Himachal Pradesh", "2016-17", "interest_payments")]
    assert cell.status == "resolved_by_priority" and cell.value == 3358.91
    assert any(f["check"] == "source_discrepancy" and f["severity"] == "warning" for f in fc.validation_flags)


def test_unmarked_deficit_with_opposite_sign_is_a_convention_difference(cfg):
    marked = obs("fiscal_deficit", "2024-25", -12611.05, table="t1", sign="surplus_positive")
    unmarked = obs("fiscal_deficit", "2024-25", 12611.05, table="t2", sign="unspecified")
    cell = reconcile([marked, unmarked], cfg)[("Himachal Pradesh", "2024-25", "fiscal_deficit")]
    assert cell.status == "consistent" and cell.value == -12611.05
    assert unmarked.sign_convention == "deficit_positive" and unmarked.value == 12611.05


def test_revenue_identity(cfg):
    items = [obs("revenue_receipts", "2024-25", 40872.35), obs("revenue_expenditure", "2024-25", 47676.96),
             obs("revenue_deficit_surplus", "2024-25", -6804.61, sign="surplus_positive")]
    check_identities(reconcile(items, cfg), cfg)
    assert any(f["check"] == "identity_revenue_balance" and f["severity"] == "info" for f in items[2].validation_flags)

    bad = [obs("revenue_receipts", "2023-24", 39173.04), obs("revenue_expenditure", "2023-24", 44731.63),
           obs("revenue_deficit_surplus", "2023-24", -5000.00, sign="surplus_positive")]
    check_identities(reconcile(bad, cfg), cfg)
    flag = next(f for f in bad[2].validation_flags if f["check"] == "identity_revenue_balance")
    assert flag["severity"] == "warning" and bad[2].value == -5000.00


def test_impossible_negative_and_unit_flags(cfg):
    o = obs("pension_expenditure", "2024-25", -9543.85)
    check_observation(o, cfg)
    assert any(f["check"] == "impossible_negative" for f in o.validation_flags)
    lakh = obs("pension_expenditure", "2024-25", 954385.0, unit="INR lakh")
    check_observation(lakh, cfg)
    assert any(f["check"] == "unit_not_crore" for f in lakh.validation_flags)


def test_scale_break_is_flagged_not_corrected(cfg):
    series = [obs("pension_expenditure", "2022-23", 9283.87), obs("pension_expenditure", "2023-24", 921264.0)]
    check_time_series(reconcile(series, cfg), cfg)
    assert any(f["check"] == "possible_scale_break" for f in series[1].validation_flags)
    assert series[1].value == 921264.0


def test_actuals_cannot_postdate_the_document(cfg):
    o = obs("revenue_expenditure", "2020-21", 41031.0)
    check_observation(o, cfg, document_date="2019-05-24")
    assert any(f["check"] == "fy_not_closed_at_document_date" for f in o.validation_flags)


def test_flagged_or_low_confidence_observations_go_to_review(cfg):
    ok, flagged, weak = obs("gsdp", "2024-25", 231690.0), obs("gsdp", "2023-24", 212169.0), obs("gsdp", "2022-23", 192026.0)
    flagged.add_flag("conflict", "warning", "x")
    weak.confidence = 0.6
    finalize([ok, flagged, weak], cfg)
    assert ok.review_status == "AUTO_ACCEPTED" and ok.validation_status == "PASSED"
    assert flagged.review_status == "NEEDS_REVIEW" and flagged.validation_status == "FLAGGED"
    assert weak.review_status == "NEEDS_REVIEW"
