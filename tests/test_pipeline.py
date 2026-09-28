"""End-to-end tests: synthetic PDF always; the real HP SFAR 2024-25 when it is present."""
import csv
import json

import pytest

from statefin.pipeline import Pipeline
from statefin.registry import DocumentRegistry

from conftest import REAL_SFAR

REQUIRED_PROVENANCE = ("source_document", "document_hash", "source_page", "raw_label", "raw_value_text",
                       "source_text", "confidence", "extraction_method", "review_status")


def test_end_to_end_synthetic(cfg, synthetic_pdf, tmp_path):
    pipe = Pipeline(cfg, tmp_path / "data", snapshots=True, log=lambda _m: None)
    result = pipe.run([synthetic_pdf], tmp_path / "out")
    assert not result.failed

    got = {(o.canonical_variable, o.financial_year): o.value for o in result.observations}
    assert got[("pension_expenditure", "2024-25")] == 9543.85
    assert got[("salary_wages", "2022-23")] == 17587.18
    assert got[("revenue_receipts", "2024-25")] == 40872.35          # Actuals, not BE/RE
    assert got[("revenue_deficit_surplus", "2024-25")] == -6804.61
    assert got[("outstanding_liabilities", "2024-25")] == 102187.62  # from the rotated page
    assert got[("gsdp", "2024-25")] == 231690.0
    assert ("revenue_receipts", "2024-25") in got and 42153.08 not in got.values()

    for o in result.observations:
        for field in REQUIRED_PROVENANCE:
            assert getattr(o, field) not in (None, ""), field
        assert o.source_table or o.source_text
        assert not any(f["check"] == "provenance_text_mismatch" for f in o.validation_flags)

    salary = next(o for o in result.observations if o.canonical_variable == "salary_wages")
    assert salary.label_footnote and "Grants-in-aid" in salary.label_footnote

    rd = next(c for k, c in result.cells.items() if k[1:] == ("2024-25", "revenue_deficit_surplus"))
    assert any(f["check"] == "identity_revenue_balance" and f["severity"] == "info"
               for f in rd.representative.validation_flags)

    out = tmp_path / "out"
    for name in ("candidates.json", "extractions.json", "extractions.csv", "review_queue.csv", "panel.csv",
                 "panel_long.csv", "summary.txt"):
        assert (out / name).exists(), name
    rows = list(csv.DictReader(open(out / "extractions.csv", encoding="utf-8-sig")))
    assert len(rows) == len(result.observations)
    panel = list(csv.DictReader(open(out / "panel.csv", encoding="utf-8-sig")))
    assert {r["financial_year"] for r in panel} >= {"2022-23", "2023-24", "2024-25"}
    assert json.load(open(out / "candidates.json"))["documents"][0]["tables"]


def test_processing_is_resumable(cfg, synthetic_pdf, tmp_path):
    data = tmp_path / "data"
    Pipeline(cfg, data, log=lambda _m: None).run([synthetic_pdf], tmp_path / "out1")
    second = Pipeline(cfg, data, log=lambda _m: None).run([synthetic_pdf], tmp_path / "out2")
    assert second.documents[0].reused_stages == ["text", "candidates", "tables", "values"]
    record = DocumentRegistry(data).list_documents()[0]
    assert record.status == "VALIDATED"
    assert (data / record.cached_path).exists()


def test_same_pdf_twice_is_one_document(cfg, synthetic_pdf, tmp_path):
    reg = DocumentRegistry(tmp_path / "data")
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(synthetic_pdf.read_bytes())
    assert reg.register_local_pdf(synthetic_pdf).document_id == reg.register_local_pdf(copy).document_id


def test_non_pdf_is_rejected(tmp_path):
    bad = tmp_path / "not_a.pdf"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        DocumentRegistry(tmp_path / "data").register_local_pdf(bad)


# ----------------------------------------------------------------------------- real document

EXPECTED_HP_2024_25 = {
    "revenue_receipts": 40872.35, "revenue_expenditure": 47676.96, "salary_wages": 17461.50,
    "pension_expenditure": 9543.85, "interest_payments": 6260.93, "capital_expenditure": 5956.83,
    "revenue_deficit_surplus": -6804.61, "fiscal_deficit": -12611.05, "outstanding_liabilities": 102187.62,
    "gsdp": 231690.0,
}
EXPECTED_HP_2015_16 = {
    "revenue_receipts": 23440.48, "revenue_expenditure": 22302.81, "interest_payments": 3155.00,
    "capital_expenditure": 2864.49, "revenue_deficit_surplus": 1137.67, "fiscal_deficit": -2164.08,
    "outstanding_liabilities": 41197.43, "gsdp": 114239.0,
}


@pytest.mark.real_pdf
@pytest.mark.skipif(not REAL_SFAR.exists(), reason="HP SFAR 2024-25 PDF not available")
def test_real_hp_sfar_regression(cfg, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("real")
    result = Pipeline(cfg, tmp / "data", snapshots=False, log=lambda _m: None).run([REAL_SFAR], tmp / "out")
    rec = result.documents[0].record
    assert rec.state == "Himachal Pradesh" and rec.document_type == "CAG_SFAR" and rec.report_year == "2024-25"
    panel = {(r["financial_year"], r["variable"]): r for r in result.panel_long}
    for year, expected in (("2024-25", EXPECTED_HP_2024_25), ("2015-16", EXPECTED_HP_2015_16)):
        for var, value in expected.items():
            row = panel[(year, var)]
            assert row["panel_status"] == "included", (year, var, row["reason"])
            assert row["value"] == pytest.approx(value), (year, var)
    assert panel[("2024-25", "central_transfers")]["value"] == pytest.approx(10681.24 + 13721.61)
    assert not any(any(f["check"] == "provenance_text_mismatch" for f in o.validation_flags)
                   for o in result.observations)
    assert not any(o.financial_year == "2024-25" and o.value in (42153.08, 43703.96) for o in result.observations)
