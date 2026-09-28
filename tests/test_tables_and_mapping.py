from typing import List, Sequence

from statefin.extraction.mapper import map_table, match_label
from statefin.extraction.structure import analyze_table
from statefin.models import Cell, DocumentRecord, ExtractedTable
from statefin.pdf.tables import extract_tables


def grid_table(rows: List[Sequence[str]], col_w: float = 80, label_w: float = 180, row_h: float = 15,
               unit: str = "INR crore", caption: str = "Table X: synthetic") -> ExtractedTable:
    """ExtractedTable from a list of rows (empty strings = empty cells), with simple geometry."""
    cells = []
    for r, row in enumerate(rows):
        x = 0.0
        for c, text in enumerate(row):
            w = label_w if c == 0 else col_w
            if text:
                cells.append(Cell(r, c, text, [x, r * row_h, x + w, (r + 1) * row_h]))
            x += w
    return ExtractedTable(table_id="t0", document_id="d", page_number=7, page_label="7", index_on_page=0,
                          bbox=[0, 0, 500, row_h * len(rows)], engine="test", strategy="lines", page_rotation=0,
                          n_rows=len(rows), n_cols=max(map(len, rows)), cells=cells, caption=caption, unit=unit,
                          unit_source="context_above")


def doc_record() -> DocumentRecord:
    return DocumentRecord(document_id="d" * 16, sha256="d" * 64, original_filename="synthetic.pdf",
                          original_path="/tmp/synthetic.pdf", cached_path=None, state="Himachal Pradesh",
                          state_code="HP", document_type="CAG_SFAR")


SYNTHETIC = [
    ["Components", "2022-23", "2023-24", "2024-25"],
    ["Salaries & Wages", "17,587.18", "17,124.03", "17,461.50"],
    ["Pension", "9,283.87", "9,212.64", "9,543.85"],
    ["Interest", "4,828.64", "5,648.37", "6,260.93"],
    ["Interest Payment/Revenue Receipts", "12.68", "14.42", "15.32"],
]


def test_synthetic_table_maps_to_canonical_variables(cfg):
    table = grid_table(SYNTHETIC)
    model = analyze_table(table, cfg)
    assert model.orientation == "years_in_columns"
    obs = map_table(table, model, doc_record(), cfg).observations
    got = {(o.raw_label, o.canonical_variable, o.financial_year): o.value for o in obs}
    assert got[("Pension", "pension_expenditure", "2024-25")] == 9543.85
    assert got[("Salaries & Wages", "salary_wages", "2024-25")] == 17461.50
    assert got[("Interest", "interest_payments", "2024-25")] == 6260.93
    assert len(obs) == 9                       # 3 variables x 3 years from one table
    assert not any(o.raw_label.startswith("Interest Payment/") for o in obs)


def test_every_observation_has_provenance(cfg):
    table = grid_table(SYNTHETIC)
    for o in map_table(table, analyze_table(table, cfg), doc_record(), cfg).observations:
        assert o.source_document == "synthetic.pdf"
        assert o.source_page == 7
        assert o.source_table or o.source_text
        assert o.raw_label and o.raw_value_text and o.source_text
        assert o.document_hash and o.observation_id and o.extraction_method.startswith("table:")
        assert 0 < o.confidence <= 1 and o.confidence_reasons


def test_budget_and_revised_estimate_columns_are_skipped(cfg):
    table = grid_table([
        ["Components", "2023-24", "2024-25", "2024-25", "2024-25"],
        ["", "(Actuals)", "(BE)", "(RE)", "(Actuals)"],
        ["Revenue Receipts", "39,173.04", "42,153.08", "43,703.96", "40,872.35"],
    ])
    model = analyze_table(table, cfg)
    accepted = [(a.fy, a.value_type) for a in model.anchors if a.accepted]
    assert accepted == [("2023-24", "actual"), ("2024-25", "actual")]
    obs = map_table(table, model, doc_record(), cfg).observations
    assert sorted((o.financial_year, o.value) for o in obs) == [("2023-24", 39173.04), ("2024-25", 40872.35)]


def test_year_subheaders_under_percent_parent_are_not_levels(cfg):
    rows = [
        ["Object", "2015-16", "2016-17", "Per cent share", ""],
        ["", "", "", "2015-16", "2016-17"],
        ["Pensions", "3836.44", "4114.17", "17.20", "16.23"],
    ]
    table = grid_table(rows)
    # make the parent header span both share columns
    parent = next(c for c in table.cells if c.text == "Per cent share")
    parent.bbox[2] = parent.bbox[0] + 160
    obs = map_table(table, analyze_table(table, cfg), doc_record(), cfg).observations
    assert sorted(o.value for o in obs) == [3836.44, 4114.17]


def test_years_in_rows_table(cfg):
    table = grid_table([
        ["Year", "Expenditure on Salaries", "Expenditure on Pensions"],
        ["2014-15", "6966.54", "2914.49"],
        ["2015-16", "6806.84", "3836.44"],
    ], label_w=60, col_w=120)
    model = analyze_table(table, cfg)
    assert model.orientation == "years_in_rows"
    obs = map_table(table, model, doc_record(), cfg).observations
    got = {(o.canonical_variable, o.financial_year): o.value for o in obs}
    assert got[("pension_expenditure", "2015-16")] == 3836.44
    assert got[("salary_wages", "2014-15")] == 6966.54


def test_percentage_composition_table_is_not_mapped(cfg):
    table = grid_table([
        ["SOE", "2015-16", "2016-17"],
        ["Salaries", "30.51", "31.98"],
        ["Pensions", "17.20", "16.23"],
        ["Total", "100.00", "100.00"],
    ], unit=None, caption=None)
    model = analyze_table(table, cfg)
    assert model.inferred_unit == "percent"
    assert map_table(table, model, doc_record(), cfg).observations == []


def test_label_matching_is_whole_label(cfg):
    assert match_label("Pensions and Other Retirement Benefits", cfg).variable == "pension_expenditure"
    assert match_label("Salaries & Wages4", cfg).variable == "salary_wages"
    assert match_label("Revenue Deficit (-)/ Surplus (+) (4-10)", cfg).variable == "revenue_deficit_surplus"
    assert match_label("Outstanding Liabilities/GSDP", cfg).variable is None
    assert match_label("Rate of Growth of Overall Debt (per cent)", cfg).variable is None
    assert match_label("Grants-in-aid and contributions", cfg).variable is None   # ambiguous by design
    caveat = match_label("Salaries", cfg)
    assert caveat.variable == "salary_wages" and caveat.note and caveat.confidence < 0.9


def test_table_detection_on_synthetic_pdf(cfg, synthetic_pdf):
    tables = extract_tables(synthetic_pdf, "doc", [2, 3, 4], cfg)
    by_page = {t.page_number: t for t in tables}
    assert by_page[2].caption.startswith("Table 1.18")
    assert by_page[2].unit == "INR crore"
    assert by_page[2].cited_source == "Finance Accounts"
    assert by_page[3].caption.startswith("Table 1.3")
    assert by_page[4].page_rotation == 270          # rotated appendix still detected
    model = analyze_table(by_page[4], cfg)
    assert [a.fy for a in model.anchors] == ["2022-23", "2023-24", "2024-25"]
