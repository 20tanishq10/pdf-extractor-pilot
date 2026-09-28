import pytest

from statefin.extraction.parsers import (dominant_financial_year, find_financial_years, is_serial_cell,
                                         parse_financial_year, parse_number)


@pytest.mark.parametrize("raw, expected", [
    ("9,543.85", 9543.85),
    ("17,461.50", 17461.50),
    ("6,260.93", 6260.93),
    ("₹ 9,543.85 crore", 9543.85),
    ("1,02,187.62", 102187.62),       # Indian lakh grouping
    ("2,31,690", 231690.0),
    ("0.00", 0.0),
    ("-12,611.05", -12611.05),
    ("- 11,265.73", -11265.73),        # sign separated by a space
    ("-\n12,379.84", -12379.84),       # sign on the previous line
    ("(-)6.04", -6.04),                # (-) notation
    ("(+)5.88", 5.88),
    ("2,217.73*", 2217.73),            # footnote marker
])
def test_parse_number_values(raw, expected):
    parsed = parse_number(raw)
    assert parsed.status == "ok"
    assert parsed.value == pytest.approx(expected)


def test_parse_number_unit_hint():
    assert parse_number("₹ 9,543.85 crore").unit_hint == "INR crore"
    assert parse_number("Rs. 12 lakh").unit_hint == "INR lakh"


@pytest.mark.parametrize("raw", ["--", "-", "NA", "N.A.", "Nil", "…", ""])
def test_placeholders_are_missing_not_zero(raw):
    parsed = parse_number(raw)
    assert parsed.status == "missing" and parsed.value is None


@pytest.mark.parametrize("raw", ["2024-25", "12,34", "abc", "1,2,3"])
def test_invalid_numbers_are_rejected(raw):
    assert parse_number(raw).status == "invalid"


def test_cell_with_several_numbers_is_multi():
    parsed = parse_number("21,210.01 25,224.89 -5,558.59")
    assert parsed.status == "multi"
    assert [p.value for p in parsed.parts] == [21210.01, 25224.89, -5558.59]


def test_notes_are_recorded():
    assert "split_sign" in parse_number("- 11,265.73").notes
    assert "footnote_marker" in parse_number("2,217.73*").notes
    assert "parentheses_negative" in parse_number("(1,234.50)").notes


@pytest.mark.parametrize("text, fy", [
    ("2024-25", "2024-25"), ("2023-24", "2023-24"), ("2022-23", "2022-23"),
    ("FY 2024–25", "2024-25"), ("2024-2025", "2024-25"), ("1999-2000", "1999-00"),
    ("1999-00", "1999-00"), ("2016-\n17", "2016-17"),
])
def test_financial_years(text, fy):
    assert parse_financial_year(text) == fy


@pytest.mark.parametrize("text", ["2015-2020", "2024-03-31", "Table 1.18", "2024-26"])
def test_not_financial_years(text):
    assert parse_financial_year(text) is None


def test_find_all_years():
    assert [fy for fy, _, _ in find_financial_years("2022-23 2023-24 2024-25")] == ["2022-23", "2023-24", "2024-25"]


def test_dominant_year_distinguishes_header_cells(cfg):
    q = cfg.all_qualifier_patterns
    assert dominant_financial_year("2024-25 (Actuals)", q) == "2024-25"
    assert dominant_financial_year("2024-25\n(BE)", q) == "2024-25"
    assert dominant_financial_year("Change over 2005-06 for", q) is None
    assert dominant_financial_year("2020-21 2021-22", q) is None


@pytest.mark.parametrize("text", ["1", "10.", "(i)", "(iv)", "6a", "(a)"])
def test_serial_cells(text):
    assert is_serial_cell(text)


@pytest.mark.parametrize("raw, normalized", [
    ("10. Revenue Deficit (-)/ Surplus (+) (4-10)", "revenue deficit / surplus"),
    ("Salaries & Wages4", "salaries and wages"),
    ("Interest payments#", "interest payments"),
    ("Fiscal Deficit {(4+5+6)-14}", "fiscal deficit"),
    ("27. Gross State Domestic Product (GSDP)", "gross state domestic product"),
    ("(b) State’s share of Union taxes and duties", "states share of union taxes and duties"),
    ("Revenue Receipts (RR) (₹ in crore)", "revenue receipts"),
    ("Pensions and Other\nRetirement Benefits", "pensions and other retirement benefits"),
])
def test_label_normalization(cfg, raw, normalized):
    assert cfg.normalize_label(raw).text == normalized


def test_label_normalization_records_what_was_removed(cfg):
    n = cfg.normalize_label("10. Revenue Deficit (-)/ Surplus (+) (4-10)")
    assert n.serial == "10." and n.sign_convention == "surplus_positive"
    assert cfg.normalize_label("Salaries & Wages4").footnote_marker == "4"
    assert cfg.normalize_label("Revenue Deficit (+)/Surplus (-)").sign_convention == "deficit_positive"
    assert cfg.normalize_label("Revenue Receipts (₹ in crore)").unit.unit == "INR crore"
