"""Shared fixtures, including a synthetic CAG-style PDF built with PyMuPDF.

The synthetic document mirrors the real layouts seen in the Himachal Pradesh SFAR 2024-25:
a committed-expenditure table, a snapshot table with Actuals/BE/RE columns and a landscape
appendix page stored with /Rotate 270. Development and CI therefore never depend on
downloading a document.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import fitz
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statefin.config import Config  # noqa: E402

REAL_SFAR = ROOT.parent / "Docs" / "HP-Report-No-2-of-2026-SFAR-2024-25-English-06a995015d08ee1.68791217.pdf"


def draw_table(page: "fitz.Page", x0: float, y0: float, col_widths: Sequence[float], rows: List[Sequence[str]],
               row_h: float = 16, fontsize: float = 8) -> None:
    xs = [x0]
    for w in col_widths:
        xs.append(xs[-1] + w)
    y1 = y0 + row_h * len(rows)
    shape = page.new_shape()
    for i in range(len(rows) + 1):
        shape.draw_line((x0, y0 + i * row_h), (xs[-1], y0 + i * row_h))
    for x in xs:
        shape.draw_line((x, y0), (x, y1))
    shape.finish(color=(0, 0, 0), width=0.6)
    shape.commit()
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            if text:
                page.insert_text((xs[c] + 3, y0 + r * row_h + row_h - 5), text, fontsize=fontsize)


def build_synthetic_sfar(path: Path) -> Path:
    doc = fitz.open()

    # Page 1: prose (candidate detection, state and document-type detection)
    p = doc.new_page()
    p.insert_text((72, 80), "State Finances Audit Report of the Government of Himachal Pradesh", fontsize=12)
    p.insert_text((72, 100), "for the year ended 31 March 2025", fontsize=11)
    prose = ("Committed expenditure of Himachal Pradesh consists of salaries and wages, "
             "Pensions and Other Retirement Benefits, and interest payments. Himachal Pradesh "
             "has high committed expenditure relative to revenue receipts.")
    p.insert_textbox(fitz.Rect(72, 130, 520, 260), prose, fontsize=10)

    # Page 2: committed expenditure (years in columns, footnote marker on a label)
    p = doc.new_page()
    p.insert_text((72, 80), "Table 1.18: Components of Committed Expenditure", fontsize=10)
    p.insert_text((420, 98), "(Rs. in crore)", fontsize=9)
    draw_table(p, 72, 105, [170, 80, 80, 80], [
        ["Components", "2022-23", "2023-24", "2024-25"],
        ["Salaries & Wages4", "17,587.18", "17,124.03", "17,461.50"],
        ["Pension", "9,283.87", "9,212.64", "9,543.85"],
        ["Interest", "4,828.64", "5,648.37", "6,260.93"],
        ["Total", "31,699.69", "31,985.04", "33,266.28"],
        ["Committed Expenditure as percentage of RE", "71.36", "71.50", "69.77"],
    ])
    p.insert_text((72, 215), "Source: Finance Accounts", fontsize=8)
    p.insert_text((72, 760), "4  Includes Grants-in-aid (Salary).", fontsize=7)

    # Page 3: snapshot with Actuals / BE / RE columns
    p = doc.new_page()
    p.insert_text((72, 80), "Table 1.3: Snapshot of Finances", fontsize=10)
    p.insert_text((420, 98), "(Rs. in crore)", fontsize=9)
    draw_table(p, 60, 105, [170, 75, 75, 75, 75], [
        ["Components", "2023-24", "2024-25", "2024-25", "2024-25"],
        ["", "(Actuals)", "(BE)", "(RE)", "(Actuals)"],
        ["Revenue Receipts", "39,173.04", "42,153.08", "43,703.96", "40,872.35"],
        ["Revenue Expenditure", "44,731.63", "46,666.63", "50,189.79", "47,676.96"],
        ["Capital Expenditure", "5,629.79", "6,269.68", "8,766.69", "5,956.83"],
        ["Revenue Deficit (-)/ Surplus (+)", "-5,558.59", "-4,513.55", "-6,485.83", "-6,804.61"],
        ["Fiscal Deficit (-)/ Surplus (+)", "-11,265.73", "-10,783.87", "-15,409.46", "-12,611.05"],
    ])

    # Page 4: landscape appendix stored rotated, as in the CAG report
    p = doc.new_page(width=842, height=595)
    p.insert_text((60, 50), "Appendix 1.1: Time Series data on the State Government Finances", fontsize=10)
    p.insert_text((680, 68), "(Rs. in crore)", fontsize=9)
    draw_table(p, 60, 75, [260, 90, 90, 90], [
        ["Particulars", "2022-23", "2023-24", "2024-25"],
        ["Outstanding Fiscal liabilities (year-end)", "82,177.21", "91,220.73", "1,02,187.62"],
        ["Gross State Domestic Product (GSDP)", "1,92,026", "2,12,169", "2,31,690"],
        ["Interest Payment/Revenue Receipts", "12.68", "14.42", "15.32"],
    ])
    p.set_rotation(270)

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config()


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory) -> Path:
    return build_synthetic_sfar(tmp_path_factory.mktemp("pdf") / "synthetic_sfar.pdf")
