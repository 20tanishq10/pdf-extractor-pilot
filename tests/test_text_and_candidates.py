from statefin.extraction.candidates import rank_pages, score_page
from statefin.extraction.metadata import detect_document_type, detect_state
from statefin.models import PageText
from statefin.pdf.text import extract_page_texts, normalize_for_search


def _page(text: str, n: int = 1) -> PageText:
    return PageText(page_number=n, page_label=None, text=text, char_count=len(text), rotation=0,
                    image_count=0, needs_ocr=False)


def test_pdf_page_is_converted_to_text(synthetic_pdf):
    pages = extract_page_texts(synthetic_pdf)
    assert len(pages) == 4
    assert "pensions and other retirement benefits" in normalize_for_search(pages[0].text)
    assert "9,543.85" in pages[1].text
    assert pages[3].rotation == 270 and "1,02,187.62" in pages[3].text
    assert not any(p.needs_ocr for p in pages)


def test_pension_label_makes_page_relevant(cfg):
    cand = score_page(_page("Expenditure on Pensions and Other Retirement Benefits rose sharply."), cfg)
    assert "pension_expenditure" in cand.matched_variables
    assert cand.score > 0


def test_multi_concept_page_outranks_single_term_page(cfg):
    rich = _page("Pension 2022-23 2023-24 2024-25 salary revenue expenditure (₹ in crore) "
                 "9,283.87 9,212.64 9,543.85 17,587.18 17,124.03 17,461.50 44,425.26 44,731.63 47,676.96 "
                 "1,234.56 2,345.67 3,456.78", n=1)
    generic = _page("The State's revenue receipts are discussed in this chapter. " * 5, n=2)
    ranked = rank_pages([generic, rich], cfg)
    assert ranked[0].page_number == 1
    assert ranked[0].score > ranked[1].score
    assert ranked[0].is_candidate


def test_prose_page_without_years_is_not_a_table_candidate(cfg):
    ranked = rank_pages([_page("Pension salary wages interest payments revenue expenditure fiscal deficit " * 3)], cfg)
    assert ranked and not ranked[0].is_candidate


def test_state_and_document_type_detection(cfg, synthetic_pdf):
    pages = extract_page_texts(synthetic_pdf)
    state, _ = detect_state(pages, cfg)
    doc_type, _ = detect_document_type(pages, cfg)
    assert state is not None and state.name == "Himachal Pradesh"
    assert doc_type == "CAG_SFAR"
