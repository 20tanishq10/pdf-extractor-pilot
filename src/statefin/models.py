"""Data records passed between pipeline stages.

Every record is a plain dataclass that round-trips through JSON, so each stage's output
can be cached on disk and inspected by hand.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

PIPELINE_VERSION = "0.3.0"


class Stage:
    """Document lifecycle. A failed document can be restarted from its last good stage."""
    DISCOVERED = "DISCOVERED"
    DOWNLOADED = "DOWNLOADED"          # for Mode B: registered + cached local copy
    TEXT_EXTRACTED = "TEXT_EXTRACTED"
    CANDIDATES_FOUND = "CANDIDATES_FOUND"
    TABLES_EXTRACTED = "TABLES_EXTRACTED"
    VALUES_EXTRACTED = "VALUES_EXTRACTED"
    SEMANTIC_MAPPED = "SEMANTIC_MAPPED"    # optional: only with --llm
    VALIDATED = "VALIDATED"
    REVIEWED = "REVIEWED"
    FINAL = "FINAL"
    ORDER = [DISCOVERED, DOWNLOADED, TEXT_EXTRACTED, CANDIDATES_FOUND, TABLES_EXTRACTED,
             VALUES_EXTRACTED, SEMANTIC_MAPPED, VALIDATED, REVIEWED, FINAL]


class _Record:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class DocumentRecord(_Record):
    document_id: str
    sha256: str
    original_filename: str
    original_path: str
    cached_path: Optional[str]
    source_kind: str = "local_upload"         # local_upload | downloaded
    source_url: Optional[str] = None
    download_date: Optional[str] = None
    registered_at: Optional[str] = None
    page_count: Optional[int] = None
    title: Optional[str] = None
    state: Optional[str] = None
    state_code: Optional[str] = None
    state_source: Optional[str] = None        # detected | override
    state_evidence: Dict[str, int] = field(default_factory=dict)
    document_type: Optional[str] = None
    document_type_source: Optional[str] = None
    document_type_evidence: Dict[str, int] = field(default_factory=dict)
    report_year: Optional[str] = None
    pdf_created: Optional[str] = None           # from PDF metadata (CreationDate / ModDate)
    pdf_modified: Optional[str] = None
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: str = Stage.DOWNLOADED
    last_error: Optional[str] = None
    pipeline_version: str = PIPELINE_VERSION


@dataclass
class PageText(_Record):
    page_number: int                  # 1-based PDF page index
    page_label: Optional[str]         # printed page number where detectable
    text: str
    char_count: int
    rotation: int
    image_count: int
    needs_ocr: bool


@dataclass
class CandidatePage(_Record):
    page_number: int
    page_label: Optional[str]
    score: float
    rank: int
    is_candidate: bool
    matched_variables: Dict[str, int]
    term_hits: int
    fy_tokens: int
    numeric_tokens: int
    unit_declared: Optional[str]
    captions: List[str]


@dataclass
class Cell(_Record):
    row: int
    col: int
    text: str
    bbox: List[float]


@dataclass
class ExtractedTable(_Record):
    table_id: str
    document_id: str
    page_number: int
    page_label: Optional[str]
    index_on_page: int
    bbox: List[float]
    engine: str
    strategy: str
    page_rotation: int
    n_rows: int
    n_cols: int
    cells: List[Cell]
    caption: Optional[str] = None
    caption_inherited_from: Optional[str] = None
    unit: Optional[str] = None
    unit_text: Optional[str] = None
    unit_source: Optional[str] = None        # context_above | header | inherited | None
    cited_source: Optional[str] = None
    footnotes: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractedTable":
        d = dict(d)
        d["cells"] = [Cell.from_dict(c) for c in d.get("cells", [])]
        return super().from_dict(d)


@dataclass
class Anchor(_Record):
    """A year column (or, for transposed tables, a year row)."""
    anchor_id: str
    fy: str
    value_type: Optional[str]         # actual | budget_estimate | ... | None (unqualified)
    header_text: str
    lo: float                         # x0 (columns) or y0 (rows)
    hi: float
    accepted: bool
    skip_reason: Optional[str] = None
    component: Optional[str] = None   # revenue | capital | total | charged | voted | ... (sub-column)


@dataclass
class ValueCell(_Record):
    """One numeric cell located at (row label x financial year)."""
    row_index: int
    raw_label: str
    row_serial: Optional[str]
    section: Optional[str]
    row_unit: Optional[str]
    anchor_id: Optional[str]
    fy: Optional[str]
    value_type: Optional[str]
    column_header: str
    raw_text: str
    value: Optional[float]
    parse_status: str
    parse_notes: List[str]
    bbox: List[float]
    row_text: str
    accepted: bool
    skip_reason: Optional[str] = None
    component: Optional[str] = None


@dataclass
class TableModel(_Record):
    table_id: str
    orientation: str                  # years_in_columns | years_in_rows | no_year_axis
    anchors: List[Anchor]
    value_cells: List[ValueCell]
    notes: List[str] = field(default_factory=list)
    inferred_unit: Optional[str] = None      # e.g. percent, when a "Total" row equals 100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TableModel":
        d = dict(d)
        d["anchors"] = [Anchor.from_dict(a) for a in d.get("anchors", [])]
        d["value_cells"] = [ValueCell.from_dict(v) for v in d.get("value_cells", [])]
        return super().from_dict(d)


@dataclass
class Flag(_Record):
    check: str
    severity: str                     # info | warning | error
    message: str

    def __str__(self) -> str:
        return f"{self.severity}:{self.check}: {self.message}"


@dataclass
class Observation(_Record):
    observation_id: str
    state: Optional[str]
    state_code: Optional[str]
    financial_year: str
    canonical_variable: str
    value: float
    unit: Optional[str]
    value_type: str
    sign_convention: Optional[str]
    raw_label: str
    raw_value_text: str
    column_header: str
    row_section: Optional[str]
    label_footnote: Optional[str]
    # provenance
    source_document: str
    document_id: str
    document_hash: str
    document_type: str
    source_priority: int
    source_url: Optional[str]
    document_download_date: Optional[str]
    source_page: int
    source_page_label: Optional[str]
    source_table: Optional[str]
    table_id: str
    table_bbox: List[float]
    table_cited_source: Optional[str]
    source_text: str
    page_snapshot: Optional[str]
    # how it was obtained
    extraction_method: str
    mapping_method: str
    mapping_note: Optional[str]
    confidence: float
    confidence_tier: str
    confidence_reasons: List[str]
    cell_bbox: Optional[List[float]] = None     # the value's cell on the page (unrotated frame)
    definition_tags: List[str] = field(default_factory=list)   # which definition this value uses
    series_base: Optional[str] = None           # base year of a statistical series (GSDP), e.g. "2011-12"
    definition_variant: str = ""                # sorted variant tags ("" = canonical definition)
    component: Optional[str] = None             # sub-column the value came from (revenue / total / ...)
    value_as_extracted: Optional[float] = None  # parser output; `value` differs only after a human correction
    value_crore: Optional[float] = None         # `value` in INR crore (set at validation; lakh / 100)
    # downstream status
    validation_status: str = "NOT_CHECKED"   # PASSED | FLAGGED
    validation_flags: List[Dict[str, str]] = field(default_factory=list)
    review_status: str = "PENDING"           # AUTO_ACCEPTED | NEEDS_REVIEW | HUMAN_ACCEPTED | HUMAN_CORRECTED | HUMAN_REJECTED
    review_reasons: List[str] = field(default_factory=list)
    reconciliation_status: Optional[str] = None
    corroborated_by: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    review_decision: Optional[str] = None    # accepted | rejected | corrected (from the review ledger)
    review_note: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    extracted_at: Optional[str] = None
    pipeline_version: str = PIPELINE_VERSION

    def add_flag(self, check: str, severity: str, message: str) -> None:
        flag = Flag(check, severity, message).to_dict()
        if flag not in self.validation_flags:
            self.validation_flags.append(flag)
