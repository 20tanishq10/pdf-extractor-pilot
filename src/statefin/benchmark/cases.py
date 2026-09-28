"""Benchmark cases: one manually verified table each.

A case lists every value row of one table with its gold canonical variable (or null for
"not a target variable") and, for target rows, the gold values by financial year, plus
the gold unit. Cases start as `draft` (pre-filled from the pipeline) and only count as
evidence once a person has checked them against the snapshot and set status `verified`.
Evaluating the rule-based system on unverified drafts is circular, because the drafts
were pre-filled by those rules.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmark"


@dataclass
class GoldRow:
    label: str
    variable: Optional[str]                   # canonical variable, or None = not a target
    values: Dict[str, float] = field(default_factory=dict)   # financial_year -> value (target rows)
    note: str = ""


@dataclass
class BenchmarkCase:
    case_id: str
    document_file: str
    document_sha256: str
    page: int
    table_id: str
    caption: Optional[str]
    snapshot: Optional[str]
    gold_unit: Optional[str]
    rows: List[GoldRow]
    status: str = "draft"                     # draft | verified
    verified_by: str = ""
    verified_at: str = ""
    notes: str = ""
    prefilled_from: str = ""

    @property
    def document_id(self) -> str:
        return self.document_sha256[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkCase":
        d = dict(d)
        d["rows"] = [GoldRow(**r) for r in d.get("rows", [])]
        return cls(**d)


def case_path(case_id: str, directory: Path = BENCHMARK_DIR / "cases") -> Path:
    return directory / f"{case_id}.json"


def save_case(case: BenchmarkCase, directory: Path = BENCHMARK_DIR / "cases") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = case_path(case.case_id, directory)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(case.to_dict(), fh, indent=1, ensure_ascii=False)
    return path


def load_cases(directory: Path = BENCHMARK_DIR / "cases", include_drafts: bool = False) -> List[BenchmarkCase]:
    cases = []
    for path in sorted(Path(directory).glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            case = BenchmarkCase.from_dict(json.load(fh))
        if include_drafts or case.status == "verified":
            cases.append(case)
    return cases
