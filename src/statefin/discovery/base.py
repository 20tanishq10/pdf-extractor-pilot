"""Source discovery interface (Mode A, Phase 2).

A discovery adapter lists official documents for a state and year range *without*
downloading them. Its output feeds the downloader, whose cached files enter the same
per-document pipeline as user-uploaded PDFs (Mode B), so nothing downstream depends on
where a document came from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DiscoveredDocument:
    source_url: str
    state: str
    document_type: str               # key of source_rules.yaml document_types
    financial_year: Optional[str]    # year the document reports on, if stated
    title: Optional[str] = None
    discovered_from: Optional[str] = None   # page on which the link was found
    discovered_at: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class SourceDiscovery(ABC):
    name = "base"

    @abstractmethod
    def discover(self, state: str, start_year: int, end_year: int) -> List[DiscoveredDocument]:
        """Return documents found for `state` covering financial years start_year..end_year."""
