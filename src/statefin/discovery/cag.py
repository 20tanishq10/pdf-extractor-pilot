"""CAG discovery adapter (Phase 2). Not implemented in Phase 1.

Planned behaviour: open the official CAG state-accounts / audit-report listing pages,
parse the document links found there, classify each link (Finance Accounts vol. I/II,
State Finances Audit Report), identify state and financial year from the link text and
page context, and return DiscoveredDocument records. URL patterns are learned from the
listing pages, not hard-coded, unless a pattern has been verified and documented.
"""
from __future__ import annotations

from typing import List

from statefin.discovery.base import DiscoveredDocument, SourceDiscovery


class CAGDiscovery(SourceDiscovery):
    name = "cag"

    def discover(self, state: str, start_year: int, end_year: int) -> List[DiscoveredDocument]:
        raise NotImplementedError("CAG discovery is Phase 2. Phase 1 processes local PDFs (--pdf).")
