"""Downloader (Phase 2). Not implemented in Phase 1.

Planned behaviour: for each DiscoveredDocument, download to a temporary file, verify it is
a PDF, hash it, store it content-addressed via DocumentRegistry (data/raw/<id>.pdf), and
record source_url and download_date in the manifest. Already-cached URLs/hashes are
skipped, so an interrupted crawl resumes where it stopped.
"""
from __future__ import annotations

from statefin.discovery.base import DiscoveredDocument


def download(document: DiscoveredDocument, registry) -> None:
    raise NotImplementedError("Downloading is Phase 2. Phase 1 processes local PDFs (--pdf).")
