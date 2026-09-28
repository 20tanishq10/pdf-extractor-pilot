#!/usr/bin/env python3
"""Entry point.

    python extract.py --pdf path/to/report.pdf
    python extract.py --all-states --start-year 1995 --end-year 2025   (Phase 2+)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from statefin.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
