#!/usr/bin/env python3
"""Live dashboard for extraction progress and human review.

    python dashboard.py                    # http://127.0.0.1:8765
    python dashboard.py --port 9000 --docs-dir ../Docs --open

Extractions started from the dashboard, or with extract.py in a terminal, appear live.
Review decisions are stored in data/statefin.db (append-only) and applied automatically.
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from statefin.cli import PROJECT_ROOT  # noqa: E402
from statefin.config import Config  # noqa: E402
from statefin.dashboard.server import serve  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(prog="dashboard.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--docs-dir", type=Path, action="append",
                   help="folder of PDFs offered in the 'start extraction' list (repeatable; default ../Docs)")
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    args = p.parse_args()
    docs_dirs = args.docs_dir or [PROJECT_ROOT.parent / "Docs"]
    httpd = serve(Config(args.config_dir), args.data_dir, args.port, docs_dirs,
                  log=lambda m: print(m, file=sys.stderr))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard running at {url}  (Ctrl-C to stop)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
