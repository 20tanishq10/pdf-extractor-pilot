"""Command-line interface.

Mode B (Phase 1):   python extract.py --pdf path/to/report.pdf [--pdf another.pdf]
Mode A (Phase 2+):  python extract.py --all-states --start-year 1995 --end-year 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract.py",
        description="Extract Indian state-finance variables from official PDFs with full provenance.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pdf", action="append", type=Path, metavar="PATH",
                      help="local PDF to process (Mode B); repeat for several documents")
    mode.add_argument("--all-states", action="store_true",
                      help="discover and download official documents for all states (Mode A, not in Phase 1)")
    p.add_argument("--states", nargs="+", help="Mode A: restrict to these states")
    p.add_argument("--start-year", type=int, help="Mode A: first financial year (start year, e.g. 1995)")
    p.add_argument("--end-year", type=int, help="Mode A: last financial year (start year, e.g. 2025)")
    p.add_argument("--state", help="override the detected state for every --pdf document")
    p.add_argument("--doc-type", help="override the detected document type "
                                      "(CAG_FINANCE_ACCOUNTS, CAG_SFAR, STATE_BUDGET, FINANCE_COMMISSION_STUDY, OTHER)")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                   help="cache and processed artifacts (default: ./data)")
    p.add_argument("--out", type=Path, default=None, help="output directory (default: <data-dir>/output)")
    p.add_argument("--config-dir", type=Path, default=None, help="configuration directory (default: ./config)")
    p.add_argument("--force", action="store_true", help="ignore cached stages and reprocess")
    p.add_argument("--no-snapshots", action="store_true", help="do not render table snapshots")
    p.add_argument("--quiet", action="store_true", help="print only the summary")
    p.add_argument("--dashboard", action="store_true",
                   help="also serve the live dashboard (http://127.0.0.1:PORT) and keep it open for review")
    p.add_argument("--port", type=int, default=8765, help="dashboard port (default 8765)")
    p.add_argument("--no-db", action="store_true",
                   help="do not record progress/results in data/statefin.db (the dashboard will not see this run)")
    llm = p.add_argument_group("LLM semantic mapping (optional; see config/llm.yaml)")
    llm.add_argument("--llm", action="store_true", help="classify unmapped table rows with an LLM")
    llm.add_argument("--llm-provider", help="primary provider: gemini | ollama | openai_compatible")
    llm.add_argument("--llm-model", help="primary model name")
    llm.add_argument("--llm-fallback-provider", help="fallback provider")
    llm.add_argument("--llm-fallback-model", help="fallback model name")
    llm.add_argument("--no-llm-fallback", action="store_true", help="never call the fallback model")
    llm.add_argument("--llm-tables", choices=["unmapped_relevant", "all_year_tables"],
                     help="which tables to send (default from config)")
    return p


def make_semantic_mapper(cfg, data_dir: Path, overrides=None):
    from statefin.extraction.semantic import SemanticMapper
    from statefin.llm.cache import ResponseCache
    from statefin.llm.settings import build_provider, load_dotenv, load_llm_settings

    load_dotenv()
    settings = load_llm_settings(cfg.raw.get("llm", {}), overrides=overrides or {})
    cache = ResponseCache(data_dir / settings.cache_dir)
    primary = build_provider(settings.primary, cache)
    fallback = build_provider(settings.fallback, cache) if settings.fallback else None
    return SemanticMapper(cfg, settings, primary, fallback)


def build_semantic_mapper(cfg, args, data_dir: Path):
    return make_semantic_mapper(cfg, data_dir, {
        "primary_provider": args.llm_provider, "primary_model": args.llm_model,
        "fallback_provider": args.llm_fallback_provider, "fallback_model": args.llm_fallback_model,
        "no_fallback": "1" if args.no_llm_fallback else None, "table_selection": getattr(args, "llm_tables", None)})


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all_states:
        print("Mode A (--all-states discovery and download) is planned for Phase 2 and is not implemented.\n"
              "Phase 1 processes local PDFs: python extract.py --pdf path/to/report.pdf", file=sys.stderr)
        return 2

    from statefin.config import Config
    from statefin.pipeline import Pipeline

    cfg = Config(args.config_dir)
    out_dir = args.out or (args.data_dir / "output")
    log = (lambda _msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr))
    for path in args.pdf:
        if not path.exists():
            print(f"error: PDF not found: {path}", file=sys.stderr)
            return 1
    semantic = None
    if args.llm:
        from statefin.llm.base import LLMConfigurationError
        try:
            semantic = build_semantic_mapper(cfg, args, args.data_dir)
        except LLMConfigurationError as exc:
            print(f"error: LLM not usable: {exc}", file=sys.stderr)
            return 1
    store = None
    if not args.no_db:
        from statefin.store import Store
        store = Store(args.data_dir / "statefin.db")
    httpd = None
    if args.dashboard:
        if store is None:
            print("error: --dashboard needs the database (drop --no-db)", file=sys.stderr)
            return 1
        from statefin.dashboard.server import serve
        httpd = serve(cfg, args.data_dir, args.port, [p.resolve().parent for p in args.pdf], log, store)
        url = f"http://127.0.0.1:{args.port}"
        print(f"Dashboard: {url}", file=sys.stderr)
        import webbrowser
        webbrowser.open(url)
    pipeline = Pipeline(cfg, args.data_dir, force=args.force, snapshots=not args.no_snapshots, log=log,
                        semantic=semantic, store=store)
    result = pipeline.run(args.pdf, out_dir, state=args.state, doc_type=args.doc_type)
    print(result.summary)
    if httpd is not None:
        print(f"\nDashboard still running at http://127.0.0.1:{args.port} for review. Ctrl-C to stop.", file=sys.stderr)
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            httpd.shutdown()
    if result.failed:
        return 1
    if args.llm and any(d.llm_error for d in result.documents):
        return 3        # deterministic outputs written, but the requested LLM step failed
    return 0


if __name__ == "__main__":
    sys.exit(main())
