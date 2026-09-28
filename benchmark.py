#!/usr/bin/env python3
"""Benchmark for the semantic-mapping layer.

    python benchmark.py draft --pdf path/to/report.pdf [--tables p042_t00 p023_t00] [--negatives 5]
    python benchmark.py list
    python benchmark.py eval --system deterministic
    python benchmark.py eval --system hybrid   [--llm-model gemini-3.5-flash-lite] [--include-drafts]
    python benchmark.py eval --system llm_only --llm-provider ollama --llm-model qwen2.5:14b --no-llm-fallback

Draft cases are pre-filled from the rule-based pipeline and must be checked by a person
against the table snapshot (then set "status": "verified") before results mean anything.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from statefin.benchmark.cases import BENCHMARK_DIR, load_cases  # noqa: E402
from statefin.cli import PROJECT_ROOT, build_semantic_mapper  # noqa: E402
from statefin.config import Config  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(prog="benchmark.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft", help="write draft cases for tables of a PDF")
    d.add_argument("--pdf", type=Path, required=True)
    d.add_argument("--tables", nargs="+", help="table ids (default: tables with observations + negatives)")
    d.add_argument("--negatives", type=int, default=5, help="year tables without observations to include")
    d.add_argument("--overwrite", action="store_true", help="replace existing case files (loses verification!)")
    sub.add_parser("list", help="list cases and their status")
    e = sub.add_parser("eval", help="score a system against the cases")
    e.add_argument("--system", choices=["deterministic", "llm_only", "hybrid"], default="deterministic")
    e.add_argument("--include-drafts", action="store_true", help="also score unverified drafts (not evidence)")
    for flag in ("--llm-provider", "--llm-model", "--llm-fallback-provider", "--llm-fallback-model"):
        e.add_argument(flag)
    e.add_argument("--no-llm-fallback", action="store_true")
    for parser in (d, e):
        parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = p.parse_args()
    cfg = Config()

    if args.cmd == "list":
        cases = load_cases(include_drafts=True)
        for c in cases:
            targets = sum(1 for r in c.rows if r.variable)
            print(f"{c.status:9} {c.case_id:28} p.{c.page:<4} rows {len(c.rows):>3} targets {targets:>3}  {(c.caption or '')[:60]}")
        print(f"{sum(c.status == 'verified' for c in cases)} verified / {len(cases)} total in {BENCHMARK_DIR / 'cases'}")
        return 0

    from statefin.pipeline import Pipeline

    if args.cmd == "draft":
        from statefin.benchmark.draft import draft_cases
        pipe = Pipeline(cfg, args.data_dir, log=lambda m: print(m, file=sys.stderr))
        res = pipe.process_document(args.pdf)
        if res.error:
            print(f"error: {res.error}", file=sys.stderr)
            return 1
        paths = draft_cases(pipe, res, args.tables, args.negatives, overwrite=args.overwrite)
        print(f"wrote {len(paths)} draft case(s) to {BENCHMARK_DIR / 'cases'} (existing cases are never overwritten)")
        return 0

    from statefin.benchmark.evaluate import evaluate, format_report, save_report
    from statefin.registry import DocumentRegistry

    cases = load_cases(include_drafts=args.include_drafts)
    if not cases:
        print("no verified cases; verify drafts (status: verified) or pass --include-drafts", file=sys.stderr)
        return 1
    semantic = None
    if args.system != "deterministic":
        args.llm_tables = None
        semantic = build_semantic_mapper(cfg, args, args.data_dir)
    report = evaluate(cases, DocumentRegistry(args.data_dir), cfg, args.system, semantic)
    print(format_report(report))
    print(f"saved {save_report(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
