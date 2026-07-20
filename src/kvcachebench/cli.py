from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostics import summarize_rows, validate_selected_failure_rows
from .io import read_jsonl, write_csv


def cmd_validate(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.path)
    if args.kind == "selected-failures":
        report = validate_selected_failure_rows(rows)
    else:
        report = {"ok": True, "rows": len(rows)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


def cmd_summarize(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.path)
    groups = [item.strip() for item in args.group_by.split(",") if item.strip()]
    summary = summarize_rows(rows, groups)
    if args.output:
        out = Path(args.output)
        if out.suffix.lower() == ".csv":
            write_csv(out, summary)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "groups": len(summary), "group_by": groups}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvcachebench", description="KVDiagnosis utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a KVDiagnosis JSONL artifact")
    validate.add_argument("path")
    validate.add_argument("--kind", default="selected-failures", choices=["selected-failures", "jsonl"])
    validate.set_defaults(func=cmd_validate)

    summarize = sub.add_parser("summarize", help="Summarize selected-failure rows")
    summarize.add_argument("path")
    summarize.add_argument("--group-by", default="dataset,method_name,retained_budget")
    summarize.add_argument("--output")
    summarize.set_defaults(func=cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
