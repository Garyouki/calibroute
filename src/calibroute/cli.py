"""Command-line interface for CalibRoute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.financial_ner import read_financial_ner_records, write_records_csv
from .io import read_policy, read_records, write_decisions, write_json
from .metrics import audit_records
from .policy import fit_policy, route_batch
from .report import render_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibroute",
        description="Evaluate confidence, fit a selective policy, and route uncertain AI outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit labeled predictions.")
    audit.add_argument("--input", required=True, help="CSV or JSONL predictions.")
    audit.add_argument("--output", required=True, help="Output .json or .md report.")
    audit.add_argument("--bins", type=int, default=10)
    audit.add_argument("--coverages", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])

    fit = subparsers.add_parser("fit", help="Fit a policy on labeled validation data.")
    fit.add_argument("--input", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--max-risk", type=float, required=True)
    fit.add_argument("--min-coverage", type=float, default=0.10)
    fit.add_argument("--review-margin", type=float, default=0.15)
    fit.add_argument("--max-shift", type=float, help="Optional fixed JSD threshold.")
    fit.add_argument(
        "--risk-method",
        choices=["clopper_pearson", "empirical"],
        default="clopper_pearson",
    )
    fit.add_argument("--confidence-level", type=float, default=0.95)
    fit.add_argument("--shift-min-batch-size", type=int, default=20)
    fit.add_argument("--shift-effect-floor", type=float, default=0.02)
    fit.add_argument("--bins", type=int, default=10)

    route = subparsers.add_parser("route", help="Route an unlabeled prediction batch.")
    route.add_argument("--input", required=True)
    route.add_argument("--policy", required=True)
    route.add_argument("--output", required=True, help="Output decisions CSV.")
    route.add_argument("--summary", help="Optional routing summary JSON.")

    convert = subparsers.add_parser(
        "convert-financial-ner", help="Convert Financial NER sentence outputs."
    )
    convert.add_argument("--input", required=True, help="Sentence-level JSONL output.")
    convert.add_argument("--output", required=True, help="Common-schema CSV output.")
    convert.add_argument("--model", choices=["encoder", "generative"], required=True)
    convert.add_argument("--signal")
    convert.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            report = audit_records(
                read_records(args.input), bins=args.bins, coverages=args.coverages
            )
            output = Path(args.output)
            if output.suffix.lower() in {".md", ".markdown"}:
                output.write_text(render_markdown(report), encoding="utf-8")
            else:
                write_json(output, report)
            print(f"wrote audit report to {output}")
        elif args.command == "fit":
            policy = fit_policy(
                read_records(args.input),
                max_risk=args.max_risk,
                min_coverage=args.min_coverage,
                review_margin=args.review_margin,
                histogram_bins=args.bins,
                max_js_divergence=args.max_shift,
                risk_method=args.risk_method,
                confidence_level=args.confidence_level,
                shift_min_batch_size=args.shift_min_batch_size,
                shift_effect_floor=args.shift_effect_floor,
            )
            write_json(args.output, policy.as_dict())
            print(
                "fitted policy: "
                f"accept>={policy.accept_threshold:.4f}, "
                f"validation risk={policy.max_validation_risk:.4f}, "
                f"risk upper bound={policy.risk_upper_bound:.4f}, "
                f"coverage={policy.validation_coverage:.4f}"
            )
        elif args.command == "route":
            decisions, summary = route_batch(read_records(args.input), read_policy(args.policy))
            write_decisions(args.output, decisions)
            if args.summary:
                write_json(args.summary, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "convert-financial-ner":
            records = read_financial_ner_records(
                args.input, model=args.model, signal=args.signal, seed=args.seed
            )
            write_records_csv(args.output, records)
            print(f"wrote {len(records)} common-schema records to {args.output}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
