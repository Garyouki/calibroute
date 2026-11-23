"""Command-line interface for CalibRoute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.financial_ner import (
    read_financial_ner_entity_records,
    read_financial_ner_records,
    write_records_csv,
)
from .adapters.shiftguard import read_shiftguard_records, write_shiftguard_csv
from .comparison import compare_policies, render_comparison
from .costs import ActionCosts
from .io import read_policy, read_records, write_decisions, write_json
from .metrics import audit_records
from .policy import InfeasiblePolicyError, fit_policy, route_batch
from .report import render_markdown
from .validation import validate_policy


def _add_cost_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "declared costs", "Cost units per outcome. Give --cost-accept-error and --cost-review."
    )
    group.add_argument("--cost-accept-error", type=float, help="Cost of a wrong acceptance.")
    group.add_argument("--cost-review", type=float, help="Cost of one human review.")
    group.add_argument(
        "--cost-abstain", type=float, help="Cost of one abstention (default: review cost)."
    )
    group.add_argument(
        "--cost-accept-correct", type=float, default=0.0, help="Cost of a correct acceptance."
    )


def _costs_from_args(args: argparse.Namespace) -> ActionCosts | None:
    given = args.cost_accept_error is not None, args.cost_review is not None
    if not any(given) and args.cost_abstain is None:
        return None
    if not all(given):
        raise ValueError("--cost-accept-error and --cost-review must be given together")
    return ActionCosts(
        accept_error=args.cost_accept_error,
        human_review=args.cost_review,
        abstain=args.cost_abstain,
        accept_correct=args.cost_accept_correct,
    )


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
    fit.add_argument(
        "--max-risk",
        type=float,
        help="Accepted-error limit. Required unless costs are declared; a constraint if both.",
    )
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
    fit.add_argument(
        "--on-infeasible",
        choices=["error", "all-human"],
        default="error",
        help="When no threshold is feasible: exit 2 with a report, or save an all-human policy.",
    )
    _add_cost_arguments(fit)

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
    convert_entities = subparsers.add_parser(
        "convert-financial-ner-entities",
        help="Convert entity-level Financial NER JSONL output.",
    )
    convert_entities.add_argument("--input", required=True, help="Entity-level JSONL output.")
    convert_entities.add_argument("--output", required=True, help="Common-schema CSV output.")
    convert_entities.add_argument("--model", choices=["encoder", "generative"], required=True)
    convert_entities.add_argument("--signal")
    convert_entities.add_argument("--seed", type=int)
    convert_entities.add_argument("--entities-key", default="entities")
    shiftguard = subparsers.add_parser(
        "convert-shiftguard", help="Convert frozen ShiftGuard always-execute outcomes."
    )
    shiftguard.add_argument("--input", required=True, help="Replayed ShiftGuard JSONL.")
    shiftguard.add_argument("--output", required=True, help="Common-schema CSV output.")
    shiftguard.add_argument("--model", help="Optional exact model tag filter.")
    shiftguard.add_argument("--seed", type=int)
    validate = subparsers.add_parser(
        "validate", help="Evaluate a frozen policy on independent holdout."
    )
    validate.add_argument("--input", required=True)
    validate.add_argument("--policy", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--max-risk", type=float)
    validate.add_argument(
        "--confidence-level", type=float, help="Override the confidence level saved in the policy."
    )
    compare = subparsers.add_parser("compare", help="Compare three frozen rules on labeled data.")
    compare.add_argument("--input", required=True)
    compare.add_argument("--policy", required=True)
    compare.add_argument("--fixed-threshold", type=float, required=True)
    compare.add_argument("--domain", help="Evaluate only this domain as one batch.")
    compare.add_argument("--output", required=True, help="Comparison .json or .md report.")
    _add_cost_arguments(compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            rows = read_records(args.input)
            if args.domain is not None:
                rows = [row for row in rows if row.domain == args.domain]
            report = compare_policies(
                rows,
                read_policy(args.policy),
                fixed_threshold=args.fixed_threshold,
                costs=_costs_from_args(args),
            )
            output = Path(args.output)
            if output.suffix.lower() in {".md", ".markdown"}:
                output.write_text(render_comparison(report), encoding="utf-8")
            else:
                write_json(output, report)
            print(f"wrote comparison report to {output}")
        elif args.command == "validate":
            report = validate_policy(
                read_records(args.input, require_id=True),
                read_policy(args.policy),
                max_risk=args.max_risk,
                confidence_level=args.confidence_level,
            )
            write_json(args.output, report)
            print(json.dumps(report, indent=2))
            print(f"verdict: {report['verdict']} - {report['verdict_explanation']}")
            if report["verdict"] in {"no_risk_limit", "all_human_fallback"}:
                return 0
            return 0 if report["passed"] else 1
        elif args.command == "audit":
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
            costs = _costs_from_args(args)
            if args.max_risk is None and costs is None:
                raise ValueError("fit requires --max-risk, declared costs, or both")
            policy = fit_policy(
                read_records(args.input),
                max_risk=args.max_risk,
                min_coverage=args.min_coverage,
                review_margin=args.review_margin,
                costs=costs,
                on_infeasible="all_human" if args.on_infeasible == "all-human" else "raise",
                histogram_bins=args.bins,
                max_js_divergence=args.max_shift,
                risk_method=args.risk_method,
                confidence_level=args.confidence_level,
                shift_min_batch_size=args.shift_min_batch_size,
                shift_effect_floor=args.shift_effect_floor,
            )
            write_json(args.output, policy.as_dict())
            bound = (
                f"risk upper bound={policy.risk_upper_bound:.4f}, "
                if policy.risk_upper_bound is not None
                else ""
            )
            if policy.accept_disabled:
                print(
                    "no feasible threshold; saved an all-human fallback policy. "
                    "Diagnosis: " + json.dumps(policy.infeasibility, sort_keys=True)
                )
                return 0
            cost = (
                f", validation cost/record={policy.validation_expected_cost:.4f} "
                f"(all-human {policy.validation_all_human_cost:.4f})"
                if policy.validation_expected_cost is not None
                else ""
            )
            print(
                "fitted policy: "
                f"objective={policy.objective}, "
                f"accept>={policy.accept_threshold:.4f}, "
                f"validation risk={policy.max_validation_risk:.4f}, "
                f"{bound}"
                f"coverage={policy.validation_coverage:.4f}"
                f"{cost}"
            )
            print("next: freeze this policy and run `calibroute validate` on an untouched holdout.")
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
        elif args.command == "convert-financial-ner-entities":
            records = read_financial_ner_entity_records(
                args.input,
                model=args.model,
                signal=args.signal,
                seed=args.seed,
                entities_key=args.entities_key,
            )
            write_records_csv(args.output, records)
            print(f"wrote {len(records)} common-schema entity records to {args.output}")
        elif args.command == "convert-shiftguard":
            records = read_shiftguard_records(args.input, model=args.model, seed=args.seed)
            write_shiftguard_csv(args.output, records)
            print(f"wrote {len(records)} common-schema agent records to {args.output}")
    except InfeasiblePolicyError as error:
        print(json.dumps(error.report, indent=2, sort_keys=True), file=sys.stderr)
        parser.error(str(error))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
