"""Offline, synthetic document-review walkthrough; compatible with PyPI 0.3.0."""

import argparse
import csv
import json
import random
from pathlib import Path

from calibroute import PredictionRecord, audit_records, fit_policy, route_batch, validate_policy
from calibroute.io import write_decisions, write_json


def sample(seed, prefix, count=300, labeled=True):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        confidence = rng.choices([0.97, 0.82, 0.25], weights=[6, 2, 2])[0]
        correct = rng.random() < {0.97: 0.99, 0.82: 0.55, 0.25: 0.1}[confidence]
        rows.append(
            PredictionRecord(
                f"{prefix}-{index}", confidence, correct if labeled else None, "synthetic_documents"
            )
        )
    return rows


def write_inputs(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "confidence", "correct", "domain"])
        for row in records:
            writer.writerow(
                [
                    row.record_id,
                    row.confidence,
                    "" if row.correct is None else int(row.correct),
                    row.domain,
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/output/document_review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    development = sample(13, "development")
    holdout = sample(42, "holdout")
    production = sample(2026, "production", labeled=False)
    shifted = [PredictionRecord(f"shifted-{i}", 0.1) for i in range(100)]
    for name, rows in (
        ("development", development),
        ("holdout", holdout),
        ("production", production),
        ("shifted", shifted),
    ):
        write_inputs(args.output / f"{name}.csv", rows)

    policy = fit_policy(development, max_risk=0.10)
    validation = validate_policy(holdout, policy)
    write_json(args.output / "policy.json", policy.as_dict())
    write_json(args.output / "holdout_report.json", validation)
    write_json(args.output / "development_audit.json", audit_records(development))
    if not validation["passed"]:
        raise SystemExit("Holdout did not pass; stop before routing. See holdout_report.json.")

    summary = {
        "data_kind": "synthetic demonstration, not deployment evidence",
        "holdout": validation,
        "batches": {},
    }
    for name, rows in (("production", production), ("shifted", shifted)):
        decisions, batch_summary = route_batch(rows, policy)
        write_decisions(args.output / f"{name}_decisions.csv", decisions)
        summary["batches"][name] = batch_summary
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Files written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
