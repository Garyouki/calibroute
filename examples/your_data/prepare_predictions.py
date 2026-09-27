"""Map explicit CSV columns to CalibRoute's schema (standard library only)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def convert_csv(
    source: Path,
    destination: Path,
    *,
    id_column: str,
    confidence_column: str,
    correct_column: str | None = None,
    prediction_column: str | None = None,
    label_column: str | None = None,
    unlabeled: bool = False,
    domain_column: str | None = None,
) -> int:
    """Validate all rows before writing. Exact matching preserves case and whitespace."""
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("input and output must be different files")
    exact_match = prediction_column is not None or label_column is not None
    if exact_match and (prediction_column is None or label_column is None):
        raise ValueError("exact matching requires both prediction and label columns")
    if sum((correct_column is not None, exact_match, unlabeled)) != 1:
        raise ValueError("choose correctness labels, exact matching, or unlabeled production")
    required = [
        column
        for column in (
            id_column,
            confidence_column,
            correct_column,
            prediction_column,
            label_column,
            domain_column,
        )
        if column is not None
    ]
    if len(set(required)) != len(required):
        raise ValueError("each mapped field must use a different source column")
    records = []
    seen = set()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("duplicate CSV headers are ambiguous")
        missing = set(required) - set(headers)
        if missing:
            raise ValueError(f"missing mapped columns: {', '.join(sorted(missing))}")
        for row in reader:
            prefix = f"CSV line {reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{prefix}: number of values does not match the header")
            record_id = row[id_column].strip()
            if not record_id or record_id in seen:
                raise ValueError(f"{prefix}: ID must be nonempty and unique within this file")
            seen.add(record_id)
            try:
                confidence = float(row[confidence_column])
            except ValueError as exc:
                raise ValueError(f"{prefix}: confidence must be numeric in [0, 1]") from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"{prefix}: confidence must be finite and in [0, 1]")
            domain = row[domain_column].strip() if domain_column else "default"
            if not domain:
                raise ValueError(f"{prefix}: mapped domain must be nonempty")
            record = {"id": record_id, "confidence": confidence, "domain": domain}
            if correct_column is not None:
                label = row[correct_column].strip().lower()
                if label not in {"1", "0", "true", "false", "yes", "no", "correct", "incorrect"}:
                    raise ValueError(f"{prefix}: correctness must be an explicit boolean label")
                record["correct"] = int(label in {"1", "true", "yes", "correct"})
            elif exact_match:
                predicted, expected = row[prediction_column], row[label_column]
                if not predicted.strip() or not expected.strip():
                    raise ValueError(f"{prefix}: prediction and label must both be nonempty")
                record["correct"] = int(predicted == expected)
            records.append(record)
    if not records:
        raise ValueError("input must contain at least one prediction")
    fields = ["id", "confidence", "domain"]
    if not unlabeled:
        fields.append("correct")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--confidence-column", required=True)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--correct-column")
    labels.add_argument("--prediction-column", help="Use literal, case-sensitive exact matching.")
    labels.add_argument("--unlabeled", action="store_true", help="Production data without labels.")
    parser.add_argument("--label-column", help="Required with --prediction-column.")
    parser.add_argument("--domain-column", help="Defaults to the literal domain 'default'.")
    args = parser.parse_args(argv)
    try:
        count = convert_csv(
            args.input,
            args.output,
            id_column=args.id_column,
            confidence_column=args.confidence_column,
            correct_column=args.correct_column,
            prediction_column=args.prediction_column,
            label_column=args.label_column,
            unlabeled=args.unlabeled,
            domain_column=args.domain_column,
        )
    except (ValueError, OSError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Converted {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
