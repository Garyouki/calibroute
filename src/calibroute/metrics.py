"""Transparent metrics for confidence quality and selective prediction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import groupby

from .models import PredictionRecord


def _labeled(records: Iterable[PredictionRecord]) -> list[PredictionRecord]:
    rows = list(records)
    if not rows:
        raise ValueError("at least one record is required")
    if any(row.correct is None for row in rows):
        raise ValueError("all records must include 'correct' for evaluation")
    return rows


def expected_calibration_error(records: Iterable[PredictionRecord], bins: int = 10) -> float:
    rows = _labeled(records)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    total = len(rows)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            row
            for row in rows
            if row.confidence >= low
            and (row.confidence < high or (index == bins - 1 and row.confidence <= high))
        ]
        if not bucket:
            continue
        accuracy = sum(bool(row.correct) for row in bucket) / len(bucket)
        confidence = sum(row.confidence for row in bucket) / len(bucket)
        error += len(bucket) / total * abs(accuracy - confidence)
    return error


def roc_auc(records: Iterable[PredictionRecord]) -> float | None:
    """AUROC for confidence ranking correct outputs above incorrect outputs."""

    rows = _labeled(records)
    positive = [row.confidence for row in rows if row.correct]
    negative = [row.confidence for row in rows if not row.correct]
    if not positive or not negative:
        return None
    wins = 0.0
    negatives_below = 0
    for _, group in groupby(
        sorted(rows, key=lambda row: row.confidence), key=lambda row: row.confidence
    ):
        members = list(group)
        positives = sum(bool(row.correct) for row in members)
        negatives = len(members) - positives
        wins += positives * (negatives_below + 0.5 * negatives)
        negatives_below += negatives
    return wins / (len(positive) * len(negative))


def risk_coverage(records: Iterable[PredictionRecord]) -> tuple[list[dict[str, float]], float]:
    """Return attainable threshold points and right-step, coverage-weighted AURC."""

    rows = sorted(_labeled(records), key=lambda row: row.confidence, reverse=True)
    points: list[dict[str, float]] = []
    errors = 0
    for index, row in enumerate(rows, start=1):
        errors += int(not row.correct)
        if index < len(rows) and rows[index].confidence == row.confidence:
            continue
        points.append(
            {
                "coverage": index / len(rows),
                "risk": errors / index,
                "threshold": row.confidence,
            }
        )
    previous = 0.0
    aurc = 0.0
    for point in points:
        aurc += (point["coverage"] - previous) * point["risk"]
        previous = point["coverage"]
    return points, aurc


def selective_points(
    records: Iterable[PredictionRecord], coverages: Sequence[float]
) -> list[dict[str, float]]:
    points, _ = risk_coverage(records)
    result = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage values must be in (0, 1]")
        point = next(point for point in points if point["coverage"] >= coverage)
        risk = point["risk"]
        result.append(
            {
                "requested_coverage": coverage,
                "coverage": point["coverage"],
                "risk": risk,
                "accuracy": 1 - risk,
                "threshold": point["threshold"],
            }
        )
    return result


def audit_records(
    records: Iterable[PredictionRecord],
    *,
    bins: int = 10,
    coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> dict[str, object]:
    """Create a machine-readable confidence and selective-prediction audit."""

    rows = _labeled(records)
    points, aurc = risk_coverage(rows)
    del points  # Full curves can be large; the selected operating points are reported below.
    by_domain: dict[str, list[PredictionRecord]] = defaultdict(list)
    for row in rows:
        by_domain[row.domain].append(row)

    def summarize(group: list[PredictionRecord]) -> dict[str, object]:
        accuracy = sum(bool(row.correct) for row in group) / len(group)
        return {
            "count": len(group),
            "accuracy": accuracy,
            "risk": 1 - accuracy,
            "mean_confidence": sum(row.confidence for row in group) / len(group),
            "ece": expected_calibration_error(group, bins=bins),
            "auroc": roc_auc(group),
            "selective": selective_points(group, coverages),
        }

    return {
        "schema_version": "1.1",
        "overall": {
            **summarize(rows),
            "aurc": aurc,
        },
        "domains": {name: summarize(group) for name, group in sorted(by_domain.items())},
    }
