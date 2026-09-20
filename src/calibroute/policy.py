"""Fit and apply auditable accept/review/abstain policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .models import Action, Decision, PredictionRecord
from .shift import confidence_histogram, js_divergence


@dataclass(frozen=True)
class GatePolicy:
    """A portable decision policy fitted only on labeled validation data."""

    accept_threshold: float
    review_threshold: float
    max_validation_risk: float
    validation_coverage: float
    reference_histogram: list[float]
    histogram_bins: int = 10
    max_js_divergence: float = 0.10
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold <= self.accept_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= review <= accept <= 1")
        if not 0 <= self.max_validation_risk <= 1:
            raise ValueError("max_validation_risk must be between 0 and 1")
        if not 0 < self.validation_coverage <= 1:
            raise ValueError("validation_coverage must be in (0, 1]")
        if self.histogram_bins != len(self.reference_histogram):
            raise ValueError("histogram_bins must match reference_histogram length")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GatePolicy":
        return cls(
            accept_threshold=float(payload["accept_threshold"]),
            review_threshold=float(payload["review_threshold"]),
            max_validation_risk=float(payload["max_validation_risk"]),
            validation_coverage=float(payload["validation_coverage"]),
            reference_histogram=[float(value) for value in payload["reference_histogram"]],
            histogram_bins=int(payload.get("histogram_bins", 10)),
            max_js_divergence=float(payload.get("max_js_divergence", 0.10)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )


def fit_policy(
    validation_records: Iterable[PredictionRecord],
    *,
    max_risk: float,
    min_coverage: float = 0.10,
    review_margin: float = 0.15,
    histogram_bins: int = 10,
    max_js_divergence: float = 0.10,
) -> GatePolicy:
    """Choose the highest-coverage validation prefix satisfying ``max_risk``.

    This is an empirical operating point, not a statistical safety guarantee.
    The returned policy records its achieved validation coverage and should be
    revalidated when the model, task, or data distribution changes.
    """

    if not 0 <= max_risk < 1:
        raise ValueError("max_risk must be in [0, 1)")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0 <= review_margin <= 1:
        raise ValueError("review_margin must be between 0 and 1")

    rows = list(validation_records)
    if not rows or any(row.correct is None for row in rows):
        raise ValueError("labeled validation records are required")
    ranked = sorted(rows, key=lambda row: row.confidence, reverse=True)
    minimum = max(1, int(min_coverage * len(ranked) + 0.999999))
    errors = 0
    best: tuple[int, float, float] | None = None
    for kept, row in enumerate(ranked, start=1):
        errors += int(not row.correct)
        risk = errors / kept
        if kept >= minimum and risk <= max_risk:
            best = kept, row.confidence, risk
    if best is None:
        raise ValueError(
            "no validation operating point satisfies max_risk and min_coverage; "
            "improve the model or relax the declared constraints"
        )
    kept, threshold, achieved_risk = best
    return GatePolicy(
        accept_threshold=threshold,
        review_threshold=max(0.0, threshold - review_margin),
        max_validation_risk=achieved_risk,
        validation_coverage=kept / len(ranked),
        reference_histogram=confidence_histogram(
            (row.confidence for row in rows), bins=histogram_bins
        ),
        histogram_bins=histogram_bins,
        max_js_divergence=max_js_divergence,
    )


def route_batch(
    records: Iterable[PredictionRecord], policy: GatePolicy
) -> tuple[list[Decision], dict[str, object]]:
    """Route a batch using shift-first, prediction-second decision control."""

    rows = list(records)
    if not rows:
        raise ValueError("at least one record is required")
    observed_histogram = confidence_histogram(
        (row.confidence for row in rows), bins=policy.histogram_bins
    )
    shift_score = js_divergence(policy.reference_histogram, observed_histogram)
    severe_shift = shift_score > policy.max_js_divergence
    decisions = []
    for row in rows:
        if severe_shift:
            action = Action.HUMAN_REVIEW
            reason = "batch_distribution_shift"
        elif row.confidence >= policy.accept_threshold:
            action = Action.ACCEPT
            reason = "confidence_at_or_above_accept_threshold"
        elif row.confidence >= policy.review_threshold:
            action = Action.HUMAN_REVIEW
            reason = "confidence_in_review_band"
        else:
            action = Action.ABSTAIN
            reason = "confidence_below_review_threshold"
        decisions.append(
            Decision(
                record_id=row.record_id,
                confidence=row.confidence,
                action=action,
                reason=reason,
                shift_score=shift_score,
            )
        )
    counts = {action.value: 0 for action in Action}
    for decision in decisions:
        counts[decision.action.value] += 1
    summary = {
        "count": len(rows),
        "shift_score": shift_score,
        "max_js_divergence": policy.max_js_divergence,
        "severe_shift": severe_shift,
        "actions": counts,
    }
    return decisions, summary

