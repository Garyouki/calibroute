"""Fit and apply auditable accept/review/abstain policies."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .bounds import clopper_pearson_upper
from .models import Action, Decision, PredictionRecord
from .shift import calibrate_js_threshold, confidence_histogram, js_divergence


@dataclass(frozen=True)
class GatePolicy:
    """A portable decision policy fitted only on labeled validation data."""

    accept_threshold: float
    review_threshold: float
    max_validation_risk: float
    validation_coverage: float
    reference_histogram: list[float]
    histogram_bins: int = 10
    max_js_divergence: float | None = None
    declared_risk_limit: float | None = None
    risk_upper_bound: float | None = None
    risk_method: str = "empirical"
    confidence_level: float = 0.95
    shift_confidence_level: float = 0.95
    shift_resamples: int = 500
    shift_min_batch_size: int = 20
    shift_effect_floor: float = 0.02
    shift_seed: int = 0
    schema_version: str = "1.2"

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold <= self.accept_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= review <= accept <= 1")
        if not 0 <= self.max_validation_risk <= 1:
            raise ValueError("max_validation_risk must be between 0 and 1")
        if not 0 < self.validation_coverage <= 1:
            raise ValueError("validation_coverage must be in (0, 1]")
        if self.histogram_bins != len(self.reference_histogram):
            raise ValueError("histogram_bins must match reference_histogram length")
        if (
            self.histogram_bins < 2
            or any(not math.isfinite(value) or value < 0 for value in self.reference_histogram)
            or sum(self.reference_histogram) <= 0
        ):
            raise ValueError("reference_histogram requires finite nonnegative positive mass")
        if self.risk_method not in {"empirical", "clopper_pearson"}:
            raise ValueError("risk_method must be empirical or clopper_pearson")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.shift_min_batch_size < 1:
            raise ValueError("shift_min_batch_size must be positive")
        if not 0 <= self.shift_effect_floor <= 1:
            raise ValueError("shift_effect_floor must be between 0 and 1")
        if self.shift_resamples < 20:
            raise ValueError("shift_resamples must be at least 20")
        if not 0 < self.shift_confidence_level < 1:
            raise ValueError("shift_confidence_level must be in (0, 1)")
        if self.max_js_divergence is not None and (
            not math.isfinite(self.max_js_divergence) or self.max_js_divergence < 0
        ):
            raise ValueError("max_js_divergence must be non-negative")
        for value, name in (
            (self.declared_risk_limit, "declared_risk_limit"),
            (self.risk_upper_bound, "risk_upper_bound"),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GatePolicy:
        return cls(
            accept_threshold=float(payload["accept_threshold"]),
            review_threshold=float(payload["review_threshold"]),
            max_validation_risk=float(payload["max_validation_risk"]),
            validation_coverage=float(payload["validation_coverage"]),
            reference_histogram=[float(value) for value in payload["reference_histogram"]],
            histogram_bins=int(payload.get("histogram_bins", 10)),
            max_js_divergence=(
                None
                if payload.get("max_js_divergence") is None
                else float(payload["max_js_divergence"])
            ),
            declared_risk_limit=(
                None
                if payload.get("declared_risk_limit") is None
                else float(payload["declared_risk_limit"])
            ),
            risk_upper_bound=(
                None
                if payload.get("risk_upper_bound") is None
                else float(payload["risk_upper_bound"])
            ),
            risk_method=str(payload.get("risk_method", "empirical")),
            confidence_level=float(payload.get("confidence_level", 0.95)),
            shift_confidence_level=float(payload.get("shift_confidence_level", 0.95)),
            shift_resamples=int(payload.get("shift_resamples", 500)),
            shift_min_batch_size=int(payload.get("shift_min_batch_size", 20)),
            shift_effect_floor=float(payload.get("shift_effect_floor", 0.02)),
            shift_seed=int(payload.get("shift_seed", 0)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )


def fit_policy(
    validation_records: Iterable[PredictionRecord],
    *,
    max_risk: float,
    min_coverage: float = 0.10,
    review_margin: float = 0.15,
    histogram_bins: int = 10,
    max_js_divergence: float | None = None,
    risk_method: str = "clopper_pearson",
    confidence_level: float = 0.95,
    shift_confidence_level: float = 0.95,
    shift_resamples: int = 500,
    shift_min_batch_size: int = 20,
    shift_effect_floor: float = 0.02,
    shift_seed: int = 0,
) -> GatePolicy:
    """Choose the highest-coverage attainable threshold satisfying ``max_risk``.

    By default the accepted set must satisfy a one-sided exact binomial upper
    bound, not only its observed error rate. ``empirical`` is available for
    exploratory analyses with small samples. Bounds used during threshold search
    are pointwise diagnostics, not selection-adjusted guarantees. Evaluate the
    frozen threshold with validate_policy on an independent IID holdout.
    """

    if not 0 <= max_risk < 1:
        raise ValueError("max_risk must be in [0, 1)")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0 <= review_margin <= 1:
        raise ValueError("review_margin must be between 0 and 1")
    if risk_method not in {"empirical", "clopper_pearson"}:
        raise ValueError("risk_method must be empirical or clopper_pearson")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    rows = list(validation_records)
    if not rows or any(row.correct is None for row in rows):
        raise ValueError("labeled validation records are required")
    ranked = sorted(rows, key=lambda row: row.confidence, reverse=True)
    minimum = max(1, int(min_coverage * len(ranked) + 0.999999))
    errors = 0
    best: tuple[int, float, float, float] | None = None
    for kept, row in enumerate(ranked, start=1):
        errors += int(not row.correct)
        # A deployed >= threshold accepts every member of a tie group.
        if kept < len(ranked) and ranked[kept].confidence == row.confidence:
            continue
        if kept < minimum:
            continue
        risk = errors / kept
        upper = (
            clopper_pearson_upper(errors, kept, confidence_level)
            if risk_method == "clopper_pearson"
            else risk
        )
        if kept >= minimum and upper <= max_risk:
            best = kept, row.confidence, risk, upper
    if best is None:
        raise ValueError(
            "no validation operating point satisfies max_risk and min_coverage; "
            "improve the model or relax the declared constraints"
        )
    kept, threshold, achieved_risk, risk_upper_bound = best
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
        declared_risk_limit=max_risk,
        risk_upper_bound=risk_upper_bound,
        risk_method=risk_method,
        confidence_level=confidence_level,
        shift_confidence_level=shift_confidence_level,
        shift_resamples=shift_resamples,
        shift_min_batch_size=shift_min_batch_size,
        shift_effect_floor=shift_effect_floor,
        shift_seed=shift_seed,
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
    shift_evaluable = len(rows) >= policy.shift_min_batch_size
    effective_threshold = policy.max_js_divergence
    if shift_evaluable and effective_threshold is None:
        effective_threshold = max(
            policy.shift_effect_floor,
            calibrate_js_threshold(
                policy.reference_histogram,
                len(rows),
                confidence_level=policy.shift_confidence_level,
                resamples=policy.shift_resamples,
                seed=policy.shift_seed,
            ),
        )
    severe_shift = bool(
        shift_evaluable and effective_threshold is not None and shift_score > effective_threshold
    )
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
        "shift_evaluable": shift_evaluable,
        "shift_status": (
            "severe_shift"
            if severe_shift
            else "within_reference_range"
            if shift_evaluable
            else "insufficient_batch"
        ),
        "max_js_divergence": effective_threshold,
        "severe_shift": severe_shift,
        "actions": counts,
    }
    return decisions, summary
