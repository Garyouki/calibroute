"""Fit and apply auditable accept/review/abstain policies."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .bounds import clopper_pearson_upper
from .costs import ActionCosts
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
    objective: str = "max_coverage"
    costs: dict[str, float] | None = None
    validation_expected_cost: float | None = None
    validation_all_human_cost: float | None = None
    accept_disabled: bool = False
    infeasibility: dict[str, object] | None = None
    schema_version: str = "1.3"

    def __post_init__(self) -> None:
        if not 0 <= self.review_threshold <= self.accept_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= review <= accept <= 1")
        if not 0 <= self.max_validation_risk <= 1:
            raise ValueError("max_validation_risk must be between 0 and 1")
        if self.accept_disabled:
            if self.validation_coverage != 0:
                raise ValueError("an all-human policy must have zero validation_coverage")
        elif not 0 < self.validation_coverage <= 1:
            raise ValueError("validation_coverage must be in (0, 1]")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
        if self.costs is not None:
            ActionCosts.from_dict(self.costs)
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
        risk_method = str(payload.get("risk_method", "empirical"))
        risk_upper_bound = payload.get("risk_upper_bound")
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
                float(risk_upper_bound)
                if risk_method == "clopper_pearson" and risk_upper_bound is not None
                else None
            ),
            risk_method=risk_method,
            confidence_level=float(payload.get("confidence_level", 0.95)),
            shift_confidence_level=float(payload.get("shift_confidence_level", 0.95)),
            shift_resamples=int(payload.get("shift_resamples", 500)),
            shift_min_batch_size=int(payload.get("shift_min_batch_size", 20)),
            shift_effect_floor=float(payload.get("shift_effect_floor", 0.02)),
            shift_seed=int(payload.get("shift_seed", 0)),
            objective=str(payload.get("objective", "max_coverage")),
            costs=(
                None
                if payload.get("costs") is None
                else ActionCosts.from_dict(payload["costs"]).as_dict()
            ),
            validation_expected_cost=_optional_float(payload.get("validation_expected_cost")),
            validation_all_human_cost=_optional_float(payload.get("validation_all_human_cost")),
            accept_disabled=bool(payload.get("accept_disabled", False)),
            infeasibility=payload.get("infeasibility"),
            schema_version=str(payload.get("schema_version", "1.0")),
        )


OBJECTIVES = {"max_coverage", "min_cost", "all_human_fallback"}
ON_INFEASIBLE = {"raise", "all_human"}


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


class InfeasiblePolicyError(ValueError):
    """No acceptance threshold satisfies the declared constraints.

    ``report`` is a JSON-serializable diagnosis with the closest attainable
    operating points, so callers can decide whether to improve the model,
    collect more validation data, or relax constraints explicitly.
    """

    def __init__(self, message: str, report: dict[str, object]):
        super().__init__(message)
        self.report = report


def fit_policy(
    validation_records: Iterable[PredictionRecord],
    *,
    max_risk: float | None = None,
    min_coverage: float = 0.10,
    review_margin: float = 0.15,
    costs: ActionCosts | None = None,
    on_infeasible: str = "raise",
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
    """Fit an acceptance threshold on labeled validation data.

    Without ``costs`` the objective is coverage: choose the highest-coverage
    attainable threshold whose accepted-error criterion satisfies ``max_risk``.
    With ``costs`` the objective is cost: choose the threshold with the lowest
    total declared cost on these records, where records below the acceptance
    threshold are charged as review (within ``review_margin``) or abstention.
    ``max_risk``, when also given, remains a hard constraint in cost mode.

    By default the risk criterion is a one-sided exact binomial upper bound;
    ``empirical`` is available for exploratory analyses with small samples.
    Bounds used during threshold search are pointwise diagnostics, not
    selection-adjusted guarantees. Evaluate the frozen threshold with
    validate_policy on an independent IID holdout.

    If no threshold is feasible, ``on_infeasible="raise"`` raises
    InfeasiblePolicyError with a structured report; ``"all_human"`` returns a
    policy that routes every record to human review and stores the report.
    """

    if max_risk is None and costs is None:
        raise ValueError("declare max_risk, costs, or both")
    if max_risk is not None and not 0 <= max_risk < 1:
        raise ValueError("max_risk must be in [0, 1)")
    if on_infeasible not in ON_INFEASIBLE:
        raise ValueError(f"on_infeasible must be one of {sorted(ON_INFEASIBLE)}")
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
    total = len(ranked)
    minimum = max(1, int(min_coverage * total + 0.999999))
    ascending = sorted(row.confidence for row in ranked)

    def cost_at(kept: int, errors: int, threshold: float) -> float:
        assert costs is not None
        review_threshold = max(0.0, threshold - review_margin)
        at_or_above_review = total - bisect_left(ascending, review_threshold)
        review = at_or_above_review - kept
        abstain = total - at_or_above_review
        return (
            errors * costs.accept_error
            + (kept - errors) * costs.accept_correct
            + review * costs.human_review
            + abstain * costs.abstain_cost
        )

    errors = 0
    best: tuple[int, float, float, float | None, float | None] | None = None
    closest: dict[str, object] | None = None
    largest_below_minimum: dict[str, object] | None = None
    for kept, row in enumerate(ranked, start=1):
        errors += int(not row.correct)
        # A deployed >= threshold accepts every member of a tie group.
        if kept < total and ranked[kept].confidence == row.confidence:
            continue
        risk = errors / kept
        criterion = None
        if max_risk is not None:
            criterion = (
                clopper_pearson_upper(errors, kept, confidence_level)
                if risk_method == "clopper_pearson"
                else risk
            )
        risk_ok = criterion is None or criterion <= max_risk
        point = {
            "accepted": kept,
            "coverage": kept / total,
            "accept_threshold": row.confidence,
            "observed_risk": risk,
            "risk_criterion": criterion,
        }
        if kept < minimum:
            if risk_ok and criterion is not None:
                largest_below_minimum = point
            continue
        if criterion is not None and (closest is None or criterion < closest["risk_criterion"]):
            closest = point
        if not risk_ok:
            continue
        if costs is None:
            best = kept, row.confidence, risk, criterion, None
        else:
            cost = cost_at(kept, errors, row.confidence)
            # Equal-cost candidates resolve to the higher coverage.
            if best is None or cost <= best[4]:
                best = kept, row.confidence, risk, criterion, cost

    all_human_cost = None if costs is None else costs.human_review
    shared = {
        "reference_histogram": confidence_histogram(
            (row.confidence for row in rows), bins=histogram_bins
        ),
        "histogram_bins": histogram_bins,
        "max_js_divergence": max_js_divergence,
        "declared_risk_limit": max_risk,
        "risk_method": risk_method,
        "confidence_level": confidence_level,
        "shift_confidence_level": shift_confidence_level,
        "shift_resamples": shift_resamples,
        "shift_min_batch_size": shift_min_batch_size,
        "shift_effect_floor": shift_effect_floor,
        "shift_seed": shift_seed,
        "costs": None if costs is None else costs.as_dict(),
        "validation_all_human_cost": all_human_cost,
    }

    if best is None:
        report = {
            "status": "no_feasible_policy",
            "validation_count": total,
            "validation_errors": sum(not row.correct for row in ranked),
            "declared_risk_limit": max_risk,
            "min_coverage": min_coverage,
            "min_accepted": minimum,
            "risk_method": risk_method,
            "confidence_level": confidence_level,
            "closest_at_or_above_min_coverage": closest,
            "largest_feasible_below_min_coverage": largest_below_minimum,
            "options": [
                "improve the model or its confidence signal",
                "collect more labeled validation data; bounds tighten with sample size",
                "relax max_risk or min_coverage explicitly and record the change",
                "route every record to human review with on_infeasible='all_human'",
            ],
        }
        if on_infeasible == "raise":
            raise InfeasiblePolicyError(
                "no validation operating point satisfies max_risk and min_coverage; "
                "improve the model or relax the declared constraints",
                report,
            )
        return GatePolicy(
            accept_threshold=1.0,
            review_threshold=0.0,
            max_validation_risk=0.0,
            validation_coverage=0.0,
            risk_upper_bound=None,
            objective="all_human_fallback",
            validation_expected_cost=all_human_cost,
            accept_disabled=True,
            infeasibility=report,
            **shared,
        )

    kept, threshold, achieved_risk, criterion, cost = best
    return GatePolicy(
        accept_threshold=threshold,
        review_threshold=max(0.0, threshold - review_margin),
        max_validation_risk=achieved_risk,
        validation_coverage=kept / total,
        risk_upper_bound=criterion if risk_method == "clopper_pearson" else None,
        objective="max_coverage" if costs is None else "min_cost",
        validation_expected_cost=None if cost is None else cost / total,
        **shared,
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
        if policy.accept_disabled:
            action = Action.HUMAN_REVIEW
            reason = "all_human_fallback_no_feasible_policy"
        elif severe_shift:
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
