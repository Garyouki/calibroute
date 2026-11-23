"""Evaluate a frozen acceptance threshold on independent labeled records."""

from collections.abc import Iterable

from .bounds import clopper_pearson_lower, clopper_pearson_upper
from .costs import ActionCosts, total_cost
from .models import Action, PredictionRecord
from .policy import GatePolicy


def validate_policy(
    records: Iterable[PredictionRecord],
    policy: GatePolicy,
    *,
    max_risk: float | None = None,
    confidence_level: float | None = None,
) -> dict[str, object]:
    """Assess a fixed threshold. Caller must supply independent, IID holdout data.

    Holdout IDs must be unique. The confidence level defaults to the saved
    policy level. This assesses threshold acceptance before the batch shift
    gate. Do not tune or select policies using this holdout.

    ``passed`` keeps its 0.5 meaning (upper bound at or below the limit).
    ``verdict`` separates ``fail`` (evidence the limit is exceeded) from
    ``inconclusive`` (the bound is too wide to tell), which a single
    pass/fail flag conflates.
    """
    rows = list(records)
    if not rows or any(row.correct is None for row in rows):
        raise ValueError("non-empty labeled holdout records are required")
    seen_ids = set()
    for row in rows:
        record_id = row.record_id.strip()
        if not record_id:
            raise ValueError("holdout prediction ID must not be blank")
        if record_id in seen_ids:
            raise ValueError(f"duplicate holdout prediction ID: {record_id!r}")
        seen_ids.add(record_id)
    limit = policy.declared_risk_limit if max_risk is None else max_risk
    if limit is None and policy.costs is None:
        raise ValueError("a max_risk in [0, 1) is required")
    if limit is not None and not 0 <= limit < 1:
        raise ValueError("a max_risk in [0, 1) is required")
    if confidence_level is None:
        confidence_level = policy.confidence_level
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if policy.accept_disabled:
        accepted = []
    else:
        accepted = [row for row in rows if row.confidence >= policy.accept_threshold]
    errors = sum(not row.correct for row in accepted)
    upper = lower = None
    if accepted:
        upper = clopper_pearson_upper(errors, len(accepted), confidence_level)
        lower = clopper_pearson_lower(errors, len(accepted), confidence_level)
    verdict = _verdict(limit, upper, lower, policy.accept_disabled)
    report: dict[str, object] = {
        "count": len(rows),
        "accepted": len(accepted),
        "errors": errors,
        "coverage": len(accepted) / len(rows),
        "empirical_risk": errors / len(accepted) if accepted else None,
        "risk_upper_bound": upper,
        "risk_lower_bound": lower,
        "confidence_level": confidence_level,
        "declared_risk_limit": limit,
        "accept_threshold": policy.accept_threshold,
        "verdict": verdict,
        "verdict_explanation": VERDICTS[verdict],
        "passed": limit is not None and upper is not None and upper <= limit,
        "scope": "fixed_threshold_before_shift_gate",
        "assumption": "independent IID holdout; policy frozen before observing labels",
    }
    if policy.costs is not None:
        costs = ActionCosts.from_dict(policy.costs)
        actions = [_threshold_action(row, policy) for row in rows]
        report["costs"] = costs.as_dict()
        report["cost_per_record"] = total_cost(rows, actions, costs) / len(rows)
        report["all_human_cost_per_record"] = costs.human_review
        report["accept_all_cost_per_record"] = total_cost(
            rows, [Action.ACCEPT] * len(rows), costs
        ) / len(rows)
        report["cost_note"] = "Declared cost units, not measured savings."
    return report


VERDICTS = {
    "pass": "The holdout upper bound is at or below the declared risk limit.",
    "inconclusive": (
        "The holdout upper bound exceeds the limit, but the lower bound does not; the data "
        "cannot show the policy violates the limit. Policies fitted at the edge of the limit "
        "land here often. Collect more holdout data, or refit with a stricter max_risk."
    ),
    "fail": (
        "The holdout lower bound exceeds the declared risk limit: the policy's accepted "
        "error rate is above the limit at this confidence level."
    ),
    "no_accepted": "No holdout record was accepted, so accepted risk is undefined.",
    "all_human_fallback": "The policy accepts nothing; every record goes to human review.",
    "no_risk_limit": "No risk limit was declared; see the cost fields.",
}


def _verdict(
    limit: float | None, upper: float | None, lower: float | None, accept_disabled: bool
) -> str:
    if accept_disabled:
        return "all_human_fallback"
    if upper is None or lower is None:
        return "no_accepted"
    if limit is None:
        return "no_risk_limit"
    if upper <= limit:
        return "pass"
    if lower > limit:
        return "fail"
    return "inconclusive"


def _threshold_action(row: PredictionRecord, policy: GatePolicy) -> Action:
    if policy.accept_disabled:
        return Action.HUMAN_REVIEW
    if row.confidence >= policy.accept_threshold:
        return Action.ACCEPT
    if row.confidence >= policy.review_threshold:
        return Action.HUMAN_REVIEW
    return Action.ABSTAIN
