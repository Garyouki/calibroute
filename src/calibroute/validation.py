"""Evaluate a frozen acceptance threshold on independent labeled records."""

from collections.abc import Iterable

from .bounds import clopper_pearson_upper
from .models import PredictionRecord
from .policy import GatePolicy


def validate_policy(
    records: Iterable[PredictionRecord],
    policy: GatePolicy,
    *,
    max_risk: float | None = None,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Assess a fixed threshold. Caller must supply independent, IID holdout data.

    Do not tune the policy or repeatedly select policies using this holdout.
    This evaluates threshold acceptance before the optional batch shift gate.
    """
    rows = list(records)
    if not rows or any(row.correct is None for row in rows):
        raise ValueError("non-empty labeled holdout records are required")
    limit = policy.declared_risk_limit if max_risk is None else max_risk
    if limit is None or not 0 <= limit < 1:
        raise ValueError("a max_risk in [0, 1) is required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    accepted = [row for row in rows if row.confidence >= policy.accept_threshold]
    errors = sum(not row.correct for row in accepted)
    upper = clopper_pearson_upper(errors, len(accepted), confidence_level) if accepted else None
    return {
        "count": len(rows),
        "accepted": len(accepted),
        "errors": errors,
        "coverage": len(accepted) / len(rows),
        "empirical_risk": errors / len(accepted) if accepted else None,
        "risk_upper_bound": upper,
        "confidence_level": confidence_level,
        "declared_risk_limit": limit,
        "accept_threshold": policy.accept_threshold,
        "passed": upper is not None and upper <= limit,
        "scope": "fixed_threshold_before_shift_gate",
        "assumption": "independent IID holdout; policy frozen before observing labels",
    }
