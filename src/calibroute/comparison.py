"""Offline comparison of frozen routing rules on one labeled evaluation batch."""

from collections.abc import Iterable

from .models import Action, PredictionRecord
from .policy import GatePolicy, route_batch


def compare_policies(
    records: Iterable[PredictionRecord], policy: GatePolicy, *, fixed_threshold: float
) -> dict[str, object]:
    """Compare all-accept, fixed-threshold/review, and CalibRoute empirically.

    Rules must be chosen before inspecting evaluation labels. Domain rows slice
    the same batch decisions; they do not rerun shift detection per domain.
    """
    rows = list(records)
    if not rows or any(row.correct is None for row in rows):
        raise ValueError("non-empty labeled evaluation records are required")
    if not 0 <= fixed_threshold <= 1:
        raise ValueError("fixed_threshold must be between 0 and 1")
    decisions, shift = route_batch(rows, policy)
    actions = {
        "accept_all": [Action.ACCEPT] * len(rows),
        "fixed_threshold": [
            Action.ACCEPT if row.confidence >= fixed_threshold else Action.HUMAN_REVIEW
            for row in rows
        ],
        "calibroute": [decision.action for decision in decisions],
    }

    def summarize(indices):
        result = {}
        for name, rule in actions.items():
            accepted = [i for i in indices if rule[i] == Action.ACCEPT]
            errors = sum(not rows[i].correct for i in accepted)
            review = sum(rule[i] == Action.HUMAN_REVIEW for i in indices)
            abstain = sum(rule[i] == Action.ABSTAIN for i in indices)
            result[name] = {
                "count": len(indices),
                "accepted": len(accepted),
                "accepted_errors": errors,
                "accepted_risk": errors / len(accepted) if accepted else None,
                "automatic_coverage": len(accepted) / len(indices),
                "human_review": review,
                "abstain": abstain,
            }
        return result

    return {
        "schema_version": "1.0",
        "evaluation": "empirical_frozen_rules",
        "fixed_threshold": fixed_threshold,
        "policy": policy.as_dict(),
        "shift": shift,
        "overall": summarize(list(range(len(rows)))),
        "domains": {
            domain: summarize([i for i, row in enumerate(rows) if row.domain == domain])
            for domain in sorted({row.domain for row in rows})
        },
        "notes": [
            "Fixed-threshold baseline sends all below-threshold records to human review.",
            "Zero accepted records means risk is undefined, not zero.",
            "Domain summaries slice decisions from one input batch; shift is assessed once.",
            "This report does not fit thresholds or certify future risk.",
            "Choose rules on development data; repeated selection on this set invalidates holdout use.",
            "Review counts are not measured review time, reviewer accuracy, or cost savings.",
        ],
    }


def render_comparison(report: dict[str, object]) -> str:
    """Render counts and observed risk without claiming a winning strategy."""
    lines = [
        "# Routing comparison",
        "",
        f"Fixed baseline threshold: {report['fixed_threshold']}",
        f"CalibRoute acceptance threshold: {report['policy']['accept_threshold']}",
        f"Batch shift status: {report['shift']['shift_status']}",
        "",
    ]
    sections = [("Overall", report["overall"])] + [
        (f"Domain: {name}", metrics) for name, metrics in report["domains"].items()
    ]
    for title, metrics in sections:
        lines.extend(
            [
                f"## {title.replace(chr(10), ' ')}",
                "",
                "| Strategy | N | Accepted | Auto coverage | Accepted errors | Accepted risk | Review | Abstain |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in metrics.items():
            risk = "n/a" if values["accepted_risk"] is None else f"{values['accepted_risk']:.2%}"
            lines.append(
                f"| {name} | {values['count']} | {values['accepted']} | "
                f"{values['automatic_coverage']:.2%} | {values['accepted_errors']} | "
                f"{risk} | {values['human_review']} | {values['abstain']} |"
            )
        lines.append("")
    lines.extend(["## Interpretation", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"
