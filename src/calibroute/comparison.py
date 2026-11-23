"""Offline comparison of frozen routing rules on one labeled evaluation batch."""

from collections.abc import Iterable

from .costs import ActionCosts, total_cost
from .models import Action, PredictionRecord
from .policy import GatePolicy, route_batch


def compare_policies(
    records: Iterable[PredictionRecord],
    policy: GatePolicy,
    *,
    fixed_threshold: float,
    costs: ActionCosts | None = None,
) -> dict[str, object]:
    """Compare all-accept, fixed-threshold/review, and CalibRoute empirically.

    Rules must be chosen before inspecting evaluation labels. Domain rows slice
    the same batch decisions; they do not rerun shift detection per domain.
    With ``costs`` (or costs stored in the policy) each rule also reports
    declared cost per record, using the same units for every rule.
    """
    rows = list(records)
    if not rows or any(row.correct is None for row in rows):
        raise ValueError("non-empty labeled evaluation records are required")
    if not 0 <= fixed_threshold <= 1:
        raise ValueError("fixed_threshold must be between 0 and 1")
    if costs is None and policy.costs is not None:
        costs = ActionCosts.from_dict(policy.costs)
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
            if costs is not None:
                result[name]["cost_per_record"] = total_cost(
                    [rows[i] for i in indices], [rule[i] for i in indices], costs
                ) / len(indices)
        return result

    return {
        "schema_version": "1.0",
        "evaluation": "empirical_frozen_rules",
        "fixed_threshold": fixed_threshold,
        "costs": None if costs is None else costs.as_dict(),
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
            "Cost columns use declared units; they are assumptions, not measured savings.",
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
    with_cost = report.get("costs") is not None
    if with_cost:
        declared = ", ".join(f"{key}={value:g}" for key, value in report["costs"].items())
        lines[-1:-1] = [f"Declared costs: {declared}"]
    sections = [("Overall", report["overall"])] + [
        (f"Domain: {name}", metrics) for name, metrics in report["domains"].items()
    ]
    for title, metrics in sections:
        lines.extend(
            [
                f"## {title.replace(chr(10), ' ')}",
                "",
                "| Strategy | N | Accepted | Auto coverage | Accepted errors | Accepted risk | Review | Abstain |"
                + (" Cost/record |" if with_cost else ""),
                "|---|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if with_cost else ""),
            ]
        )
        for name, values in metrics.items():
            risk = "n/a" if values["accepted_risk"] is None else f"{values['accepted_risk']:.2%}"
            cost = f" {values['cost_per_record']:.4f} |" if with_cost else ""
            lines.append(
                f"| {name} | {values['count']} | {values['accepted']} | "
                f"{values['automatic_coverage']:.2%} | {values['accepted_errors']} | "
                f"{risk} | {values['human_review']} | {values['abstain']} |{cost}"
            )
        lines.append("")
    lines.extend(["## Interpretation", ""])
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"
