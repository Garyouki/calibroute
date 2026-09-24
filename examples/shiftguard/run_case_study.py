"""Audit frozen agent proposals and try a template-separated empirical policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from calibroute.adapters.shiftguard import read_shiftguard_records, write_shiftguard_csv
from calibroute.io import write_decisions, write_json
from calibroute.metrics import audit_records
from calibroute.models import Action, PredictionRecord
from calibroute.policy import fit_policy, route_batch

HERE = Path(__file__).resolve().parent


def split_templates(records: list[PredictionRecord]) -> tuple[list, list, dict]:
    """Hold out the lexicographically last task in each domain, without labels.

    All scenario variants, models, seeds, and telemetry profiles for a template
    remain together. This is retrospective template transfer, not an IID test.
    """
    domains = sorted({record.domain for record in records})
    fit_groups, evaluation_groups = set(), set()
    for domain in domains:
        tasks = sorted(
            {record.metadata["task_name"] for record in records if record.domain == domain}
        )
        if len(tasks) < 2:
            raise ValueError(f"domain {domain!r} needs at least two task templates")
        fit_groups.update((domain, task) for task in tasks[:-1])
        evaluation_groups.add((domain, tasks[-1]))
    fitting = [r for r in records if (r.domain, r.metadata["task_name"]) in fit_groups]
    evaluation = [r for r in records if (r.domain, r.metadata["task_name"]) in evaluation_groups]
    return (
        fitting,
        evaluation,
        {
            "rule": "lexicographically last task per domain held out; all variants grouped",
            "fitting_templates": [list(pair) for pair in sorted(fit_groups)],
            "evaluation_templates": [list(pair) for pair in sorted(evaluation_groups)],
            "fitting_count": len(fitting),
            "evaluation_count": len(evaluation),
            "iid_claim": False,
        },
    )


def run_case_study(input_path: Path, output: Path, max_risk: float = 0.10) -> dict:
    if not 0 <= max_risk < 1:
        raise ValueError("max_risk must be in [0, 1)")
    if input_path.resolve() == (HERE / "traces.jsonl").resolve():
        manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != manifest["fixture_sha256"]:
            raise ValueError("bundled trace SHA-256 does not match source_manifest.json")
    records = read_shiftguard_records(input_path)
    fitting, evaluation, split = split_templates(records)
    policy = None
    decisions = []
    try:
        policy = fit_policy(fitting, max_risk=max_risk, min_coverage=0.10, risk_method="empirical")
    except ValueError as error:
        if not str(error).startswith("no validation operating point"):
            raise
        routing = {"status": "not_run", "reason": str(error)}
    else:
        # No ground truth or outcome metadata is supplied to the router.
        unlabeled = [
            PredictionRecord(r.record_id, r.confidence, domain=r.domain) for r in evaluation
        ]
        decisions, routing = route_batch(unlabeled, policy)
        accepted = [
            row for row, decision in zip(evaluation, decisions) if decision.action == Action.ACCEPT
        ]
        routing.update(
            status="empirical_replay_only",
            accepted_empirical_risk=(
                sum(not row.correct for row in accepted) / len(accepted) if accepted else None
            ),
        )
    report = {
        "schema_version": 1,
        "scope": "Retrospective frozen simulator replay; no IID or deployment-risk guarantee.",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "split": split,
        "audit": audit_records(records),
        "fitting_audit": audit_records(fitting),
        "evaluation_audit": audit_records(evaluation),
        "policy_attempt": {
            "max_risk": max_risk,
            "min_coverage": 0.10,
            "risk_method": "empirical",
            "status": "fitted" if policy else "no_feasible_policy",
            "policy": policy.as_dict() if policy else None,
        },
        "routing": routing,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_shiftguard_csv(output / "predictions.csv", records)
    write_shiftguard_csv(output / "fitting.csv", fitting)
    write_shiftguard_csv(output / "evaluation.csv", evaluation)
    write_json(output / "report.json", report)
    write_json(output / "policy.json", report["policy_attempt"]["policy"])
    write_decisions(output / "decisions.csv", decisions)
    overall = report["audit"]["overall"]
    lines = [
        "# ShiftGuard confidence audit",
        "",
        report["scope"],
        "",
        f"Records: {len(records)}. Fitting: {len(fitting)}. Evaluation: {len(evaluation)}.",
        "",
        "Correct means valid JSON AND task success AND no unsafe outcome under always_execute.",
        "The original confidence question concerned safe commit; this label also requires task success.",
        "",
        f"Mean confidence: {overall['mean_confidence']:.4f}.",
        f"Safe task success: {overall['accuracy']:.4f}. ECE: {overall['ece']:.4f}.",
        "",
        f"Policy status: **{report['policy_attempt']['status']}** at risk limit {max_risk:.2f}.",
        "",
    ]
    if policy is None:
        lines.extend(
            [
                "No nonempty threshold met the fitting risk/coverage constraints. Routing was not run.",
                "`policy.json` is null and `decisions.csv` contains only its header.",
                "This is a valid audit outcome, not a successful reliability claim.",
            ]
        )
    else:
        lines.append(
            "Routing results are retrospective and include no measured human-review outcome."
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "traces.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--max-risk", type=float, default=0.10)
    args = parser.parse_args()
    try:
        report = run_case_study(args.input, args.output_dir, args.max_risk)
    except (OSError, ValueError) as error:
        parser.exit(1, f"ShiftGuard example failed: {error}\n")
    print(f"{report['policy_attempt']['status']}; report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
