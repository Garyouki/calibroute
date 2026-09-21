"""Reproduce the encoder validation policy and cross-domain routing summary."""

from pathlib import Path

from calibroute.io import read_records, write_json
from calibroute.metrics import audit_records
from calibroute.models import Action
from calibroute.policy import fit_policy, route_batch
from calibroute.report import render_markdown
from calibroute.validation import validate_policy

HERE = Path(__file__).resolve().parent
records = read_records(HERE / "encoder_seed42.csv")
validation = [record for record in records if record.domain == "fin_valid"]
policy = fit_policy(validation, max_risk=0.10, min_coverage=0.10)

summaries = {}
for domain in ("fin_test", "finer_ord_test", "tweetner7_test"):
    batch = [record for record in records if record.domain == domain]
    decisions, summaries[domain] = route_batch(batch, policy)
    accepted = [
        record for record, decision in zip(batch, decisions) if decision.action == Action.ACCEPT
    ]
    summaries[domain]["accepted_empirical_risk"] = (
        sum(not record.correct for record in accepted) / len(accepted) if accepted else None
    )
    summaries[domain]["threshold_evaluation"] = validate_policy(batch, policy)
    summaries[domain]["evaluation_context"] = "retrospective benchmark, not fresh certification"

write_json(HERE / "encoder_policy.json", policy.as_dict())
write_json(HERE / "routing_summary.json", summaries)
for model, filename in (
    ("encoder", "encoder_seed42.csv"),
    ("generative", "generative_self_consistency_seed42.csv"),
):
    report = audit_records(read_records(HERE / filename))
    write_json(HERE / f"{model}_audit.json", report)
    (HERE / f"{model}_audit.md").write_text(render_markdown(report), encoding="utf-8")
print(f"wrote policy and routing summary to {HERE}")
