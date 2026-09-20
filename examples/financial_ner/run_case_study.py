"""Reproduce the encoder validation policy and cross-domain routing summary."""

from pathlib import Path

from calibroute.io import read_records, write_json
from calibroute.policy import fit_policy, route_batch

HERE = Path(__file__).resolve().parent
records = read_records(HERE / "encoder_seed42.csv")
validation = [record for record in records if record.domain == "fin_valid"]
policy = fit_policy(validation, max_risk=0.10, min_coverage=0.10)

summaries = {}
for domain in ("fin_test", "finer_ord_test", "tweetner7_test"):
    batch = [record for record in records if record.domain == domain]
    _, summaries[domain] = route_batch(batch, policy)

write_json(HERE / "encoder_policy.json", policy.as_dict())
write_json(HERE / "routing_summary.json", summaries)
print(f"wrote policy and routing summary to {HERE}")
