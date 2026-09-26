# Compare routing rules on your evaluation data

Requires GitHub `main`; `compare` is not included in PyPI 0.3.0. Install from
the checkout with `python -m pip install -e .`.

Supply a previously fitted policy and labeled evaluation records. The command
evaluates three rules on exactly the same records:

| Rule | Automatic acceptance | Remaining outputs |
|---|---|---|
| `accept_all` | Everything | None |
| `fixed_threshold` | Confidence >= your baseline threshold | Human review |
| `calibroute` | Saved policy with its batch shift gate | Review or abstain |

```bash
calibroute compare --input evaluation.csv --policy policy.json --fixed-threshold 0.90 --output comparison.md
calibroute compare --input evaluation.csv --policy policy.json --fixed-threshold 0.90 --output comparison.json
```

Choose the baseline and policy using development data before examining the
evaluation labels. `compare` never refits a policy and does not rank a winner.
Compare accepted error rate alongside coverage, review counts, and abstentions:
lower risk accompanied by zero automatic acceptance is not evidence of useful
automation. With no acceptances, risk is `null` in JSON and `n/a` in Markdown.
Counts do not establish saved labor, reviewer accuracy, or financial benefit.

One input file is treated as one batch. Domain tables slice those same decisions;
they do not recompute shift within each domain. If domains are separate batches
in your application, run once per domain using `--domain DOMAIN`. For time-window
deployment, supply separate files for the actual windows. Batch composition and
size affect shift detection. Small batches may skip shift assessment; the report
includes `shift_status` and the JSON includes the full shift summary and policy.

These are observed evaluation results, not confidence guarantees. Use `validate`
on an independent IID holdout for fixed-threshold risk assessment. Do not use
this comparison repeatedly to tune a policy on the same holdout.

## Replay a supplied research example

From the repository root, run:

```bash
calibroute compare --input examples/financial_ner/encoder_seed42.csv --domain fin_test --policy examples/financial_ner/encoder_policy.json --fixed-threshold 0.90 --output comparison-fin.md
calibroute compare --input examples/financial_ner/encoder_seed42.csv --domain finer_ord_test --policy examples/financial_ner/encoder_policy.json --fixed-threshold 0.90 --output comparison-shift.md
```

The policy was fitted on `fin_valid`; filtering excludes that development split.
The 0.90 baseline is an illustrative preset, not an optimized baseline. These
data were inspected during development; the example is retrospective and cannot
be presented as fresh certification or independent adoption. See the
[data provenance](../examples/financial_ner/PROVENANCE.md).
