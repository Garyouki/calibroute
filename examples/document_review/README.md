# Document extraction: decide what needs review

A team already has a model extracting a field from each document. CalibRoute
consumes its confidence scores, evaluates a threshold, and produces routing
recommendations. It does not read PDFs, call a model, generate confidence scores,
or operate a review queue. This demo runs locally without API keys or data uploads.

## Run in a few minutes

Install `calibroute-ai==0.3.0`, clone this repository for the example script, then
run `python examples/document_review/run_demo.py` from the repository root.
The script uses only APIs included in the published 0.3.0 package. Generated
files go to `examples/output/document_review/` (ignored by Git).

The example creates explicitly synthetic development, holdout, and unlabeled
production records using separate random seeds. It fits a policy on development
labels, assesses the frozen threshold on holdout labels, then routes a regular
batch and a deliberately shifted batch. A failed holdout stops the example.
The synthetic pass demonstrates the workflow, not performance on real documents.

Open `summary.json` for the validation result and action counts, or
`production_decisions.csv` for each record's action and explanation. The shifted
batch illustrates escalation; a confidence histogram cannot detect every shift.

## Use your own predictions

One row represents one unit you are prepared to accept or review (for example,
one extracted invoice amount). Provide `id,confidence,correct,domain` columns.
Use `correct=1` only when that unit meets your predefined correctness rule,
and `0` otherwise. Omit correctness for production. Confidence must be a
numeric score in [0, 1]; the package does not make a model's self-reported
confidence reliable. Evaluate candidate signals on development data.

1. Keep model training, threshold development, and final holdout data separate.
   Split by document/customer where appropriate to avoid leakage between related
   records. The binomial interpretation requires IID holdout units; splitting
   identifiers alone does not establish that assumption.
2. Choose the risk limit and operational review budget before looking at holdout
   outcomes. Fit once on development data and freeze the policy.
3. Validate on independent labels. If validation fails, collect more independent
   evidence or revise the model using development data; do not keep adjusting
   thresholds against the same holdout until one passes.
4. Route new predictions. Your application implements the actions: `accept`
   permits its configured automation, `human_review` queues a reviewer, and
   `abstain` withholds the result or requests more information.

These commands run against the generated files; replace them with your own
datasets when ready (each command fits on one line for Windows and Unix shells):

```bash
calibroute audit --input examples/output/document_review/development.csv --output examples/output/document_review/audit.md
calibroute fit --input examples/output/document_review/development.csv --max-risk 0.10 --output examples/output/document_review/policy.json
calibroute validate --input examples/output/document_review/holdout.csv --policy examples/output/document_review/policy.json --output examples/output/document_review/holdout_report.json
calibroute route --input examples/output/document_review/production.csv --policy examples/output/document_review/policy.json --output examples/output/document_review/decisions.csv
```

Proceed to routing only if validation returns exit code 0. Code 1 means the
bound does not meet the target; code 2 indicates input errors. `--max-risk 0.10`
is an illustrative choice, not a recommendation for a particular application.

## Evaluate usefulness

Compare with accepting everything and with your existing review rule, using
the same untouched evaluation set. Record accepted-set error rate, automatic
coverage, review/abstention counts, and actual review time if measured. Zero
automatic accepts does not mean the model became accurate. The toolkit does
not measure reviewer accuracy, costs, or time savings for you.

For observed research results, see the [Financial NER example](../financial_ner/README.md).
For assumptions and small-batch behavior, see [design](../../docs/design.md).
Share a sanitized [usage report](https://github.com/Garyouki/calibroute/issues/new?template=usage-report.md)
with the task, package version, baseline, results, and limitations.
