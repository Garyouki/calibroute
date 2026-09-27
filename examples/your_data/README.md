# Bring your own prediction CSV

This walkthrough uses the **published `calibroute-ai==0.4.0` package** and a
standard-library column converter obtained from GitHub. The converter is an
example script on `main`, added after `v0.4.0`; it is not a new installed CLI
command. Record `git rev-parse HEAD` to pin the example version.

See the [internal onboarding check](../../docs/onboarding-check-2026-09-27.md)
for a completed attempt with the released package, including unfavorable results.

## 1. Install in a clean environment

Clone this repository and open a terminal at its root. On Windows PowerShell:

1. Create an environment: `python -m venv .venv`.
2. Install the release: `.\.venv\Scripts\python.exe -I -m pip --isolated install --index-url https://pypi.org/simple calibroute-ai==0.4.0`.
3. Inspect it: `.\.venv\Scripts\python.exe -I -m pip show calibroute-ai`.
4. Check commands: `.\.venv\Scripts\python.exe -I -m calibroute.cli --help`.

Use that same Python executable throughout. On macOS/Linux substitute
`./.venv/bin/python` for `.\.venv\Scripts\python.exe`. `-I` excludes the working
directory and `PYTHONPATH` from imports, making it easier to check the installed
release. Do not install an editable checkout for this walkthrough.

## 2. Define your unit, labels, and split

One row must represent a unit that can actually be accepted or reviewed, such
as one extracted invoice amount. Obtain a numeric confidence signal in [0, 1]
from your model. A model's self-reported confidence needs evaluation too.

Create separate `development-user.csv` and `evaluation-user.csv` files. Keep
training data separate as well. Define correctness and the risk limit before
examining evaluation results. Split related records by source document,
customer, or task template; check IDs and source groups for overlap. Distinct
IDs alone do not establish independence. The converter checks IDs within each
file; it does not create splits or verify independence across files.

The following is an **illustrative format**, not an adequate evaluation dataset:

| item_id | score | is_correct | task_domain |
|---|---:|---:|---|
| invoice-001 | 0.96 | 1 | invoice_amount |
| invoice-002 | 0.91 | 0 | invoice_amount |

Here `is_correct` is an externally determined evaluation label. It must not be
derived from confidence. For extraction, first define normalization, acceptable
tolerance, and how missing or ambiguous outputs count. For structured or agent
tasks, compute a task-specific correctness label with a suitable evaluator.

Choose a fixed baseline using development data or an existing operating rule.
The risk limit 0.10, minimum coverage 0.10, and baseline 0.90 below are examples;
they are not recommended limits for your application.

## 3. Map your column names

Convert development data:

`.\.venv\Scripts\python.exe -I examples/your_data/prepare_predictions.py --input development-user.csv --output examples/output/your_data/development.csv --id-column item_id --confidence-column score --correct-column is_correct --domain-column task_domain`

Convert the separate evaluation file:

`.\.venv\Scripts\python.exe -I examples/your_data/prepare_predictions.py --input evaluation-user.csv --output examples/output/your_data/evaluation.csv --id-column item_id --confidence-column score --correct-column is_correct --domain-column task_domain`

Supported correctness values are `1/0`, `true/false`, `yes/no`, or
`correct/incorrect` (case-insensitive). Blank labels, non-finite scores, scores
outside [0, 1], duplicate IDs, and malformed rows fail with exit code 2. All rows
are checked before writing; invalid data leaves an existing output untouched.
Input and output must be different files. A successful conversion replaces an
existing output, so choose an output path for generated files.

If literal string equality matches your task's correctness definition, replace
`--correct-column is_correct` with `--prediction-column prediction --label-column label`.
This comparison is case-sensitive and preserves whitespace; `USD`, `usd`, and
`USD ` differ. Empty strings are rejected. This mode does not implement semantic
evaluation, numeric tolerance, JSON equivalence, or NER span matching.

For production, replace the label arguments with `--unlabeled`. This explicitly
omits the `correct` column. If you omit `--domain-column`, the output domain is
`default`. Only mapped fields are exported: original text, predictions, and
extra metadata are not copied. Keep original source/group information separately
for provenance and split checks. The helper runs locally and uploads nothing.

## 4. Fit once, then assess the frozen policy

Audit development data:

`.\.venv\Scripts\python.exe -I -m calibroute.cli audit --input examples/output/your_data/development.csv --output examples/output/your_data/audit.md`

Fit using development labels:

`.\.venv\Scripts\python.exe -I -m calibroute.cli fit --input examples/output/your_data/development.csv --max-risk 0.10 --min-coverage 0.10 --output examples/output/your_data/policy.json`

If no feasible threshold exists, stop here. Preserve the failure and the chosen
constraints. Check label quality, the confidence signal, sample size, or the
model on development data before designing a new evaluation.

Validate using the separate labels:

`.\.venv\Scripts\python.exe -I -m calibroute.cli validate --input examples/output/your_data/evaluation.csv --policy examples/output/your_data/policy.json --output examples/output/your_data/validation.json`

Check `$LASTEXITCODE` immediately in PowerShell (`echo $?` in a POSIX shell).
Code 0 means the reported upper bound meets the limit; code 1 means it fails;
code 2 means invalid input. A failed assessment is still a useful recorded
outcome. Do not keep tuning against the same evaluation labels until it passes.
The reported binomial bound assumes independent IID holdout units and a policy
frozen before viewing their labels. Historical splits do not prove those assumptions.

## 5. Compare actual routing, even when validation fails

Run a diagnostic comparison using the same frozen rules:

`.\.venv\Scripts\python.exe -I -m calibroute.cli compare --input examples/output/your_data/evaluation.csv --policy examples/output/your_data/policy.json --fixed-threshold 0.90 --output examples/output/your_data/comparison.md`

Repeat with a `.json` output for machine-readable results. This command records
observed behavior and does not authorize deployment. Keep both the validation
and comparison reports, including unfavorable outcomes.

**Validation and routing answer different questions.** `validate` assesses the
acceptance threshold **before the shift gate**. `compare` applies the complete
router, including its batch shift gate. Validation can pass while all rows are
sent to review because their confidence distribution differs from development.
In that case automatic coverage is 0% and accepted risk is undefined. Do not
describe this as zero error, reduced review work, or a demonstrated improvement
over the baseline. Check the shift status and both reports together.

If the gate appears overly conservative, investigate stable development batches
and a new evaluation design. Do not relax it using the final evaluation batch
just to improve the reported coverage. A shift alarm detects confidence changes;
it does not by itself establish that accuracy degraded.

## 6. Route a new unlabeled batch only after reviewing the evidence

Prepare `production-user.csv` using the same unit, model, and confidence signal:

`.\.venv\Scripts\python.exe -I examples/your_data/prepare_predictions.py --input production-user.csv --output examples/output/your_data/production.csv --id-column item_id --confidence-column score --unlabeled --domain-column task_domain`

After a satisfactory evaluation for the intended setting, inspect routing:

`.\.venv\Scripts\python.exe -I -m calibroute.cli route --input examples/output/your_data/production.csv --policy examples/output/your_data/policy.json --output examples/output/your_data/decisions.csv --summary examples/output/your_data/routing-summary.json`

Your application implements the recommendations and human review process.
CalibRoute does not execute model actions, run a review queue, or measure
reviewer accuracy, time saved, or production safety.

Record package/example versions, dataset provenance and split rules, sample
counts, fixed limits, baseline, accepted error counts, coverage, and review
counts. Distinguish synthetic demonstrations, internal historical evaluations,
and actual external use. A sanitized [usage report](https://github.com/Garyouki/calibroute/issues/new?template=usage-report.md)
can document a real attempt, including installation problems and negative results.
