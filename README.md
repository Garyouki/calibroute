# CalibRoute

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24430274.svg)](https://doi.org/10.5281/zenodo.24430274)

CalibRoute is a lightweight Python toolkit for confidence evaluation and
uncertainty-aware routing. It converts model predictions into three actions:
`accept`, `human_review`, or `abstain`.

**Try it in your browser:** the [interactive demo on Hugging Face Spaces](https://huggingface.co/spaces/youkijiudai/calibroute-demo)
runs the 0.6.0 package with Pyodide on the Financial NER and ShiftGuard examples, or on a CSV you
upload. Nothing leaves your browser.

## Features

- Confidence calibration and error-ranking metrics
- Risk-coverage analysis
- Validation-only threshold fitting with a risk-limit or declared-cost objective
- Tie-aware threshold fitting and independent holdout risk assessment with
  `pass` / `inconclusive` / `fail` verdicts
- Structured no-feasible-policy reports and an opt-in all-human fallback
- Batch-size-aware confidence-shift detection
- Financial NER sentence- and entity-level output adapters
- ShiftGuard agent-replay adapter and offline confidence audit
- CSV and JSONL support
- Dependency-free Python API and CLI

## Install

Requires Python 3.10 or newer.

Install with `python -m pip install calibroute-ai`.

To obtain example scripts, clone the repository:

```bash
git clone https://github.com/Garyouki/calibroute.git
cd calibroute
```

Use the installed release for the walkthroughs. For source development, install
the checkout with `python -m pip install -e .`.

The distribution name is `calibroute-ai`; the Python import and command are
`calibroute`. No API key or model service is needed.

## Version compatibility

Release **0.6.0** adds declared action costs (`ActionCosts`, `--cost-*` flags)
to `fit`, `validate`, and `compare`; a `verdict` and lower bound in holdout
validation; and `InfeasiblePolicyError` reports with an opt-in all-human
fallback. Existing calls, saved policies, and exit codes keep their meaning.

Release **0.5.0** includes the 0.4.0 features below and hardens holdout
validation. When upgrading, give each holdout prediction a stable, unique ID.
Validation uses the policy's saved confidence level unless explicitly overridden.

Release **0.4.0** adds entity-level NER, ShiftGuard, and comparison reports to
the PyPI package. Upgrade with `python -m pip install --upgrade calibroute-ai`.

| Capability | PyPI 0.3.0 | PyPI 0.4.0 |
|---|---|---|
| `audit`, `fit`, `validate`, `route` | Available | Available |
| Three-rule comparison report: `compare` | Not included | Available |
| Tie-aware thresholds, holdout assessment, and confidence-shift routing | Available | Available |
| Financial NER sentence adapter: `convert-financial-ner` | Available | Available |
| Financial NER entity adapter: `convert-financial-ner-entities` | Not included | Available |
| ShiftGuard replay adapter: `convert-shiftguard` | Not included | Available |
| Document-review walkthrough | Compatible; obtain script from GitHub | Compatible; obtain script from GitHub |
| Financial NER sentence case study | Compatible; obtain script/data from GitHub | Compatible; obtain script/data from GitHub |
| ShiftGuard case study | Not compatible | Compatible; obtain script/data from GitHub |

For reproducible installation, use `python -m pip install calibroute-ai==0.6.0`.
Example scripts and data are available in the repository; installing the wheel
does not install the example folders. Check out tag `v0.6.0` for examples matching
this release; GitHub `main` may receive further development changes.

To check the active environment, run `python -m calibroute.cli --help` and
`python -m pip show calibroute-ai`. Check the listed commands and installation location; record
`git rev-parse HEAD` when reporting results from a source checkout.

## Quick start

To compare all-accept, a fixed threshold, and CalibRoute on the same labeled
batch, see the [one-command comparison guide](docs/comparison.md) (0.4.0 or newer).

New here? Follow the [document-review walkthrough](examples/document_review/README.md):
one offline script demonstrates development data, a separate holdout, and routing
using the published 0.5.0 package.
For your own exports, follow the [CSV onboarding walkthrough](examples/your_data/README.md)
for explicit column mapping, clean installation, and frozen-policy comparison.
The column helper is a repository example. Use the matching tag and record the
checkout commit when reporting results.

For measured limitations of confidence-shift monitoring, see the
[synthetic shift study](examples/shift_study/README.md), including false alarms,
label-only accuracy losses, and additional review workload.

The workflow has three steps, and the middle one is not optional: a policy
that fits successfully can still fail on new data.

1. **Fit** on labeled development (validation) data.
2. **Validate** the frozen policy on a separate, untouched labeled holdout.
3. **Route** unlabeled production batches only after step 2.

```bash
# 0. Audit labeled predictions
calibroute audit \
  --input examples/validation.csv \
  --output examples/audit.md

# 1. Fit an acceptance policy on development data
calibroute fit \
  --input examples/validation.csv \
  --max-risk 0.20 \
  --min-coverage 0.25 \
  --risk-method empirical \
  --output examples/policy.json

# 2. Assess the frozen policy on an independent labeled holdout (your file)
calibroute validate \
  --input holdout.csv \
  --policy examples/policy.json \
  --output holdout-report.json

# 3. Route a new prediction batch
calibroute route \
  --input examples/production_batch.csv \
  --policy examples/policy.json \
  --output examples/decisions.csv \
  --summary examples/routing-summary.json
```

## Choosing an objective: risk limit or costs

`--max-risk` answers "how much can I automate while keeping accepted errors
under a limit?" Declared costs answer "which threshold minimizes the total cost
of wrong acceptances, reviews, and abstentions?" They are different questions
and generally give different thresholds.

```bash
# Cost objective: a wrong acceptance costs 5 units, a review costs 1
calibroute fit --input dev.csv --cost-accept-error 5 --cost-review 1 --output policy.json

# Both: minimize cost subject to a 10% accepted-error bound
calibroute fit --input dev.csv --cost-accept-error 5 --cost-review 1 --max-risk 0.10 --output policy.json
```

`--cost-abstain` defaults to the review cost, since an abstained item usually
still has to be resolved. Cost units are your declared assumptions, not
measured savings. The three actions mean:

| Action | What the caller should do |
|---|---|
| `accept` | Use the model output automatically. |
| `human_review` | Send to a reviewer, with the model output shown as a suggestion. |
| `abstain` | Discard the model output; handle the item through the default non-AI path. |

If no threshold meets the constraints, `fit` exits 2 and prints a structured
report of the closest attainable operating points. Add
`--on-infeasible all-human` to save an explicit all-human policy instead.
See [design principles](docs/design.md) for details.

## Input format

| Field | Required | Description |
|---|---|---|
| `id` | required for `validate` | Unique prediction identifier. `record_id` is also accepted. |
| `confidence` | yes | Number between 0 and 1 |
| `correct` | `audit`, `fit`, `validate`, and `compare` | Boolean or `1`/`0` label |
| `domain` | no | Evaluation slice or deployment domain |

Additional columns are preserved as metadata.

The default fit searches thresholds using pointwise 95% Clopper-Pearson bounds.
Threshold selection on the same labels does **not** provide a 95% guarantee for
the selected policy. Freeze the policy, then assess it on a separate IID holdout:

```bash
calibroute validate --input holdout.csv --policy examples/policy.json --output holdout-report.json
```

Provide an explicit, unique ID for each holdout prediction. Assign the ID to
the underlying prediction before combining exports so duplicates can be
detected. The bound assumes independent IID holdout units.
`validate` uses the confidence level saved in the policy unless you pass
`--confidence-level`. Empirical fitting reports observed risk and stores
`null` in `risk_upper_bound`.

Exit code 0 means the holdout upper bound meets the declared risk limit. Exit
code 1 means it does not, including when there are zero accepted samples.
The report's `verdict` distinguishes `fail` (the lower bound is above the
limit: evidence of a violation) from `inconclusive` (the bound is too wide to
tell). Policies fitted right at the limit are often `inconclusive` on a
holdout; fit with a stricter `--max-risk` than the limit you will validate
against, or collect a larger holdout.
Exit code 2 indicates invalid input. Use an untouched holdout and interpret
the result for its sampled distribution. See [statistical scope](docs/design.md).

## Python API

```python
from calibroute import ActionCosts, PredictionRecord, fit_policy, route_batch

validation = [
    PredictionRecord("a", 0.98, True),
    PredictionRecord("b", 0.82, True),
    PredictionRecord("c", 0.55, False),
]

policy = fit_policy(validation, max_risk=0.10, min_coverage=0.50,
                    risk_method="empirical")  # Tiny illustrative sample only.
decisions, summary = route_batch(
    [PredictionRecord("new", 0.74)],
    policy,
)

# Cost objective: a wrong acceptance costs 5, a review or abstention costs 1.
cost_policy = fit_policy(validation, costs=ActionCosts(accept_error=5, human_review=1))
```

## Limitations

CalibRoute is an evaluation and routing tool, not a safety certification.
Thresholds should be revalidated after changes to the model, task, prompt, or
deployment distribution. See [design principles](docs/design.md) for details.

## Development

```bash
python -m unittest discover -s tests -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ROADMAP.md](ROADMAP.md).
Tried it on a task? Share a [usage report](https://github.com/Garyouki/calibroute/issues/new?template=usage-report.md),
including unsuccessful trials or integration difficulties.
Release preparation is documented in [releasing](docs/releasing.md).
The [Financial NER case study](examples/financial_ner/README.md) demonstrates
the adapter and cross-domain failure pattern on 2,098 derived prediction rows.
The [ShiftGuard agent example](examples/shiftguard/README.md) audits 384 frozen
proposals, keeps task templates separate, and reports when no confidence
threshold meets the declared empirical risk target. Use CalibRoute 0.4.0 or newer
for the `convert-shiftguard` command.

## Related work

CalibRoute builds on selective prediction and risk-controlled abstention.
Closely related work on cost-aware deferral for operational triage:

- Sachin Gupta. *Cost-Aware Abstention for LLM-Based IT Ticket Classification.*
  Accepted at IEEE ICTAI 2026. Project and artifact:
  [RouteGuard](https://github.com/sachinkg12/RouteGuard),
  [DOI 10.5281/zenodo.22609128](https://doi.org/10.5281/zenodo.22609128).

## License

MIT. See [LICENSE](LICENSE).
