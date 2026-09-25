# CalibRoute

CalibRoute is a lightweight Python toolkit for confidence evaluation and
uncertainty-aware routing. It converts model predictions into three actions:
`accept`, `human_review`, or `abstain`.

## Features

- Confidence calibration and error-ranking metrics
- Risk-coverage analysis
- Validation-only threshold fitting
- Tie-aware threshold fitting and independent holdout risk assessment
- Batch-size-aware confidence-shift detection
- Financial NER sentence- and entity-level output adapters
- ShiftGuard agent-replay adapter and offline confidence audit (repository version)
- CSV and JSONL support
- Dependency-free Python API and CLI

## Install

Requires Python 3.10 or newer.

Install with `python -m pip install calibroute-ai`.

To run the checked-in examples or contribute, clone the repository:

```bash
git clone https://github.com/Garyouki/calibroute.git
cd calibroute
python -m pip install -e .
```

The distribution name is `calibroute-ai`; the Python import and command are
`calibroute`. No API key or model service is needed.

## Quick start

New here? Follow the [document-review walkthrough](examples/document_review/README.md):
one offline script demonstrates development data, a separate holdout, and routing
using the published 0.3.0 API. It also explains how to use your own predictions.

```bash
# Audit labeled predictions
calibroute audit \
  --input examples/validation.csv \
  --output examples/audit.md

# Fit an acceptance policy on validation data
calibroute fit \
  --input examples/validation.csv \
  --max-risk 0.20 \
  --min-coverage 0.25 \
  --risk-method empirical \
  --output examples/policy.json

# Route a new prediction batch
calibroute route \
  --input examples/production_batch.csv \
  --policy examples/policy.json \
  --output examples/decisions.csv \
  --summary examples/routing-summary.json
```

## Input format

| Field | Required | Description |
|---|---|---|
| `id` | recommended | Prediction identifier |
| `confidence` | yes | Number between 0 and 1 |
| `correct` | `audit` and `fit` only | Boolean or `1`/`0` label |
| `domain` | no | Evaluation slice or deployment domain |

Additional columns are preserved as metadata.

The default fit searches thresholds using pointwise 95% Clopper-Pearson bounds.
Threshold selection on the same labels does **not** provide a 95% guarantee for
the selected policy. Freeze the policy, then assess it on a separate IID holdout:

```bash
calibroute validate --input holdout.csv --policy examples/policy.json --output holdout-report.json
```

Exit code 0 means the holdout upper bound meets the declared risk limit; 1 means
it does not (including zero accepted samples); 2 means invalid input. Never tune
against this holdout or reuse it to select among multiple policies. A pass does
not cover distribution shift. See [statistical scope](docs/design.md).

## Python API

```python
from calibroute import PredictionRecord, fit_policy, route_batch

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
threshold meets the declared empirical risk target. Install from this checkout
to use the new `convert-shiftguard` command.

## License

MIT. See [LICENSE](LICENSE).
