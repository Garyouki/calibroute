# CalibRoute

CalibRoute is a lightweight Python toolkit for confidence evaluation and
uncertainty-aware routing. It converts model predictions into three actions:
`accept`, `human_review`, or `abstain`.

## Features

- Confidence calibration and error-ranking metrics
- Risk-coverage analysis
- Validation-only threshold fitting
- Batch confidence-shift detection
- CSV and JSONL support
- Dependency-free Python API and CLI

## Install

Requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

## Quick start

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

## Python API

```python
from calibroute import PredictionRecord, fit_policy, route_batch

validation = [
    PredictionRecord("a", 0.98, True),
    PredictionRecord("b", 0.82, True),
    PredictionRecord("c", 0.55, False),
]

policy = fit_policy(validation, max_risk=0.10, min_coverage=0.50)
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

## License

MIT. See [LICENSE](LICENSE).
