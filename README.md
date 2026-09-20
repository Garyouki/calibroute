# TrustGate

**Uncertainty-aware evaluation and decision control for AI systems.**

TrustGate turns model confidence into an auditable control decision:

```text
prediction -> confidence audit -> batch shift check -> accept | human_review | abstain
```

The project addresses a practical trustworthy-AI question:

> When should an AI system trust its own output, and what should it do when it should not?

TrustGate is model- and task-agnostic. It accepts ordinary CSV or JSONL files,
fits thresholds only on labeled validation data, detects confidence-distribution
shift before instance-level routing, and records a reason for every action.

## Why this exists

Accuracy or F1 alone does not tell a deployment system which individual outputs
are safe to automate. A confidence score also cannot be assumed reliable after
the input distribution changes. TrustGate separates three responsibilities:

1. **Evaluation:** Does confidence rank and calibrate errors?
2. **Uncertainty monitoring:** Does the production batch resemble the validation regime?
3. **Decision control:** Should the output be accepted, reviewed by a person, or withheld?

The initial use case is financial named-entity recognition under domain shift,
but the interfaces apply to classifiers, extractors, language systems, and agents
that emit a scalar confidence signal.

TrustGate is maintained by Zihao Zheng and is being developed as reusable
research software for trustworthy AI evaluation and control under uncertainty.

## Install

TrustGate requires Python 3.10 or newer and has no runtime dependencies.

```bash
python -m pip install -e .
```

## Five-minute example

Audit labeled validation predictions:

```bash
trustgate audit \
  --input examples/validation.csv \
  --output examples/audit.md
```

Fit the highest-coverage empirical operating point with validation risk at or
below 20%:

```bash
trustgate fit \
  --input examples/validation.csv \
  --max-risk 0.20 \
  --min-coverage 0.25 \
  --output examples/policy.json
```

Route a new, unlabeled batch:

```bash
trustgate route \
  --input examples/production_batch.csv \
  --policy examples/policy.json \
  --output examples/decisions.csv \
  --summary examples/routing-summary.json
```

Each decision contains an action and reason. If batch-level shift exceeds the
declared limit, TrustGate routes the whole batch to human review before applying
instance-level thresholds.

## Input schema

CSV and JSONL are supported.

| Field | Required | Meaning |
|---|---|---|
| `id` | recommended | Stable prediction identifier; row number is the fallback |
| `confidence` | yes | Number in `[0, 1]`; larger means more likely correct |
| `correct` | for `audit` and `fit` | `true/false`, `1/0`, or `correct/incorrect` |
| `domain` | no | Slice or deployment domain; defaults to `default` |
| other fields | no | Preserved as metadata when records are loaded in Python |

## Python API

```python
from trustgate import PredictionRecord, fit_policy, route_batch

validation = [
    PredictionRecord("a", confidence=0.98, correct=True),
    PredictionRecord("b", confidence=0.82, correct=True),
    PredictionRecord("c", confidence=0.55, correct=False),
]

policy = fit_policy(validation, max_risk=0.10, min_coverage=0.50)
decisions, summary = route_batch(
    [PredictionRecord("new", confidence=0.74)],
    policy,
)
```

## Reported evidence

- Accuracy and error risk
- Expected calibration error (ECE)
- Confidence AUROC for separating correct and incorrect outputs
- Area under the empirical risk-coverage curve (AURC)
- Selective risk at declared coverage levels
- Per-domain slices
- Jensen-Shannon shift score for confidence distributions
- Accept, review, and abstain counts with reason codes

## Safety and scope

TrustGate is an evaluation and routing aid, not a safety certification. Empirical
validation risk is not a statistical or regulatory guarantee. Confidence-only
shift detection cannot detect every semantic or adversarial distribution change.
Thresholds must be refitted or revalidated after changes to the model, prompt,
task, label definition, population, or deployment environment.

For high-impact uses, combine TrustGate with domain-specific review, stronger
shift signals, documented error costs, privacy controls, and qualified human
oversight. See [design principles](docs/design.md) for the intended evidence and
control boundaries.

## Development

```bash
python -m unittest discover -s tests -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ROADMAP.md](ROADMAP.md).

## Research alignment

TrustGate supports a broader research program in **Trustworthy AI Evaluation and
Control Under Uncertainty**, with emphasis on reliability, evaluation, selective
prediction, distribution shift, and bounded decision authority for AI agents and
language systems. See [research alignment](docs/research-alignment.md).

## License

MIT. See [LICENSE](LICENSE).
