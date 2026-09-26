# Financial NER case study

This case study applies the same evaluation contract to sentence-level outputs
from an encoder NER system and a generative NER system. It spans an in-domain
financial test set plus two shifted domains: FiNER-ORD and TweetNER7.

The checked-in CSV files contain only derived prediction metadata (identifier,
confidence, correctness, domain, model, seed, and signal). They contain no
source sentences or dataset text.

To acquire and verify the original data splits, inspect exact prompts/model
configuration, or review the recovered training and evaluation scripts, see
[experiment sources and data preparation](reproduction/README.md). The data
preparation command runs without model dependencies; model retraining remains
a separate, not-yet-validated path.

## Results

| Model / confidence signal | Domain | n | Accuracy | ECE | AUROC |
|---|---|---:|---:|---:|---:|
| Encoder / min of span-mean MSP | FIN validation | 150 | 0.940 | 0.037 | 0.787 |
| Encoder / min of span-mean MSP | FIN test | 299 | 0.880 | 0.050 | 0.905 |
| Encoder / min of span-mean MSP | FiNER-ORD | 300 | 0.480 | 0.265 | 0.742 |
| Encoder / min of span-mean MSP | TweetNER7 | 300 | 0.083 | 0.519 | 0.558 |
| Generative / self-consistency | FIN validation | 150 | 0.467 | 0.499 | 0.555 |
| Generative / self-consistency | FIN test | 299 | 0.656 | 0.225 | 0.729 |
| Generative / self-consistency | FiNER-ORD | 300 | 0.407 | 0.418 | 0.672 |
| Generative / self-consistency | TweetNER7 | 300 | 0.067 | 0.528 | 0.441 |

The result is the point of the project: confidence that looks useful in-domain
can become badly miscalibrated after a domain change. A trustworthy system must
measure that change and escalate or abstain instead of silently treating every
high-confidence output as reliable.

Fitting on FIN validation gives 99.3% validation coverage, 5.4% observed risk,
and a 9.5% pointwise upper bound used during selection. This is not a
selection-adjusted 95% guarantee. The
batch-aware shift gate allows the closely matched FIN test batch to proceed,
while routing all FiNER-ORD and TweetNER7 outputs to human review.

The generated `routing_summary.json` also reports accepted-set empirical risk
and fixed-threshold binomial bounds. The shifted domains have no automatically
accepted outputs, so accepted risk is null, not zero. Human-review capacity and
the accuracy of reviewers are not measured. See [provenance](PROVENANCE.md) for
the distinction between replaying these predictions and reproducing the models.

| Test slice | Threshold-only accepted | Errors among them | Final automatic accepts | Final review |
|---|---:|---:|---:|---:|
| FIN | 278 / 299 | 19 / 278 (6.8%) | 278 | 16 |
| FiNER-ORD | 222 / 300 | 90 / 222 (40.5%) | 0 | 300 |
| TweetNER7 | 160 / 300 | 145 / 160 (90.6%) | 0 | 300 |

The shift gate trades automation coverage for review workload. It does not
improve the underlying NER predictions. FIN additionally has five abstentions.

The 0.02 shift effect floor was inspected against these test batches during
development. This example is retrospective; fresh held-out data is required
before making a deployment reliability claim.

Reproduce the reports from the repository root:

```bash
calibroute audit --input examples/financial_ner/encoder_seed42.csv --output examples/financial_ner/encoder_audit.md
calibroute audit --input examples/financial_ner/generative_self_consistency_seed42.csv --output examples/financial_ner/generative_audit.md
python examples/financial_ner/run_case_study.py
```

To convert a new experiment output:

```bash
calibroute convert-financial-ner --input sentences.jsonl --model encoder --seed 42 --output predictions.csv
calibroute convert-financial-ner --input sentences.jsonl --model generative --signal conf_min_sc --output predictions.csv
```

## Entity-level routing

Requires CalibRoute 0.4.0 or newer. Upgrade with
`python -m pip install --upgrade calibroute-ai`. See the
[version compatibility table](../../README.md#version-compatibility).

For entity-level evaluation, each JSONL row can contain an `entities` list. An
entity needs a boolean `correct` and a `confidence` field, or a selected
model-specific confidence signal. `entity_id` and `type` are optional; entity
text is neither required nor exported. The converter emits one common-schema
record per entity, so the existing `audit`, `fit`, `validate`, and `route`
commands apply without a separate policy format.

```json
{"seed": 42, "sent_id": "example-1", "domain": "fin_test", "entities": [{"entity_id": "e1", "confidence": 0.93, "correct": true, "type": "ORG"}]}
```

```bash
calibroute convert-financial-ner-entities \
  --input entity_predictions.jsonl --model encoder --seed 42 \
  --output entity_predictions.csv
```

The entity-level correctness definition must be fixed before policy fitting;
do not mix relaxed span overlap and exact-match labels in a single audit.

These are empirical research results, not claims of deployment safety. Dataset
and model licensing remains the responsibility of users who reproduce the
underlying experiments.
