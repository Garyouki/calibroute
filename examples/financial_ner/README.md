# Financial NER case study

This case study applies the same evaluation contract to sentence-level outputs
from an encoder NER system and a generative NER system. It spans an in-domain
financial test set plus two shifted domains: FiNER-ORD and TweetNER7.

The checked-in CSV files contain only derived prediction metadata (identifier,
confidence, correctness, domain, model, seed, and signal). They contain no
source sentences or dataset text.

## Results

| Model / confidence signal | Domain | n | Accuracy | ECE | AUROC |
|---|---|---:|---:|---:|---:|
| Encoder / mean span MSP | FIN validation | 150 | 0.940 | 0.037 | 0.787 |
| Encoder / mean span MSP | FIN test | 299 | 0.880 | 0.050 | 0.905 |
| Encoder / mean span MSP | FiNER-ORD | 300 | 0.480 | 0.265 | 0.742 |
| Encoder / mean span MSP | TweetNER7 | 300 | 0.083 | 0.519 | 0.558 |
| Generative / self-consistency | FIN validation | 150 | 0.467 | 0.499 | 0.555 |
| Generative / self-consistency | FIN test | 299 | 0.656 | 0.225 | 0.729 |
| Generative / self-consistency | FiNER-ORD | 300 | 0.407 | 0.418 | 0.672 |
| Generative / self-consistency | TweetNER7 | 300 | 0.067 | 0.528 | 0.441 |

The result is the point of the project: confidence that looks useful in-domain
can become badly miscalibrated after a domain change. A trustworthy system must
measure that change and escalate or abstain instead of silently treating every
high-confidence output as reliable.

Fitting the default 95% risk-bound policy on FIN validation gives 99.3%
validation coverage, 5.4% observed risk, and a 9.5% upper risk bound. The
batch-aware shift gate allows the closely matched FIN test batch to proceed,
while routing all FiNER-ORD and TweetNER7 outputs to human review.

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

These are empirical research results, not claims of deployment safety. Dataset
and model licensing remains the responsibility of users who reproduce the
underlying experiments.
