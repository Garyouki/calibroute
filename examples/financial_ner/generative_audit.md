# CalibRoute audit

## Overall

| Metric | Value |
|---|---:|
| count | 1049 |
| accuracy | 0.3889 |
| risk | 0.6111 |
| mean_confidence | 0.7827 |
| ece | 0.4051 |
| auroc | 0.6865 |
| aurc | 0.5119 |

## By domain

| Domain | Count | Accuracy | ECE | AUROC |
|---|---:|---:|---:|---:|
| fin_test | 299 | 0.6555 | 0.2247 | 0.7287 |
| fin_valid | 150 | 0.4667 | 0.4987 | 0.5548 |
| finer_ord_test | 300 | 0.4067 | 0.4180 | 0.6723 |
| tweetner7_test | 300 | 0.0667 | 0.5280 | 0.4406 |

## Selective operating points

| Coverage | Risk | Accuracy | Confidence threshold |
|---:|---:|---:|---:|
| 0.7112 | 0.4893 | 0.5107 | 1.0000 |
| 0.7112 | 0.4893 | 0.5107 | 1.0000 |
| 0.7598 | 0.5107 | 0.4893 | 0.6000 |
| 1.0000 | 0.6111 | 0.3889 | 0.0000 |

> These measurements describe the supplied data. They are not a safety guarantee,
> and thresholds must be revalidated after model, task, or distribution changes.
