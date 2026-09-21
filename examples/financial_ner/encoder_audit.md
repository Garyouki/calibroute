# CalibRoute audit

## Overall

| Metric | Value |
|---|---:|
| count | 1049 |
| accuracy | 0.5462 |
| risk | 0.4538 |
| mean_confidence | 0.7854 |
| ece | 0.2392 |
| auroc | 0.7378 |
| aurc | 0.3230 |

## By domain

| Domain | Count | Accuracy | ECE | AUROC |
|---|---:|---:|---:|---:|
| fin_test | 299 | 0.8796 | 0.0496 | 0.9052 |
| fin_valid | 150 | 0.9400 | 0.0374 | 0.7872 |
| finer_ord_test | 300 | 0.4800 | 0.2647 | 0.7419 |
| tweetner7_test | 300 | 0.0833 | 0.5186 | 0.5576 |

## Selective operating points

| Coverage | Risk | Accuracy | Confidence threshold |
|---:|---:|---:|---:|
| 0.5424 | 0.3111 | 0.6889 | 1.0000 |
| 0.5424 | 0.3111 | 0.6889 | 1.0000 |
| 0.7502 | 0.3088 | 0.6912 | 0.4218 |
| 1.0000 | 0.4538 | 0.5462 | 0.1809 |

> These measurements describe the supplied data. They are not a safety guarantee,
> and thresholds must be revalidated after model, task, or distribution changes.
