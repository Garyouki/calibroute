# Synthetic shift-monitoring study

Mode: full_protocol

Protocol SHA-256: `d082e1616d4fc133e95e5bff9f006206d2f858e5b273edf11f856586636dda82`

All data are simulated. Intervals are marginal 95% Wilson Monte Carlo intervals.
The population oracle knows the true score distribution; it is a diagnostic only.

## Unchanged distributions

| Profile | Reference N | Batch N | Alarms / trials | Alarm rate (95% interval) | Population oracle |
|---|---:|---:|---:|---|---:|
| high_confidence | 50 | 50 | 215/300 | 71.7% (66.3%–76.5%) | 6.3% |
| high_confidence | 50 | 300 | 171/300 | 57.0% (51.3%–62.5%) | 0.0% |
| high_confidence | 150 | 50 | 132/300 | 44.0% (38.5%–49.7%) | 5.0% |
| high_confidence | 150 | 300 | 28/300 | 9.3% (6.5%–13.2%) | 0.0% |
| high_confidence | 500 | 50 | 43/300 | 14.3% (10.8%–18.8%) | 3.7% |
| high_confidence | 500 | 300 | 0/300 | 0.0% (0.0%–1.3%) | 0.0% |
| three_modes | 50 | 50 | 49/300 | 16.3% (12.6%–20.9%) | 2.0% |
| three_modes | 50 | 300 | 16/300 | 5.3% (3.3%–8.5%) | 0.0% |
| three_modes | 150 | 50 | 21/300 | 7.0% (4.6%–10.5%) | 5.0% |
| three_modes | 150 | 300 | 1/300 | 0.3% (0.1%–1.9%) | 0.0% |
| three_modes | 500 | 50 | 10/300 | 3.3% (1.8%–6.0%) | 1.7% |
| three_modes | 500 | 300 | 0/300 | 0.0% (0.0%–1.3%) | 0.0% |
| uniform | 50 | 50 | 122/300 | 40.7% (35.3%–46.3%) | 8.0% |
| uniform | 50 | 300 | 205/300 | 68.3% (62.9%–73.3%) | 0.0% |
| uniform | 150 | 50 | 42/300 | 14.0% (10.5%–18.4%) | 6.0% |
| uniform | 150 | 300 | 25/300 | 8.3% (5.7%–12.0%) | 0.0% |
| uniform | 500 | 50 | 28/300 | 9.3% (6.5%–13.2%) | 5.0% |
| uniform | 500 | 300 | 0/300 | 0.0% (0.0%–1.3%) | 0.0% |

## Paired scenarios: reference 150, batch 300

| Profile | Scenario | Alarm rate | Fixed 0.90 coverage / risk | Threshold-only coverage / risk | Full coverage / risk | Full review rate | Additional reviews | Accepted errors withheld |
|---|---|---:|---|---|---|---:|---:|---:|
| high_confidence | benign_score_shift_30 | 100.0% | 66.67% / 2.28% | 67.06% / 2.31% | 0.00% / undefined | 100.00% | 89231 | 1396 |
| high_confidence | label_only | 9.3% | 94.87% / 32.28% | 95.42% / 32.31% | 86.59% / 32.32% | 10.37% | 8270 | 2554 |
| high_confidence | score_shift_05 | 72.3% | 90.20% / 2.27% | 90.73% / 2.30% | 25.31% / 2.21% | 72.63% | 64363 | 1378 |
| high_confidence | score_shift_15 | 100.0% | 80.77% / 2.27% | 81.24% / 2.30% | 0.00% / undefined | 100.00% | 89095 | 1683 |
| high_confidence | score_shift_30 | 100.0% | 66.67% / 2.28% | 67.06% / 2.31% | 0.00% / undefined | 100.00% | 89231 | 1396 |
| high_confidence | score_shift_60 | 100.0% | 38.21% / 2.35% | 38.45% / 2.38% | 0.00% / undefined | 100.00% | 89548 | 824 |
| high_confidence | stable | 9.3% | 94.87% / 2.25% | 95.42% / 2.28% | 86.59% / 2.31% | 10.37% | 8270 | 162 |
| three_modes | benign_score_shift_30 | 100.0% | 35.07% / 2.26% | 35.07% / 2.26% | 0.00% / undefined | 100.00% | 77349 | 714 |
| three_modes | label_only | 0.3% | 50.02% / 32.30% | 50.02% / 32.30% | 49.85% / 32.32% | 20.43% | 255 | 42 |
| three_modes | score_shift_05 | 48.3% | 47.59% / 2.29% | 47.59% / 2.29% | 24.86% / 2.37% | 58.30% | 35310 | 451 |
| three_modes | score_shift_15 | 100.0% | 42.55% / 2.27% | 42.55% / 2.27% | 0.00% / undefined | 100.00% | 74641 | 871 |
| three_modes | score_shift_30 | 100.0% | 35.07% / 2.26% | 35.07% / 2.26% | 0.00% / undefined | 100.00% | 77349 | 714 |
| three_modes | score_shift_60 | 100.0% | 20.01% / 2.26% | 20.01% / 2.26% | 0.00% / undefined | 100.00% | 82763 | 407 |
| three_modes | stable | 0.3% | 50.02% / 2.29% | 50.02% / 2.29% | 49.85% / 2.29% | 20.43% | 255 | 1 |
| uniform | benign_score_shift_30 | 100.0% | 6.81% / 1.99% | 13.74% / 4.28% | 0.00% / undefined | 100.00% | 77318 | 529 |
| uniform | label_only | 8.3% | 9.93% / 32.29% | 19.81% / 34.31% | 18.11% / 34.22% | 26.75% | 5943 | 541 |
| uniform | score_shift_05 | 17.7% | 9.43% / 2.06% | 18.85% / 4.21% | 15.51% / 4.18% | 33.43% | 12845 | 131 |
| uniform | score_shift_15 | 74.0% | 8.38% / 2.03% | 16.81% / 4.30% | 4.49% / 4.26% | 78.54% | 55242 | 478 |
| uniform | score_shift_30 | 100.0% | 6.81% / 1.99% | 13.74% / 4.28% | 0.00% / undefined | 100.00% | 77318 | 529 |
| uniform | score_shift_60 | 100.0% | 3.90% / 2.11% | 7.85% / 4.58% | 0.00% / undefined | 100.00% | 82708 | 324 |
| uniform | stable | 8.3% | 9.93% / 2.10% | 19.81% / 4.25% | 18.11% / 4.22% | 26.75% | 5943 | 71 |

## Interpretation limits

- Routing thresholds 0.90, 0.80, and review 0.60 were predeclared, not fitted.
- Pooled routing counts are descriptive, not independent per-record confidence intervals.
- Errors withheld means removed from automatic acceptance; it does not mean a reviewer corrected them.
- The gate also sends previously abstained rows to review, so additional reviews include those rows.
- Scenarios share random streams within a trial. Trials regenerate both reference and batch.
- Stable alarm rates are marginal over new references, not a guarantee for each deployed reference.
- Label-only changes preserve every score and must yield identical alarms to stable data.
- Synthetic findings do not classify the historical FIN alarm as a false positive.
- No defaults were optimized using these outcomes; no review time or production benefit was measured.
