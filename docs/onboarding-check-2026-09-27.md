# Internal onboarding check — 2026-09-27

This is a maintainer-run installation and integration rehearsal. It uses a
synthetic tutorial and historical model outputs. It is not an external user
testimonial, a fresh model experiment, or evidence of production adoption.

## Environment and scope

- Windows, CPython 3.13.7, a newly created virtual environment.
- Installed `calibroute-ai==0.4.0` from `https://pypi.org/simple`, without an
  editable installation. The environment contained only `pip` and `calibroute-ai`.
- Invoked the environment's Python with `-I` from outside the source checkout;
  `calibroute.__file__` resolved to that environment's `site-packages`.
- Ran the document-review script from tag `v0.4.0`.
- Ran explicit CSV conversion, development audit, policy fitting, separate-split
  validation, comparison, and an unlabeled offline replay with the released CLI.
- The converter and updated tutorial were added after the release and are
  available from [the repository example](../examples/your_data/README.md).

## Friction found and addressed

1. A CSV containing `item_id,score,is_correct,task_domain` failed direct `audit`
   with exit code 2: `input requires a 'confidence' column`. Added a standalone
   converter with explicit column mappings, boolean-label or literal exact-match
   modes, and explicit unlabeled production mode. It rejects invalid scores,
   blank labels, duplicate IDs, ambiguous headers, and malformed rows before
   writing output. It does not infer task correctness or generate scores.
2. The document-review guide still pinned 0.3.0, while comparison requires 0.4.0.
   Updated the guide and documented clean installation of the published package.
3. A passing threshold assessment can coexist with zero automatic routing
   coverage. Documented the difference between `validate` (before the shift
   gate) and `compare` (the complete router).

## Synthetic walkthrough

The generated holdout had 300 rows. At threshold 0.97, 176 were accepted with
zero observed errors; the one-sided 95% risk upper bound was 1.69%, below the
declared 10% limit. The generated unlabeled production batch had 182 accepts,
62 reviews, and 56 abstentions. The deliberately shifted batch sent all 100
rows to review. These are synthetic workflow checks.

## Historical-data plan and provenance

The source was the original Financial NER repository's Git object
`70df7dca0c5c55509d34bd115ae5740b410d7752:experiment/results/encoder_sentences.jsonl`.
Its SHA-256 was `fdf15df0b49511847083dbf49d5dda9119bdce008757a5fb95e2f08a6a6224f2`.
The source was read from Git without restoring files in that repository.

Selected seed **13**, sentence confidence `sent_conf_msp`, and correctness
`sent_error == 0`. The foreign-column CSVs were constructed locally from these
historical fields to exercise mapping; they were not an external user's exports.
There were no prediction/label strings in this source, so the historical check
used correctness labels, not invented exact-match strings.

The plan recorded these choices before running this seed's evaluation:

| Setting | Value |
|---|---|
| Development | `fin_valid`, 150 rows |
| Evaluation | `fin_test`, 299 rows |
| Cross-domain diagnostic | `finer_ord_test`, 300 rows |
| Maximum accepted risk | 0.10 |
| Minimum fitting coverage | 0.10 |
| Fixed baseline | 0.90 |
| Fitting method | Default Clopper–Pearson, 95% confidence |
| Shift gate | Released defaults; no tuning after evaluation |

The files had zero overlapping sentence IDs across splits. Document/group
independence and prior exposure of these historical outputs were not established.
Consequently, this exercise does not establish the IID assumptions required for
a new risk certification. It did not retrain the model.

## Results

Fitting selected threshold **0.3514011204**. The frozen policy file SHA-256 was
`4441dbd06a8a359ebbbefda1eb633432fa3fcb33fb7ad56dff6d48638f8fa126`; it was unchanged
after evaluation. Constraints were not relaxed to improve the results.

### Threshold assessment before the shift gate

| Dataset | N | Accepted | Accepted errors | Observed accepted risk | 95% upper bound | CLI assessment |
|---|---:|---:|---:|---:|---:|---|
| FIN test | 299 | 281 | 14 | 4.98% | 7.68% | Pass (exit 0) |
| FiNER-ORD test | 300 | 226 | 81 | 35.84% | 41.43% | Fail (exit 1) |

These are the CLI's calculations under its stated holdout assumptions.

### Complete routing comparison

| Dataset | Rule | Accepted | Coverage | Accepted errors | Accepted risk | Review |
|---|---|---:|---:|---:|---:|---:|
| FIN test | Accept all | 299 | 100.00% | 27 | 9.03% | 0 |
| FIN test | Fixed 0.90 | 264 | 88.29% | 5 | 1.89% | 35 |
| FIN test | CalibRoute | 0 | 0.00% | 0 | Undefined | 299 |
| FiNER-ORD test | Accept all | 300 | 100.00% | 144 | 48.00% | 0 |
| FiNER-ORD test | Fixed 0.90 | 171 | 57.00% | 52 | 30.41% | 129 |
| FiNER-ORD test | CalibRoute | 0 | 0.00% | 0 | Undefined | 300 |

All strategies had zero abstentions in these two batches. FIN's shift score
was 0.02493 and FiNER-ORD's was 0.14383; both exceeded the effective gate threshold
0.02. The router sent every row to review. An unlabeled replay of the same FIN
evaluation rows reproduced the 299 review decisions; it was not new production data.

The complete router demonstrated **no review-work reduction** here. The fixed
0.90 baseline had useful observed behavior on FIN and a much higher error rate
on FiNER-ORD. These observations do not establish that either rule is universally
better. A confidence-shift alarm does not by itself prove accuracy degradation.

## Verification and next experiment

All 69 repository unit tests passed, including eight new converter tests. The
eight converter tests also passed in the isolated release environment. Ruff
lint and formatting checks passed for the CI scope including the new example.
The release workflow ran without installing development dependencies into the
clean package environment.

Next, investigate shift-gate behavior across several independently defined,
stable development batches and shifted development batches. Define acceptable
review capacity and compare fixed rules before collecting a new final evaluation
set. Keep these historical evaluation batches out of tuning. An external pilot
can then record installation friction, actual review effort, and outcomes; no
external pilot was performed in this check.
