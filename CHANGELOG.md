# Changelog

## Unreleased

- Add `compare` reports for all-accept, fixed-threshold/review, and CalibRoute
  routing on the same labeled batch, including domain slices and undefined-risk handling.
- Add a ShiftGuard replay-log adapter and `convert-shiftguard` CLI command.
- Preserve task-template grouping and explicit safe-success label provenance in CSV exports.
- Add an offline agent example with a frozen 384-record slice, source hashes,
  template-separated empirical fitting, and explicit infeasible-policy reporting.

## 0.3.0 - 2026-09-20

- Fix tie groups in threshold fitting, risk-coverage curves, and selective reports.
- Add independent holdout validation with CLI pass/fail exit codes.
- Clarify pointwise fit bounds versus independent fixed-policy risk assessment.
- Replace quadratic AUROC with tie-aware O(n log n) rank aggregation.
- Make missing shift threshold default to adaptive monitoring; validate inputs.
- Regenerate case-study reports and record provenance and reproducibility limits.
- Align package/citation versions; expand CI to lint, build, and check distributions.

Refit 0.2 policies before deployment. The 0.2 policy selection and audit reports
could split ties and depend on input order. No 0.2 PyPI release is intended.

## 0.2.0 - 2026-09-19

- Add exact Clopper-Pearson upper risk bounds to policy fitting.
- Add batch-size-aware JSD calibration and explicit small-batch handling.
- Add a Financial NER sentence-output adapter and multi-domain case study.
- Expand policy schema to record declared risk and calibration settings.

## 0.1.0 - 2026-09-19

- Initial model-agnostic confidence audit.
- Validation-grounded selective policy fitting.
- Batch confidence-shift detection.
- Auditable accept, human-review, and abstain routing.
- Dependency-free Python API and CLI.
