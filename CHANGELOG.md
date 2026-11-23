# Changelog

## 0.6.0 - 2026-11-22

- Add declared action costs (`ActionCosts`) and a cost objective to `fit_policy`
  and `calibroute fit` (`--cost-accept-error`, `--cost-review`, `--cost-abstain`,
  `--cost-accept-correct`). `max_risk` is optional when costs are declared and
  remains a hard constraint when both are given.
- Report declared cost per record in `compare`, and in `validate` for cost policies.
- Add a `verdict` (`pass`, `inconclusive`, `fail`, `no_accepted`) and a
  Clopper-Pearson lower bound to holdout validation. `passed` and CLI exit codes
  are unchanged.
- Raise `InfeasiblePolicyError` (a `ValueError` subclass) with a structured
  report when no threshold is feasible; the CLI prints it to stderr. Add an
  opt-in all-human fallback (`on_infeasible="all_human"`, `--on-infeasible all-human`).
  In cost mode, a fit whose best threshold costs more than all-human review is
  reported as `all_human_is_cheaper` instead of returning a dominated policy.
- Document the fit, validate, route workflow in the quick start, define
  `human_review` versus `abstain` operationally, and add related work.
- Policy schema 1.3 adds `objective`, `costs`, `validation_expected_cost`,
  `validation_all_human_cost`, `accept_disabled`, and `infeasibility`; schema 1.2
  policies load unchanged.
- Thanks to Sachin Gupta for an independent trial of 0.5.1 on saved IT-ticket
  predictions that motivated these changes.

## 0.5.1 - 2026-11-21

- Add Zenodo metadata so each release is archived with a citable DOI. No code changes.

## 0.5.0 - 2026-11-19

- Require an explicit, unique `id` or `record_id` for every holdout prediction
  passed to the `validate` CLI. Duplicate or missing IDs produce exit code 2.
- Validate frozen policies at their stored confidence level by default.
  Use `--confidence-level` to select a different level explicitly.
- Report observed risk for empirical fits, with `null` as the statistical
  upper bound, including when reading older empirical policies.
- Report malformed prediction records with input line numbers and handle
  invalid policy values as CLI input errors.
- Separate ShiftGuard recorded safety, task success, safe task success, and
  valid safe-success reporting with explicit event definitions and per-target ECE.
- Add outcome counts for fitting/evaluation splits and domains to example report
  schema 2; retain the original combined policy label and fitting constraints.
- Add a walkthrough for mapping and evaluating prediction CSVs from other tools.
- Document a frozen synthetic study of shift alarms and review workload.

## 0.4.0 - 2026-09-25

- Add entity-level Financial NER conversion with `convert-financial-ner-entities`.
- Add a document-review walkthrough, usage feedback template, and version compatibility guide.
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
