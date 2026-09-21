# Design principles

## Shift first, confidence second

A model can be confidently wrong when deployment inputs differ from validation
data. CalibRoute therefore performs a batch-level shift check before applying an
instance-level confidence threshold. Severe shift results in human review rather
than silently trusting a threshold learned under a different regime.

## Validation-only fitting

Policies are fitted on labeled validation records. Test or production labels are
for evaluation, not threshold selection. The serialized policy records its
achieved validation risk and coverage so an operating point can be audited.

## Bounded actions

CalibRoute emits one of three explicit actions:

- `accept`: permit the configured downstream automation;
- `human_review`: pause automation and request qualified review;
- `abstain`: withhold the output when confidence is below the review band.

CalibRoute itself does not execute downstream actions. The calling application
decides what each action is allowed to do.

## Observable reasons

Every decision carries a stable reason code and the batch shift score. This is
more useful for audit than returning a binary answer without the control path.

## Current boundary

CalibRoute uses the Jensen-Shannon divergence between validation and deployment
confidence histograms as a transparent baseline. A confidence distribution is
not a complete representation of input shift. Planned adapters will add feature,
embedding, and task-specific shift detectors without changing the control API.

## Statistical scope

`fit_policy` evaluates only complete confidence groups: its reported accepted
set is exactly `confidence >= accept_threshold`. Equal scores are never split.
The same rule applies to audit curves. Requested coverage rounds upward to the
next attainable threshold; AURC is the right-step integral weighted by coverage
increments, not the average over unique thresholds.

The Clopper-Pearson bound is an exact one-sided binomial interval for a fixed
acceptance rule on independent identically distributed labeled examples.
Searching thresholds on those same labels invalidates a claim of nominal
coverage for the selected threshold. The fit bound is therefore a selection
diagnostic. `validate_policy` / `calibroute validate` assess a frozen threshold
on a separate IID holdout. The caller must enforce independence; the software
cannot establish it from identifiers. Repeated policy selection against the
holdout also invalidates the interpretation. No accepted examples means no
evidence and a failed validation, not zero risk.

The validation API assesses threshold acceptance before the batch shift gate.
It does not certify the conditional risk of the full batch-dependent pipeline,
future domains, correlated records, or changing models/prompts.

JSD calibration simulates batches from the empirical reference histogram. It
does not account for uncertainty in that reference or provide a distribution-
free false-alarm guarantee. The default 0.02 effect floor is a heuristic and
must be chosen on development data for the application. Batches smaller than
20 are marked `insufficient_batch` and use instance thresholds; callers may
instead queue them or require review. Unchanged confidence histograms can hide
large correctness changes. Monitoring confidence cannot replace labeled audits.

## Migrating from 0.2

Refit policies and regenerate audits. Old policies can retain thresholds selected
inside ties. Missing `max_js_divergence` now means adaptive monitoring, matching
the constructor; specify `0.10` explicitly to retain that older default.
Policy schema is 1.2 and audit schema is 1.1. Earlier AURC and selective-point
values with ties are not directly comparable to the corrected reports.

AUROC uses an O(n log n) sort and tie-aware rank aggregation. Exact binomial
threshold search can still be expensive for many unique scores and errors;
the dependency-free implementation targets modest validation sets. Use the
empirical mode for exploratory large audits, then validate a frozen threshold.
