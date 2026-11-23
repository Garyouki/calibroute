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

Operationally, `human_review` and `abstain` differ in what happens next:

| Action | Automatic output used? | Enters the review queue? | Typical handling |
|---|---|---|---|
| `accept` | Yes | No | The output is acted on automatically. |
| `human_review` | No, but shown to the reviewer as a suggestion | Yes | A reviewer confirms or corrects the model output. |
| `abstain` | No, discarded | No | The item falls back to the application's default path, such as manual triage from scratch, a slower queue, or a "cannot decide" response. |

An abstained item is not free: if it still has to be resolved, its cost is
usually at least one review. That is why `ActionCosts.abstain` defaults to the
review cost. Declare a different abstention cost only when the fallback path
really costs more or less, and state that assumption with any cost result.
`review_margin` sets how far below the acceptance threshold the model output
is still considered useful enough to show a reviewer.

## Objectives: coverage or cost

`fit_policy` supports two declared objectives:

- Coverage (default, `max_risk` only): the largest accepted set whose
  accepted-error criterion is at or below `max_risk`.
- Cost (`costs=ActionCosts(...)`): the threshold with the lowest total declared
  cost on the validation records. `max_risk`, if also given, stays a hard
  constraint.

A risk limit and a cost ratio are not interchangeable. A 10% risk limit
accepts every record up to the point where accepted errors would exceed 10%,
even when each marginal acceptance costs more than a review. With costs of 5
for a wrong acceptance and 1 for review, accepting a record is only worthwhile
where its error probability is below 1/5 = 20%; the cost objective picks the
threshold where that trade-off is best on the validation records, whatever
accepted error rate results. Choose the objective that matches the decision you need
to defend, and compare rules on the same declared costs with
`calibroute compare --cost-accept-error ... --cost-review ...`.

## No feasible policy

When no threshold meets the declared constraints, `fit_policy` raises
`InfeasiblePolicyError`, a `ValueError` subclass, whose `report` attribute
records the closest attainable operating point at the required coverage and
the largest feasible coverage below it. The CLI prints the same report to
stderr and exits 2. Pass `on_infeasible="all_human"` (CLI:
`--on-infeasible all-human`) to save an explicit policy that routes every
record to human review instead. The constraints are never relaxed silently.

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

`validate` reports a `verdict` alongside the original `passed` flag:

- `pass`: the holdout upper bound is at or below the limit (`passed` is true).
- `inconclusive`: the upper bound is above the limit but the lower bound is
  not, so the holdout cannot show that the limit is violated.
- `fail`: the holdout lower bound is above the limit, which is evidence that the
  accepted error rate exceeds it.

`inconclusive` is common for policies fitted at the edge of the limit. In a
simulation with 4,784 fitting and 9,567 holdout records and a 10% limit, the
fitted threshold's true accepted risk exceeded 10% in about 5% of runs, yet
the holdout upper bound exceeded 10% in about a third of runs. To expect a
`pass` at limit L, fit with a stricter `max_risk` (for example 0.09 for
L = 0.10) or use a larger holdout. The CLI exit code is unchanged: 0 only for
`pass`.

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

The [finite-reference simulation](../examples/shift_study/README.md) measures
these limitations with a protocol recorded before evaluation. Under unchanged
synthetic distributions, small empirical references produced frequent alarms;
label-only accuracy losses produced identical alarms to stable paired trials.
The results include routing costs and do not change the default 0.02 floor.

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
