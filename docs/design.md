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

Version 0.1 uses the Jensen-Shannon divergence between validation and deployment
confidence histograms as a transparent baseline. A confidence distribution is
not a complete representation of input shift. Planned adapters will add feature,
embedding, and task-specific shift detectors without changing the control API.
