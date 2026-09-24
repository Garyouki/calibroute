# Roadmap

## 0.1 - Core control contract

- CSV and JSONL input
- confidence, calibration, and risk-coverage audit
- validation-only threshold fitting
- shift-first accept/review/abstain routing
- deterministic tests and GitHub Actions

## 0.2 - Financial NER reference adapter

- [x] convert encoder and generative sentence outputs into the common schema
- [x] reproduce multi-domain confidence and risk analyses
- [x] add exact one-sided binomial risk bounds
- [x] calibrate shift thresholds by batch size and handle small batches
- [x] add entity-level policies
- [x] add pinned external data acquisition manifest and historical hash verification

## 0.3 - Correctness and reproducible evaluation

- [x] tie-aware fitting and audit operating points
- [x] independent frozen-policy holdout assessment
- [x] O(n log n) tie-aware AUROC
- [x] input validation, package metadata, provenance, and build checks

## Future uncertainty monitoring

- pluggable embedding and feature-shift detectors
- bootstrap uncertainty for audit metrics
- drift alerts over time windows

## 0.4 - Agent and language-system adapters

- [x] ShiftGuard proposal adapter and offline template-separated confidence example
- structured-output and tool-call correctness adapters
- cost-sensitive escalation policies
- regression gates for persistent agent updates
- policy comparison reports

## 1.0 - Stable public API

- versioned policy and report schemas
- expanded documentation and examples
- independent reproducibility report
- PyPI release and archival DOI
