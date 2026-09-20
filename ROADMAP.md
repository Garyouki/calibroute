# Roadmap

## 0.1 - Core control contract

- CSV and JSONL input
- confidence, calibration, and risk-coverage audit
- validation-only threshold fitting
- shift-first accept/review/abstain routing
- deterministic tests and GitHub Actions

## 0.2 - Financial NER reference adapter

- convert encoder and generative NER outputs into the common schema
- reproduce domain-shift risk-coverage analyses
- add entity- and sentence-level policies
- publish data acquisition manifests without redistributing restricted data

## 0.3 - Stronger uncertainty monitoring

- pluggable embedding and feature-shift detectors
- bootstrap uncertainty for audit metrics
- optional conservative risk bounds
- drift alerts over time windows

## 0.4 - Agent and language-system adapters

- structured-output and tool-call correctness adapters
- cost-sensitive escalation policies
- regression gates for persistent agent updates
- policy comparison reports

## 1.0 - Stable public API

- versioned policy and report schemas
- expanded documentation and examples
- independent reproducibility report
- PyPI release and archival DOI

