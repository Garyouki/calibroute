# Contributing

Contributions are welcome, especially new metrics, shift-monitor adapters,
task-specific examples, and tests that expose unsafe assumptions.

## Local workflow

1. Create a focused branch.
2. Install the package with `python -m pip install -e .`.
3. Run `python -m unittest discover -s tests -v`.
4. Add tests for every behavior change.
5. Document whether a new metric is descriptive, empirically calibrated, or
   accompanied by a formal guarantee.

Please do not commit private, proprietary, personally identifiable, or
license-restricted datasets. Prefer download scripts, hashes, and synthetic
fixtures for examples.

## Pull requests

Keep pull requests narrow. Explain the uncertainty assumption, expected failure
mode, and how the change affects accept/review/abstain decisions.

