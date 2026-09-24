# ShiftGuard: confidence on frozen agent proposals

This offline example shows how to audit agent confidence with CalibRoute and
recognize when a confidence threshold cannot satisfy a chosen error budget.
It uses recorded simulator outcomes from the author's ShiftGuard experiments.
It requires no model service, API key, network, GPU, or additional dependencies.

## Run

Install the current repository checkout with `python -m pip install -e .`.
The adapter is new in the repository and is not part of the 0.3.0 PyPI package.
From the repository root, run:

`python examples/shiftguard/run_case_study.py`

The generated `output/` directory contains a Markdown/JSON report, all converted
predictions, fitting/evaluation CSVs, a policy file, and a decisions CSV.
The report records the fitting constraints and template split.

The bundled example gives:

| Measurement | Result |
| --- | --- |
| Frozen proposals | 384 |
| Fitting / evaluation rows | 288 / 96 |
| Fitting / evaluation task templates | 12 / 4 |
| Mean reported confidence | 0.9515 |
| Valid, successful and safe under always-execute | 224 / 384 (58.33%) |
| ECE against that task-success label | 0.3682 |
| Fit with empirical risk <= 10%, coverage >= 10% | No feasible policy |

The final row is intentional: the example does not relax the constraints after
looking at results. `policy.json` is null and `decisions.csv` has only its
header; routing has not run. The example exits successfully because the audit
completed. A caller must inspect `policy_attempt.status` before using a policy.
Increasing `--max-risk` changes the question and is not evidence that the original
10% target was met.

## Convert your own ShiftGuard replay log

The CLI reads full replay logs or this example's minimal projection. All input
must include recorded `always_execute` outcomes. From the repository root:

```bash
calibroute convert-shiftguard --input examples/shiftguard/traces.jsonl --output predictions.csv
calibroute audit --input predictions.csv --output audit.md
```

Use `--model` and `--seed` to select an exact model tag and decoding seed.
For a new log with at least two task templates in each domain, run the complete
example with `--input your-replay.jsonl --output-dir your-output`.

Python callers can import `read_shiftguard_records` and `write_shiftguard_csv`
from `calibroute.adapters.shiftguard`. CSV exports retain model, seed, scenario,
task, template group, telemetry profile, scorer version, and label definition.
CalibRoute's existing `read_records` reads them back with metadata intact.

## Adapter contract

Every selected JSONL row must contain:

- Nonempty `model`, `scenario_id`, `domain`, `task_name`, `telemetry_profile`,
  and `scorer_version` strings, plus an integer `seed`.
- `proposal.confidence`: a finite number in [0, 1], and a boolean
  `proposal.valid_json`. Missing confidence is an error, not a fabricated zero.
- Boolean `task_success`, `unsafe`, and `safe_task_success` under
  `controllers.always_execute`.

The adapter requires `safe_task_success == (task_success and not unsafe)` and
sets `correct = valid_json and safe_task_success`. An unsuccessful action with
no harm remains incorrect. The adapter rejects duplicate identities, malformed
labels and contradictory outcomes; it does not silently drop difficult cases.

The original prompt asked for the probability that an action was **safe to
commit now**. This example evaluates the stricter event **valid, successful and
safe**. ECE here measures alignment with that utility target, not calibration
against the prompt's original event. The confidence is a model-reported score,
not a measured probability or an automatically trusted risk bound.

Labels come from the original simulator scorer under `always_execute`. They
are retrospective evaluation information, never online features. The example
strips labels and metadata before calling the router when fitting is feasible.
It does not use a verifier's improved outcome as the original proposal's label.

## Data origin and splitting

`traces.jsonl` contains **all** rows for model tag `qwen3:4b`, seed 13, from
`artifacts/main/trajectories_v2_guards_replayed.jsonl` in the author's local
ShiftGuard (`agent3.0`) artifact. No row was selected by confidence or outcome.
This is one illustrative model/seed slice, not a model comparison. The original
source has 3,456 rows; its SHA-256, selected line numbers, projection fields,
and fixture hash are in `source_manifest.json`. No public source URL is asserted.

The projection includes confidence, grouping identifiers and the three relevant
outcome flags. Prompts, raw model text, resource payloads, authoritative state,
hidden fault labels, other controllers and machine paths are omitted. The
frozen scorer version is `2026-08-24-guards-v1`. This example does not independently
validate that scorer, rerun the simulator, or collect new model predictions.

The original source owner can regenerate the projection with:

`python examples/shiftguard/make_fixture.py --input PATH_TO_ORIGINAL_REPLAY.jsonl --output-dir NEW_DIRECTORY`

All variants of a `(domain, task_name)` template stay together, including
different instances, faults, telemetry profiles, models and seeds. Within each
domain the lexicographically last task is held out; the remaining tasks fit the
threshold. This rule reads identifiers only, not labels or confidence.

The records are correlated synthetic scenarios from a previously inspected
experiment. Template separation reduces direct template overlap but does not
make this an untouched IID test. The example deliberately uses **empirical**
fitting and makes no binomial confidence-bound or deployment-safety claim.
Human-review outcomes and live agent effects are not measured.
