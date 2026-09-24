"""Extract the documented model/seed slice from an author's local replay log."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from calibroute.adapters.shiftguard import read_shiftguard_records

MODEL = "qwen3:4b"
SEED = 13
TOP_FIELDS = (
    "model",
    "seed",
    "scenario_id",
    "domain",
    "task_name",
    "telemetry_profile",
    "scorer_version",
)


def make_fixture(source: Path, output: Path) -> dict:
    # Reject malformed or duplicate selected records before writing a fixture.
    records = read_shiftguard_records(source, model=MODEL, seed=SEED)
    source_bytes = source.read_bytes()
    selected, line_numbers = [], []
    total = 0
    for index, line in enumerate(source_bytes.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if row["model"] != MODEL or row["seed"] != SEED:
            continue
        projected = {key: row[key] for key in TOP_FIELDS}
        projected["proposal"] = {key: row["proposal"][key] for key in ("confidence", "valid_json")}
        projected["controllers"] = {
            "always_execute": {
                key: row["controllers"]["always_execute"][key]
                for key in ("task_success", "unsafe", "safe_task_success")
            }
        }
        selected.append(projected)
        line_numbers.append(index)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_nonempty_rows": total,
        "source_public_url": None,
        "selection": {"model": MODEL, "seed": SEED},
        "selection_note": "All rows for this fixed model/seed; no confidence/outcome filtering.",
        "source_line_numbers": line_numbers,
        "fixture": "traces.jsonl",
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "fixture_rows": len(records),
        "template_groups": sorted({record.metadata["template_group"] for record in records}),
        "scorer_versions": sorted({record.metadata["scorer_version"] for record in records}),
        "projection": {
            "top_level_fields": list(TOP_FIELDS),
            "proposal_fields": ["confidence", "valid_json"],
            "outcome_controller": "always_execute",
            "outcome_fields": ["task_success", "unsafe", "safe_task_success"],
        },
        "scope": "Frozen simulator outcomes; no new model inference or live tool execution.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces.jsonl").write_bytes(payload)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = make_fixture(args.input, args.output_dir)
    print(f"Extracted {manifest['fixture_rows']} records from {manifest['source_nonempty_rows']}")


if __name__ == "__main__":
    main()
