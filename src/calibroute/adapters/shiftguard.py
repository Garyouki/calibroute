"""Convert frozen ShiftGuard proposals and always-execute outcomes for evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

from ..models import PredictionRecord

LABEL_DEFINITION = "valid_json_and_task_success_and_not_unsafe_under_always_execute"
METADATA_FIELDS = [
    "model",
    "seed",
    "scenario_id",
    "task_name",
    "template_group",
    "telemetry_profile",
    "scorer_version",
    "valid_json",
    "task_success",
    "unsafe",
    "signal",
    "label_definition",
]


def _text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _boolean(row: dict, key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _object(row: dict, key: str) -> dict:
    value = row.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def read_shiftguard_records(
    path: str | Path, *, model: str | None = None, seed: int | None = None
) -> list[PredictionRecord]:
    """Read full replay logs or their documented minimal projection.

    Correctness concerns the *unmodified proposal* under ``always_execute``.
    A refusal with no harm is not automatically a successful task. Oracle
    outcomes supply evaluation labels only; no outcome enters confidence.
    Duplicate proposal identities are rejected rather than overweighted.
    """
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model filter must be a nonempty string")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed filter must be an integer")
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("record must be an object")
                row_model = _text(row, "model")
                row_seed = row.get("seed")
                if type(row_seed) is not int:
                    raise ValueError("seed must be an integer")
                if (model is not None and row_model != model) or (
                    seed is not None and row_seed != seed
                ):
                    continue
                scenario = _text(row, "scenario_id")
                domain = _text(row, "domain")
                task = _text(row, "task_name")
                telemetry = _text(row, "telemetry_profile")
                scorer = _text(row, "scorer_version")
                proposal = _object(row, "proposal")
                confidence = proposal.get("confidence")
                if (
                    type(confidence) not in (int, float)
                    or not 0 <= confidence <= 1
                    or not math.isfinite(confidence)
                ):
                    raise ValueError("proposal.confidence must be a finite number in [0, 1]")
                valid = _boolean(proposal, "valid_json")
                outcome = _object(_object(row, "controllers"), "always_execute")
                success = _boolean(outcome, "task_success")
                unsafe = _boolean(outcome, "unsafe")
                safe_success = _boolean(outcome, "safe_task_success")
                if safe_success != (success and not unsafe):
                    raise ValueError("safe_task_success contradicts task_success/unsafe")
                identity = (row_model, str(row_seed), scenario, telemetry, scorer)
                record_id = "shiftguard:" + ":".join(quote(part, safe="") for part in identity)
                if record_id in seen:
                    raise ValueError("duplicate proposal identity")
                seen.add(record_id)
                records.append(
                    PredictionRecord(
                        record_id=record_id,
                        confidence=float(confidence),
                        correct=valid and safe_success,
                        domain=domain,
                        metadata={
                            "model": row_model,
                            "seed": row_seed,
                            "scenario_id": scenario,
                            "task_name": task,
                            "template_group": json.dumps([domain, task], separators=(",", ":")),
                            "telemetry_profile": telemetry,
                            "scorer_version": scorer,
                            "valid_json": valid,
                            "task_success": success,
                            "unsafe": unsafe,
                            "signal": "proposal.confidence",
                            "label_definition": LABEL_DEFINITION,
                        },
                    )
                )
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"line {line_number}: {error}") from error
    if not records:
        raise ValueError("no compatible ShiftGuard records found")
    return records


def write_shiftguard_csv(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    """Preserve grouping and label provenance through CSV round trips."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "confidence", "correct", "domain", *METADATA_FIELDS]
        )
        writer.writeheader()
        for record in records:
            if record.correct is None:
                raise ValueError("ShiftGuard evaluation export requires correctness labels")
            writer.writerow(
                {
                    "id": record.record_id,
                    "confidence": record.confidence,
                    "correct": int(record.correct),
                    "domain": record.domain,
                    **{name: record.metadata[name] for name in METADATA_FIELDS},
                }
            )
