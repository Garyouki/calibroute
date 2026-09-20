"""CSV/JSONL adapters for model-agnostic TrustGate records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import Decision, PredictionRecord
from .policy import GatePolicy


def _parse_correct(value: object) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "correct"}:
        return True
    if normalized in {"0", "false", "no", "incorrect"}:
        return False
    raise ValueError(f"cannot parse correctness label: {value!r}")


def _record_from_mapping(row: dict[str, object], row_number: int) -> PredictionRecord:
    known = {"id", "record_id", "confidence", "correct", "domain"}
    record_id = str(row.get("id") or row.get("record_id") or row_number)
    if "confidence" not in row:
        raise ValueError("input requires a 'confidence' column")
    return PredictionRecord(
        record_id=record_id,
        confidence=float(row["confidence"]),
        correct=_parse_correct(row.get("correct")),
        domain=str(row.get("domain") or "default"),
        metadata={key: value for key, value in row.items() if key not in known},
    )


def read_records(path: str | Path) -> list[PredictionRecord]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as handle:
            return [
                _record_from_mapping(dict(row), index)
                for index, row in enumerate(csv.DictReader(handle), start=1)
            ]
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if line.strip():
                    records.append(_record_from_mapping(json.loads(line), index))
        return records
    raise ValueError("input must be .csv, .jsonl, or .ndjson")


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_decisions(path: str | Path, decisions: Iterable[Decision]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "confidence", "action", "reason", "shift_score"]
        )
        writer.writeheader()
        writer.writerows(decision.as_dict() for decision in decisions)


def read_policy(path: str | Path) -> GatePolicy:
    return GatePolicy.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

