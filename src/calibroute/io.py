"""CSV/JSONL adapters for model-agnostic CalibRoute records."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

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


def _record_from_mapping(
    row: dict[str, object], row_number: int, *, require_id: bool
) -> PredictionRecord:
    known = {"id", "record_id", "confidence", "correct", "domain"}
    record_id = row.get("id")
    if record_id is None or record_id == "":
        record_id = row.get("record_id")
    if record_id is None or not str(record_id).strip():
        if require_id:
            raise ValueError("holdout records require an explicit prediction ID (id or record_id)")
        record_id = row_number
    if "confidence" not in row:
        raise ValueError("input requires a 'confidence' column")
    confidence = row["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float, str)):
        raise TypeError("confidence must be a number between 0 and 1")
    try:
        score = float(confidence)
    except (ValueError, OverflowError) as error:
        raise ValueError("confidence must be a number between 0 and 1") from error
    return PredictionRecord(
        record_id=str(record_id),
        confidence=score,
        correct=_parse_correct(row.get("correct")),
        domain=str(row.get("domain") or "default"),
        metadata={key: value for key, value in row.items() if key not in known},
    )


def read_records(path: str | Path, *, require_id: bool = False) -> list[PredictionRecord]:
    """Read prediction records, optionally requiring IDs supplied by the caller."""
    source = Path(path)
    suffix = source.suffix.lower()
    records = []
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                try:
                    records.append(_record_from_mapping(row, index, require_id=require_id))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"line {reader.line_num}: {error}") from error
        return records
    if suffix in {".jsonl", ".ndjson"}:
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError("prediction record must be a JSON object")
                    records.append(_record_from_mapping(row, index, require_id=require_id))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"line {index}: {error}") from error
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        if not isinstance(payload, dict):
            raise TypeError("policy must be a JSON object")
        return GatePolicy.from_dict(payload)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"invalid policy value: {error}") from error
