"""Adapter for sentence-level outputs from the Financial NER experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from ..models import PredictionRecord

ENCODER_SIGNALS = {"sent_conf_msp"}
GENERATIVE_SIGNALS = {"conf_seq", "conf_tokens", "conf_min_span", "conf_min_sc"}


def read_financial_ner_records(
    path: str | Path,
    *,
    model: str,
    signal: str | None = None,
    seed: int | None = None,
) -> list[PredictionRecord]:
    """Convert sentence-level encoder or generative JSONL output."""

    if model not in {"encoder", "generative"}:
        raise ValueError("model must be encoder or generative")
    allowed = ENCODER_SIGNALS if model == "encoder" else GENERATIVE_SIGNALS
    signal = signal or ("sent_conf_msp" if model == "encoder" else "conf_min_sc")
    if signal not in allowed:
        raise ValueError(f"unsupported {model} confidence signal: {signal}")
    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if seed is not None and "seed" not in row:
                raise ValueError(f"line {line_number}: seed required when filtering")
            if seed is not None and int(row["seed"]) != seed:
                continue
            if signal not in row or row[signal] is None:
                raise ValueError(f"line {line_number}: missing confidence signal {signal}")
            if row.get("sent_error") not in (0, 1):
                raise ValueError(f"line {line_number}: sent_error must be 0 or 1")
            record_id = str(row.get("sent_id") or f"row-{line_number}")
            row_seed = row.get("seed")
            records.append(
                PredictionRecord(
                    record_id=f"{model}-{row_seed}-{record_id}",
                    confidence=float(row[signal]),
                    correct=not bool(int(row["sent_error"])),
                    domain=str(row.get("domain") or "default"),
                    metadata={"model": model, "seed": row_seed, "signal": signal},
                )
            )
    if not records:
        raise ValueError("no compatible records found")
    return records


def read_financial_ner_entity_records(
    path: str | Path,
    *,
    model: str,
    signal: str | None = None,
    seed: int | None = None,
    entities_key: str = "entities",
) -> list[PredictionRecord]:
    """Convert entity-level Financial NER outputs into the common schema.

    Each non-empty JSONL row must contain a list at ``entities_key``. Every
    entity must provide ``confidence`` (or the selected ``signal``) and a
    boolean ``correct``. ``entity_id`` is optional and falls back to the
    entity's zero-based position within its sentence. Optional ``type`` is
    retained as metadata; entity text is deliberately not read or written.
    """

    if model not in {"encoder", "generative"}:
        raise ValueError("model must be encoder or generative")
    allowed = ENCODER_SIGNALS if model == "encoder" else GENERATIVE_SIGNALS
    if signal is not None and signal not in allowed:
        raise ValueError(f"unsupported {model} confidence signal: {signal}")
    if not entities_key:
        raise ValueError("entities_key must not be empty")

    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if seed is not None and "seed" not in row:
                raise ValueError(f"line {line_number}: seed required when filtering")
            if seed is not None and int(row["seed"]) != seed:
                continue
            entities = row.get(entities_key)
            if not isinstance(entities, list):
                raise ValueError(f"line {line_number}: {entities_key} must be a list")
            sentence_id = str(row.get("sent_id") or f"row-{line_number}")
            row_seed = row.get("seed")
            for entity_index, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    raise ValueError(f"line {line_number}: entity {entity_index} must be an object")
                confidence_key = signal or "confidence"
                if confidence_key not in entity or entity[confidence_key] is None:
                    raise ValueError(
                        f"line {line_number}: entity {entity_index} missing {confidence_key}"
                    )
                if not isinstance(entity.get("correct"), bool):
                    raise ValueError(
                        f"line {line_number}: entity {entity_index} correct must be a boolean"
                    )
                entity_id = str(entity.get("entity_id") or entity_index)
                records.append(
                    PredictionRecord(
                        record_id=f"{model}-{row_seed}-{sentence_id}-{entity_id}",
                        confidence=float(entity[confidence_key]),
                        correct=entity["correct"],
                        domain=str(row.get("domain") or "default"),
                        metadata={
                            "model": model,
                            "seed": row_seed,
                            "signal": confidence_key,
                            "parent_sentence_id": sentence_id,
                            "entity_type": entity.get("type"),
                        },
                    )
                )
    if not records:
        raise ValueError("no compatible entity records found")
    return records


def write_records_csv(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    """Write common-schema records while retaining adapter provenance."""

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "confidence", "correct", "domain", "model", "seed", "signal"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.record_id,
                    "confidence": record.confidence,
                    "correct": int(bool(record.correct)),
                    "domain": record.domain,
                    "model": record.metadata.get("model"),
                    "seed": record.metadata.get("seed"),
                    "signal": record.metadata.get("signal"),
                }
            )
