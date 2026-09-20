"""Small, model-agnostic data structures used by TrustGate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    """A downstream control action for one AI output."""

    ACCEPT = "accept"
    HUMAN_REVIEW = "human_review"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction and the confidence assigned to it.

    ``correct`` is optional because production routing often happens before a
    gold label exists. Evaluation commands require labels; routing does not.
    """

    record_id: str
    confidence: float
    correct: bool | None = None
    domain: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.record_id:
            raise ValueError("record_id must not be empty")


@dataclass(frozen=True)
class Decision:
    """A control decision with an auditable reason."""

    record_id: str
    confidence: float
    action: Action
    reason: str
    shift_score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.record_id,
            "confidence": self.confidence,
            "action": self.action.value,
            "reason": self.reason,
            "shift_score": self.shift_score,
        }

