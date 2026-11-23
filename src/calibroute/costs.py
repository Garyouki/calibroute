"""Explicit per-action costs for cost-aware threshold selection and reporting."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from .models import Action, PredictionRecord


@dataclass(frozen=True)
class ActionCosts:
    """Cost units charged for each routing outcome.

    ``accept_error`` is charged when an accepted output is wrong and
    ``accept_correct`` when it is right. ``human_review`` is charged for every
    record routed to a reviewer. ``abstain`` is charged for every record that
    receives no automatic output and no review queue entry; it defaults to the
    review cost because, in most workflows, an abstained item still has to be
    resolved by someone. Set it explicitly when abstention means something
    cheaper (dropping a low-value item) or more expensive (escalation).

    Units are whatever the caller declares. They are assumptions, not measured
    savings; report them together with any cost result.
    """

    accept_error: float
    human_review: float
    abstain: float | None = None
    accept_correct: float = 0.0

    def __post_init__(self) -> None:
        for name in ("accept_error", "human_review", "accept_correct"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} cost must be a finite non-negative number")
        if self.abstain is not None and (not math.isfinite(self.abstain) or self.abstain < 0):
            raise ValueError("abstain cost must be a finite non-negative number")
        if self.accept_error <= self.accept_correct:
            raise ValueError("accept_error cost must exceed accept_correct cost")

    @property
    def abstain_cost(self) -> float:
        return self.human_review if self.abstain is None else self.abstain

    def as_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["abstain"] = self.abstain_cost
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ActionCosts:
        abstain = payload.get("abstain")
        return cls(
            accept_error=float(payload["accept_error"]),
            human_review=float(payload["human_review"]),
            abstain=None if abstain is None else float(abstain),
            accept_correct=float(payload.get("accept_correct", 0.0)),
        )


def total_cost(
    records: Sequence[PredictionRecord], actions: Iterable[Action], costs: ActionCosts
) -> float:
    """Sum declared costs for labeled records and their routed actions."""

    total = 0.0
    for row, action in zip(records, actions, strict=True):
        if action == Action.ACCEPT:
            if row.correct is None:
                raise ValueError("cost of accepted records requires labels")
            total += costs.accept_correct if row.correct else costs.accept_error
        elif action == Action.HUMAN_REVIEW:
            total += costs.human_review
        else:
            total += costs.abstain_cost
    return total
