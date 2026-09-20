"""Lightweight confidence-distribution shift monitoring."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def confidence_histogram(confidences: Iterable[float], bins: int = 10) -> list[float]:
    """Return a normalized histogram over the fixed interval [0, 1]."""

    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = list(confidences)
    if not values:
        raise ValueError("at least one confidence value is required")
    counts = [0] * bins
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        index = min(int(value * bins), bins - 1)
        counts[index] += 1
    total = len(values)
    return [count / total for count in counts]


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in nats, bounded by ln(2)."""

    if len(p) != len(q) or not p:
        raise ValueError("distributions must be non-empty and have equal length")
    if any(value < 0 for value in (*p, *q)):
        raise ValueError("distribution values must be non-negative")
    p_total, q_total = sum(p), sum(q)
    if p_total <= 0 or q_total <= 0:
        raise ValueError("each distribution must have positive mass")
    pn = [value / p_total for value in p]
    qn = [value / q_total for value in q]
    midpoint = [(left + right) / 2 for left, right in zip(pn, qn)]

    def kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(a * math.log(a / b) for a, b in zip(left, right) if a > 0)

    return 0.5 * kl_divergence(pn, midpoint) + 0.5 * kl_divergence(qn, midpoint)

