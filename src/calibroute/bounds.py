"""Distribution-free risk bounds for selective prediction."""

from __future__ import annotations

import math


def _binomial_cdf(errors: int, total: int, probability: float) -> float:
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 1.0 if errors >= total else 0.0
    log_p, log_q = math.log(probability), math.log1p(-probability)
    terms = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        + index * log_p
        + (total - index) * log_q
        for index in range(errors + 1)
    ]
    peak = max(terms)
    return math.exp(peak) * sum(math.exp(term - peak) for term in terms)


def clopper_pearson_upper(errors: int, total: int, confidence_level: float = 0.95) -> float:
    """Return an exact one-sided upper confidence bound on error probability."""

    if total <= 0 or not 0 <= errors <= total:
        raise ValueError("require total > 0 and 0 <= errors <= total")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if errors == total:
        return 1.0
    alpha = 1.0 - confidence_level
    low, high = errors / total, 1.0
    for _ in range(70):
        midpoint = (low + high) / 2
        if _binomial_cdf(errors, total, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2
