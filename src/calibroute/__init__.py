"""CalibRoute: uncertainty-aware evaluation and decision control."""

from .metrics import audit_records
from .models import Action, Decision, PredictionRecord
from .policy import GatePolicy, fit_policy, route_batch
from .shift import confidence_histogram, js_divergence

__all__ = [
    "Action",
    "Decision",
    "GatePolicy",
    "PredictionRecord",
    "audit_records",
    "confidence_histogram",
    "fit_policy",
    "js_divergence",
    "route_batch",
]

__version__ = "0.1.0"
