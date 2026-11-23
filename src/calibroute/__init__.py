"""CalibRoute: uncertainty-aware evaluation and decision control."""

from .bounds import clopper_pearson_lower, clopper_pearson_upper
from .comparison import compare_policies
from .costs import ActionCosts
from .metrics import audit_records
from .models import Action, Decision, PredictionRecord
from .policy import GatePolicy, InfeasiblePolicyError, fit_policy, route_batch
from .shift import calibrate_js_threshold, confidence_histogram, js_divergence
from .validation import validate_policy

__all__ = [
    "Action",
    "ActionCosts",
    "Decision",
    "GatePolicy",
    "InfeasiblePolicyError",
    "PredictionRecord",
    "audit_records",
    "calibrate_js_threshold",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "compare_policies",
    "confidence_histogram",
    "fit_policy",
    "js_divergence",
    "route_batch",
    "validate_policy",
]

__version__ = "0.6.0"
