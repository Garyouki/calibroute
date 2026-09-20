import unittest

from calibroute.models import Action, PredictionRecord
from calibroute.policy import GatePolicy, fit_policy, route_batch


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.validation = [
            PredictionRecord("a", 0.99, True),
            PredictionRecord("b", 0.95, True),
            PredictionRecord("c", 0.90, True),
            PredictionRecord("d", 0.80, False),
            PredictionRecord("e", 0.60, False),
            PredictionRecord("f", 0.30, False),
        ]

    def test_fit_uses_largest_valid_prefix(self):
        policy = fit_policy(
            self.validation,
            max_risk=0.0,
            min_coverage=0.3,
            histogram_bins=5,
            risk_method="empirical",
        )
        self.assertEqual(policy.accept_threshold, 0.90)
        self.assertEqual(policy.validation_coverage, 0.5)

    def test_fit_rejects_impossible_constraint(self):
        with self.assertRaises(ValueError):
            fit_policy(self.validation, max_risk=0.0, min_coverage=0.8, risk_method="empirical")

    def test_normal_routing_has_three_actions(self):
        policy = fit_policy(
            self.validation,
            max_risk=0.0,
            min_coverage=0.3,
            review_margin=0.2,
            histogram_bins=5,
            max_js_divergence=1.0,
            risk_method="empirical",
        )
        rows = [
            PredictionRecord("accept", 0.95),
            PredictionRecord("review", 0.75),
            PredictionRecord("abstain", 0.20),
        ]
        decisions, summary = route_batch(rows, policy)
        self.assertEqual(
            [decision.action for decision in decisions],
            [Action.ACCEPT, Action.HUMAN_REVIEW, Action.ABSTAIN],
        )
        self.assertFalse(summary["severe_shift"])

    def test_shift_routes_everything_to_review(self):
        policy = GatePolicy(
            accept_threshold=0.8,
            review_threshold=0.6,
            max_validation_risk=0.1,
            validation_coverage=0.5,
            reference_histogram=[1.0, 0.0],
            histogram_bins=2,
            max_js_divergence=0.01,
            shift_min_batch_size=1,
        )
        rows = [PredictionRecord("x", 0.95), PredictionRecord("y", 0.90)]
        decisions, summary = route_batch(rows, policy)
        self.assertTrue(summary["severe_shift"])
        self.assertTrue(all(item.action is Action.HUMAN_REVIEW for item in decisions))

    def test_policy_round_trip(self):
        policy = fit_policy(
            self.validation, max_risk=0.25, histogram_bins=5, risk_method="empirical"
        )
        self.assertEqual(GatePolicy.from_dict(policy.as_dict()), policy)

    def test_default_fit_records_conservative_bound(self):
        rows = [PredictionRecord(str(index), 1 - index / 200, True) for index in range(100)]
        policy = fit_policy(rows, max_risk=0.05, min_coverage=0.5)
        self.assertEqual(policy.risk_method, "clopper_pearson")
        self.assertLessEqual(policy.risk_upper_bound, 0.05)
        self.assertEqual(policy.declared_risk_limit, 0.05)

    def test_conservative_fit_rejects_insufficient_evidence(self):
        rows = [PredictionRecord(str(index), 0.9, True) for index in range(10)]
        with self.assertRaises(ValueError):
            fit_policy(rows, max_risk=0.1)

    def test_calibrated_shift_detects_clear_change(self):
        policy = GatePolicy(
            accept_threshold=0.8,
            review_threshold=0.6,
            max_validation_risk=0.0,
            validation_coverage=1.0,
            reference_histogram=[1.0, 0.0],
            histogram_bins=2,
            shift_min_batch_size=20,
            shift_resamples=100,
        )
        decisions, summary = route_batch(
            [PredictionRecord(str(index), 0.9) for index in range(20)], policy
        )
        self.assertTrue(summary["severe_shift"])
        self.assertTrue(all(item.action is Action.HUMAN_REVIEW for item in decisions))

    def test_small_batch_skips_shift_decision(self):
        policy = GatePolicy(
            accept_threshold=0.8,
            review_threshold=0.6,
            max_validation_risk=0.1,
            validation_coverage=0.5,
            reference_histogram=[1.0, 0.0],
            histogram_bins=2,
            max_js_divergence=0.0,
            shift_min_batch_size=3,
        )
        decisions, summary = route_batch([PredictionRecord("x", 0.95)], policy)
        self.assertFalse(summary["shift_evaluable"])
        self.assertEqual(summary["shift_status"], "insufficient_batch")
        self.assertEqual(decisions[0].action, Action.ACCEPT)


if __name__ == "__main__":
    unittest.main()
