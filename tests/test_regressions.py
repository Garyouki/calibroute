import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

from calibroute.cli import main
from calibroute.metrics import risk_coverage, roc_auc, selective_points
from calibroute.models import Action, PredictionRecord
from calibroute.policy import GatePolicy, fit_policy, route_batch
from calibroute.validation import validate_policy


class RegressionTests(unittest.TestCase):
    def test_rejects_invalid_labels_and_histograms(self):
        with self.assertRaises(ValueError):
            PredictionRecord("a", 0.9, "false")
        policy = fit_policy([PredictionRecord("a", 1, True)], max_risk=0.1, risk_method="empirical")
        for histogram in ([0.0] * 10, [float("nan")] * 10, [-1.0] * 10):
            payload = policy.as_dict()
            payload["reference_histogram"] = histogram
            with self.assertRaises(ValueError):
                GatePolicy.from_dict(payload)

    def test_ties_cannot_hide_errors_in_either_mode(self):
        rows = [PredictionRecord(str(i), 0.9, i < 6) for i in range(11)]
        for method in ("empirical", "clopper_pearson"):
            for order in (rows, list(reversed(rows))):
                with self.assertRaises(ValueError):
                    fit_policy(order, max_risk=0.4, risk_method=method)

    def test_fit_matches_route_and_is_permutation_invariant(self):
        rows = [PredictionRecord(str(i), 0.9 if i < 40 else 0.5, i < 35) for i in range(60)]
        for method in ("empirical", "clopper_pearson"):
            baseline = fit_policy(rows, max_risk=0.3, risk_method=method, max_js_divergence=1)
            for seed in range(10):
                shuffled = rows.copy()
                random.Random(seed).shuffle(shuffled)
                self.assertEqual(
                    baseline,
                    fit_policy(shuffled, max_risk=0.3, risk_method=method, max_js_divergence=1),
                )
            decisions, _ = route_batch(rows, baseline)
            accepted = [
                row for row, decision in zip(rows, decisions) if decision.action == Action.ACCEPT
            ]
            self.assertEqual(baseline.validation_coverage, len(accepted) / len(rows))
            self.assertEqual(
                baseline.max_validation_risk,
                sum(not row.correct for row in accepted) / len(accepted),
            )

    def test_metrics_match_deployable_tie_groups(self):
        rows = [PredictionRecord(str(i), 0.8, i < 6) for i in range(11)]
        points, area = risk_coverage(rows)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(area, 5 / 11)
        self.assertEqual(risk_coverage(rows), risk_coverage(reversed(rows)))
        self.assertEqual(selective_points(rows, [0.25])[0]["coverage"], 1)

    def test_rank_auc_matches_pairwise_reference(self):
        rng = random.Random(42)
        for _ in range(30):
            rows = [
                PredictionRecord(str(i), rng.choice([0, 0.2, 0.5, 1]), bool(rng.randrange(2)))
                for i in range(40)
            ]
            pos = [r.confidence for r in rows if r.correct]
            neg = [r.confidence for r in rows if not r.correct]
            expected = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            self.assertAlmostEqual(roc_auc(rows), expected / (len(pos) * len(neg)))

    def test_missing_shift_default_matches_constructor(self):
        policy = fit_policy([PredictionRecord("a", 1, True)], max_risk=0.1, risk_method="empirical")
        payload = policy.as_dict()
        del payload["max_js_divergence"]
        self.assertEqual(GatePolicy.from_dict(payload), policy)

    def test_holdout_failure_success_and_no_acceptances(self):
        policy = fit_policy(
            [PredictionRecord("a", 0.9, True)], max_risk=0.1, risk_method="empirical"
        )
        for correct, passed in ((True, True), (False, False)):
            rows = [PredictionRecord(str(i), 0.95, correct) for i in range(100)]
            self.assertEqual(validate_policy(rows, policy)["passed"], passed)
        empty = validate_policy([PredictionRecord("b", 0.1, True)], policy)
        self.assertFalse(empty["passed"])
        self.assertIsNone(empty["risk_upper_bound"])

    def test_cli_validation_exit_code(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = fit_policy(
                [PredictionRecord("a", 0.9, True)], max_risk=0.1, risk_method="empirical"
            )
            (root / "policy.json").write_text(json.dumps(policy.as_dict()), encoding="utf-8")
            (root / "holdout.csv").write_text("id,confidence,correct\nb,1,0\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "validate",
                        "--input",
                        str(root / "holdout.csv"),
                        "--policy",
                        str(root / "policy.json"),
                        "--output",
                        str(root / "result.json"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertFalse(json.loads((root / "result.json").read_text())["passed"])
