import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from calibroute import (
    ActionCosts,
    GatePolicy,
    InfeasiblePolicyError,
    PredictionRecord,
    clopper_pearson_lower,
    clopper_pearson_upper,
    compare_policies,
    fit_policy,
    route_batch,
    validate_policy,
)
from calibroute.cli import main
from calibroute.models import Action


def ladder():
    # Ten records; errors concentrate at low confidence.
    pattern = [True, True, True, True, False, True, False, True, False, False]
    return [
        PredictionRecord(f"r{i}", round(0.95 - 0.05 * i, 2), correct)
        for i, correct in enumerate(pattern)
    ]


class ActionCostsTests(unittest.TestCase):
    def test_abstain_defaults_to_review_and_round_trips(self):
        costs = ActionCosts(accept_error=5, human_review=1)
        self.assertEqual(costs.abstain_cost, 1)
        self.assertEqual(ActionCosts.from_dict(costs.as_dict()).as_dict(), costs.as_dict())

    def test_rejects_invalid_costs(self):
        for kwargs in (
            {"accept_error": -1, "human_review": 1},
            {"accept_error": 1, "human_review": float("nan")},
            {"accept_error": 1, "human_review": 1, "abstain": -0.5},
            {"accept_error": 1, "human_review": 1, "accept_correct": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActionCosts(**kwargs)


class CostFitTests(unittest.TestCase):
    def brute_force(self, rows, costs, margin):
        best = None
        for threshold in sorted({row.confidence for row in rows}, reverse=True):
            review_threshold = max(0.0, threshold - margin)
            cost = 0.0
            for row in rows:
                if row.confidence >= threshold:
                    cost += costs.accept_correct if row.correct else costs.accept_error
                elif row.confidence >= review_threshold:
                    cost += costs.human_review
                else:
                    cost += costs.abstain_cost
            if best is None or cost <= best[1]:
                best = threshold, cost
        return best

    def test_cost_fit_matches_brute_force(self):
        rows = ladder()
        for costs, margin in (
            (ActionCosts(accept_error=5, human_review=1), 0.15),
            (ActionCosts(accept_error=2, human_review=1, abstain=0.2), 0.10),
            (ActionCosts(accept_error=10, human_review=1, abstain=0.5), 0.05),
        ):
            with self.subTest(costs=costs, margin=margin):
                policy = fit_policy(rows, costs=costs, min_coverage=0.1, review_margin=margin)
                threshold, cost = self.brute_force(rows, costs, margin)
                self.assertEqual(policy.objective, "min_cost")
                self.assertEqual(policy.accept_threshold, threshold)
                self.assertAlmostEqual(policy.validation_expected_cost, cost / len(rows))
                self.assertIsNone(policy.declared_risk_limit)

    def test_cost_ratio_changes_the_threshold(self):
        rows = ladder()
        cheap = fit_policy(rows, costs=ActionCosts(accept_error=1.5, human_review=1))
        strict = fit_policy(rows, costs=ActionCosts(accept_error=20, human_review=1))
        self.assertGreater(cheap.validation_coverage, strict.validation_coverage)

    def test_max_risk_remains_a_constraint_in_cost_mode(self):
        rows = ladder()
        unconstrained = fit_policy(rows, costs=ActionCosts(accept_error=1.2, human_review=1))
        constrained = fit_policy(
            rows,
            costs=ActionCosts(accept_error=1.2, human_review=1),
            max_risk=0.0,
            risk_method="empirical",
        )
        self.assertGreater(unconstrained.max_validation_risk, 0.0)
        self.assertEqual(constrained.max_validation_risk, 0.0)
        self.assertEqual(constrained.accept_threshold, 0.80)

    def test_coverage_objective_is_unchanged_without_costs(self):
        rows = ladder()
        policy = fit_policy(rows, max_risk=0.0, min_coverage=0.1, risk_method="empirical")
        self.assertEqual(policy.objective, "max_coverage")
        self.assertIsNone(policy.costs)
        self.assertIsNone(policy.validation_expected_cost)
        self.assertEqual(policy.accept_threshold, 0.80)

    def test_requires_a_risk_limit_or_costs(self):
        with self.assertRaisesRegex(ValueError, "max_risk, costs, or both"):
            fit_policy(ladder())

    def test_cost_policy_round_trips(self):
        policy = fit_policy(ladder(), costs=ActionCosts(accept_error=5, human_review=1))
        self.assertEqual(GatePolicy.from_dict(json.loads(json.dumps(policy.as_dict()))), policy)

    def test_older_policy_payload_still_loads(self):
        payload = fit_policy(ladder(), max_risk=0.5, risk_method="empirical").as_dict()
        for key in (
            "objective",
            "costs",
            "validation_expected_cost",
            "validation_all_human_cost",
            "accept_disabled",
            "infeasibility",
        ):
            del payload[key]
        payload["schema_version"] = "1.2"
        policy = GatePolicy.from_dict(payload)
        self.assertEqual(policy.objective, "max_coverage")
        self.assertFalse(policy.accept_disabled)


class InfeasibleTests(unittest.TestCase):
    def test_structured_report_is_a_value_error(self):
        with self.assertRaises(InfeasiblePolicyError) as caught:
            fit_policy(ladder(), max_risk=0.0, min_coverage=0.8, risk_method="empirical")
        self.assertIsInstance(caught.exception, ValueError)
        report = caught.exception.report
        self.assertEqual(report["status"], "no_feasible_policy")
        self.assertEqual(report["min_accepted"], 8)
        self.assertEqual(report["validation_errors"], 4)
        self.assertAlmostEqual(report["closest_at_or_above_min_coverage"]["observed_risk"], 2 / 8)
        self.assertEqual(report["largest_feasible_below_min_coverage"]["accepted"], 4)
        json.dumps(report)

    def test_all_human_fallback_routes_everything_to_review(self):
        policy = fit_policy(
            ladder(),
            max_risk=0.0,
            min_coverage=0.8,
            risk_method="empirical",
            on_infeasible="all_human",
        )
        self.assertTrue(policy.accept_disabled)
        self.assertEqual(policy.objective, "all_human_fallback")
        self.assertEqual(policy.validation_coverage, 0.0)
        self.assertEqual(policy.infeasibility["status"], "no_feasible_policy")
        rows = [PredictionRecord("top", 1.0), PredictionRecord("low", 0.1)]
        decisions, _ = route_batch(rows, policy)
        self.assertEqual({d.action for d in decisions}, {Action.HUMAN_REVIEW})
        self.assertEqual(GatePolicy.from_dict(json.loads(json.dumps(policy.as_dict()))), policy)
        holdout = [PredictionRecord(str(i), 0.99, True) for i in range(10)]
        report = validate_policy(holdout, policy)
        self.assertEqual(report["verdict"], "all_human_fallback")
        self.assertEqual(report["accepted"], 0)

    def test_cost_mode_never_returns_a_threshold_worse_than_all_human(self):
        # Every record has confidence 0.9 and only 40% are correct: accepting the
        # single attainable group costs 0.6 * 5 = 3 per record, review costs 1.
        rows = [PredictionRecord(f"t{i}", 0.9, i < 4) for i in range(10)]
        costs = ActionCosts(accept_error=5, human_review=1)
        with self.assertRaises(InfeasiblePolicyError) as caught:
            fit_policy(rows, costs=costs)
        report = caught.exception.report
        self.assertEqual(report["status"], "all_human_is_cheaper")
        self.assertAlmostEqual(report["cheapest_threshold"]["cost_per_record"], 3.0)
        self.assertEqual(report["cheapest_threshold"]["all_human_cost_per_record"], 1)
        fallback = fit_policy(rows, costs=costs, on_infeasible="all_human")
        self.assertTrue(fallback.accept_disabled)
        self.assertEqual(fallback.validation_expected_cost, 1)

    def test_rejects_unknown_infeasible_mode(self):
        with self.assertRaises(ValueError):
            fit_policy(ladder(), max_risk=0.1, on_infeasible="relax")


class VerdictTests(unittest.TestCase):
    def policy(self, limit=0.10):
        return GatePolicy(
            accept_threshold=0.5,
            review_threshold=0.35,
            max_validation_risk=limit,
            validation_coverage=1.0,
            reference_histogram=[1.0] * 10,
            declared_risk_limit=limit,
            risk_method="clopper_pearson",
        )

    def holdout(self, total, errors):
        return [PredictionRecord(str(i), 0.9, i >= errors) for i in range(total)]

    def test_lower_bound_matches_complement_identity(self):
        self.assertEqual(clopper_pearson_lower(0, 50), 0.0)
        self.assertAlmostEqual(clopper_pearson_lower(50, 50), 0.05 ** (1 / 50))
        lower = clopper_pearson_lower(948, 9169)
        upper = clopper_pearson_upper(948, 9169)
        self.assertLess(lower, 948 / 9169)
        self.assertLess(948 / 9169, upper)

    def test_pass_inconclusive_and_fail_are_separated(self):
        policy = self.policy()
        cases = {
            "pass": self.holdout(1000, 70),
            "inconclusive": self.holdout(1000, 105),
            "fail": self.holdout(1000, 160),
        }
        for expected, rows in cases.items():
            with self.subTest(expected=expected):
                report = validate_policy(rows, policy)
                self.assertEqual(report["verdict"], expected)
                self.assertEqual(report["passed"], expected == "pass")

    def test_edge_fitted_case_is_inconclusive_not_fail(self):
        # Mirrors an external trial: 948 errors among 9,169 accepted, limit 10%.
        report = validate_policy(self.holdout(9169, 948), self.policy())
        self.assertEqual(report["verdict"], "inconclusive")
        self.assertFalse(report["passed"])

    def test_no_accepted_verdict(self):
        report = validate_policy([PredictionRecord("a", 0.1, True)], self.policy())
        self.assertEqual(report["verdict"], "no_accepted")
        self.assertIsNone(report["risk_lower_bound"])

    def test_cost_only_policy_reports_holdout_cost(self):
        policy = fit_policy(ladder(), costs=ActionCosts(accept_error=5, human_review=1))
        report = validate_policy(ladder(), policy)
        self.assertEqual(report["verdict"], "no_risk_limit")
        self.assertAlmostEqual(report["cost_per_record"], policy.validation_expected_cost)
        self.assertEqual(report["all_human_cost_per_record"], 1)


class CostCompareAndCliTests(unittest.TestCase):
    def test_compare_reports_cost_per_rule(self):
        rows = ladder()
        policy = fit_policy(rows, max_risk=0.0, risk_method="empirical", max_js_divergence=1.0)
        report = compare_policies(
            rows,
            policy,
            fixed_threshold=0.9,
            costs=ActionCosts(accept_error=5, human_review=1),
        )
        overall = report["overall"]
        self.assertAlmostEqual(overall["accept_all"]["cost_per_record"], 4 * 5 / 10)
        # Two correct acceptances at >= 0.9; eight records go to review.
        self.assertAlmostEqual(overall["fixed_threshold"]["cost_per_record"], 8 / 10)
        self.assertAlmostEqual(overall["calibroute"]["cost_per_record"], 6 / 10)

    def test_compare_without_costs_has_no_cost_fields(self):
        rows = ladder()
        policy = fit_policy(rows, max_risk=0.0, risk_method="empirical", max_js_divergence=1.0)
        report = compare_policies(rows, policy, fixed_threshold=0.9)
        self.assertIsNone(report["costs"])
        self.assertNotIn("cost_per_record", report["overall"]["accept_all"])

    def write_rows(self, path, rows):
        lines = ["id,confidence,correct"]
        lines += [f"{r.record_id},{r.confidence},{int(r.correct)}" for r in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_cli_cost_fit_validate_and_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_rows(root / "dev.csv", ladder())
            common = ["--cost-accept-error", "5", "--cost-review", "1"]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = main(
                    ["fit", "--input", str(root / "dev.csv"), "--output", str(root / "p.json")]
                    + common
                )
            self.assertEqual(code, 0)
            self.assertIn("objective=min_cost", out.getvalue())
            saved = json.loads((root / "p.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["costs"]["accept_error"], 5)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "validate",
                        "--input",
                        str(root / "dev.csv"),
                        "--policy",
                        str(root / "p.json"),
                        "--output",
                        str(root / "v.json"),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("cost_per_record", json.loads((root / "v.json").read_text()))
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "compare",
                        "--input",
                        str(root / "dev.csv"),
                        "--policy",
                        str(root / "p.json"),
                        "--fixed-threshold",
                        "0.9",
                        "--output",
                        str(root / "c.md"),
                    ]
                )
            self.assertIn("Cost/record", (root / "c.md").read_text(encoding="utf-8"))

    def test_cli_fit_infeasible_reports_and_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_rows(root / "dev.csv", ladder())
            base = [
                "fit",
                "--input",
                str(root / "dev.csv"),
                "--output",
                str(root / "p.json"),
                "--max-risk",
                "0",
                "--min-coverage",
                "0.8",
                "--risk-method",
                "empirical",
            ]
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as exited:
                main(base)
            self.assertEqual(exited.exception.code, 2)
            self.assertIn('"status": "no_feasible_policy"', err.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([*base, "--on-infeasible", "all-human"]), 0)
            saved = json.loads((root / "p.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["accept_disabled"])

    def test_cli_fit_requires_risk_or_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_rows(root / "dev.csv", ladder())
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as exited,
            ):
                main(["fit", "--input", str(root / "dev.csv"), "--output", str(root / "p.json")])
            self.assertEqual(exited.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
