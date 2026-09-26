import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from calibroute.cli import main
from calibroute.comparison import compare_policies, render_comparison
from calibroute.models import PredictionRecord
from calibroute.policy import GatePolicy


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.policy = GatePolicy(
            0.9, 0.6, 0.1, 0.5, [0.5, 0.5], histogram_bins=2, max_js_divergence=1
        )
        self.rows = [
            PredictionRecord("same", 0.9, True, "a"),
            PredictionRecord("same", 0.9, False, "b"),
            PredictionRecord("c", 0.7, False, "a"),
            PredictionRecord("d", 0.2, True, "b"),
        ]

    def test_counts_risk_and_threshold_ties(self):
        report = compare_policies(self.rows, self.policy, fixed_threshold=0.9)
        all_accept = report["overall"]["accept_all"]
        self.assertEqual(all_accept["accepted_risk"], 0.5)
        fixed = report["overall"]["fixed_threshold"]
        self.assertEqual((fixed["accepted"], fixed["human_review"]), (2, 2))
        actual = report["overall"]["calibroute"]
        self.assertEqual(actual["accepted_errors"], 1)
        self.assertEqual(actual["accepted_risk"], 0.5)
        self.assertEqual((actual["human_review"], actual["abstain"]), (1, 1))
        for rule in report["overall"]:
            for field in ("count", "accepted", "accepted_errors", "human_review", "abstain"):
                self.assertEqual(
                    sum(d[rule][field] for d in report["domains"].values()),
                    report["overall"][rule][field],
                )

    def test_shift_zero_acceptance_is_not_zero_risk(self):
        policy = GatePolicy(
            0.9,
            0.6,
            0.1,
            0.5,
            [1.0, 0.0],
            histogram_bins=2,
            max_js_divergence=0.01,
            shift_min_batch_size=1,
        )
        report = compare_policies(self.rows, policy, fixed_threshold=1)
        self.assertTrue(report["shift"]["severe_shift"])
        self.assertIsNone(report["overall"]["calibroute"]["accepted_risk"])
        self.assertIsNone(report["overall"]["fixed_threshold"]["accepted_risk"])
        self.assertEqual(report["overall"]["calibroute"]["human_review"], 4)
        self.assertIn("n/a", render_comparison(report))

    def test_rejects_missing_labels_and_invalid_thresholds(self):
        for rows, threshold in (
            ([], 0.5),
            ([PredictionRecord("x", 0.9)], 0.5),
            (self.rows, float("nan")),
            (self.rows, 1.1),
        ):
            with self.assertRaises(ValueError):
                compare_policies(rows, self.policy, fixed_threshold=threshold)

    def test_permutation_does_not_change_report(self):
        self.assertEqual(
            compare_policies(self.rows, self.policy, fixed_threshold=0.8),
            compare_policies(reversed(self.rows), self.policy, fixed_threshold=0.8),
        )

    def test_cli_outputs_and_domain_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.json").write_text(json.dumps(self.policy.as_dict()), encoding="utf-8")
            (root / "input.csv").write_text(
                "id,confidence,correct,domain\na,.9,1,a\nb,.9,0,b\n", encoding="utf-8"
            )
            for suffix in ("json", "md"):
                output = root / f"report.{suffix}"
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(
                        [
                            "compare",
                            "--input",
                            str(root / "input.csv"),
                            "--policy",
                            str(root / "policy.json"),
                            "--fixed-threshold",
                            ".9",
                            "--domain",
                            "a",
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(code, 0)
                content = output.read_text(encoding="utf-8")
                if suffix == "json":
                    self.assertEqual(json.loads(content)["overall"]["accept_all"]["count"], 1)
                else:
                    self.assertIn("# Routing comparison", content)
