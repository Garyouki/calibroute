import unittest

from trustgate.metrics import (
    audit_records,
    expected_calibration_error,
    risk_coverage,
    roc_auc,
)
from trustgate.models import PredictionRecord


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            PredictionRecord("a", 0.95, True, "filings"),
            PredictionRecord("b", 0.85, True, "filings"),
            PredictionRecord("c", 0.60, False, "news"),
            PredictionRecord("d", 0.20, False, "news"),
        ]

    def test_auc_perfect_ranking(self):
        self.assertEqual(roc_auc(self.rows), 1.0)

    def test_auc_returns_none_for_one_class(self):
        self.assertIsNone(roc_auc(self.rows[:2]))

    def test_ece_is_bounded(self):
        value = expected_calibration_error(self.rows, bins=5)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_risk_coverage_has_all_prefixes(self):
        points, aurc = risk_coverage(self.rows)
        self.assertEqual(len(points), 4)
        self.assertEqual(points[0]["risk"], 0.0)
        self.assertEqual(points[-1]["risk"], 0.5)
        self.assertGreaterEqual(aurc, 0.0)

    def test_audit_groups_domains(self):
        report = audit_records(self.rows, coverages=(0.5, 1.0))
        self.assertEqual(report["overall"]["accuracy"], 0.5)
        self.assertEqual(set(report["domains"]), {"filings", "news"})

    def test_unlabeled_audit_fails(self):
        with self.assertRaises(ValueError):
            audit_records([PredictionRecord("x", 0.5)])


if __name__ == "__main__":
    unittest.main()

