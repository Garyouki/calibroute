import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shift_study", ROOT / "examples/shift_study/run_study.py"
)
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)
PLAN = json.loads((ROOT / "examples/shift_study/protocol.json").read_text())


class ShiftStudyTests(unittest.TestCase):
    def test_random_streams_are_repeatable_and_separate(self):
        self.assertEqual(
            STUDY.stream(7, "reference").random(), STUDY.stream(7, "reference").random()
        )
        self.assertNotEqual(
            STUDY.stream(7, "reference").random(), STUDY.stream(7, "batch").random()
        )

    def test_scenario_labels_and_scores_are_paired(self):
        scenarios = {s["name"]: s for s in PLAN["scenarios"]}
        inputs = ([0.95, 0.95], [0.1, 0.9], [0.8, 0.8])
        stable = STUDY.make_rows(*inputs, scenarios["stable"])
        benign = STUDY.make_rows(*inputs, scenarios["benign_score_shift_30"])
        harmful = STUDY.make_rows(*inputs, scenarios["score_shift_30"])
        label_only = STUDY.make_rows(*inputs, scenarios["label_only"])
        self.assertEqual([r.correct for r in stable], [r.correct for r in benign])
        self.assertEqual([r.confidence for r in benign], [r.confidence for r in harmful])
        self.assertNotEqual([r.correct for r in benign], [r.correct for r in harmful])
        self.assertEqual([r.confidence for r in stable], [r.confidence for r in label_only])
        self.assertEqual([r.correct for r in label_only], [False, False])

    def test_wilson_boundary_and_symmetry(self):
        self.assertAlmostEqual(STUDY.wilson(0, 300)[0], 0)
        self.assertAlmostEqual(STUDY.wilson(300, 300)[1], 1)
        left, right = STUDY.wilson(150, 300)
        self.assertLess(left, 0.5)
        self.assertAlmostEqual(left + right, 1)

    def test_full_cell_reproducibility_counts_and_gate_blind_spot(self):
        profile = PLAN["profiles"][0]
        first = STUDY.study_cell(PLAN, profile, 150, 300, 2)
        self.assertEqual(first, STUDY.study_cell(PLAN, profile, 150, 300, 2))
        self.assertEqual(len(first), 14)
        for replication in range(2):
            scenarios = {r["scenario"]: r for r in first if r["replication"] == replication}
            stable, label_only = scenarios["stable"], scenarios["label_only"]
            for field in ("alarm", "shift_score", "threshold", "observed_histogram"):
                self.assertEqual(stable[field], label_only[field])
            for row in scenarios.values():
                for metrics in row["metrics"].values():
                    self.assertEqual(
                        metrics["accepted"] + metrics["review"] + metrics["abstain"], 300
                    )
                    self.assertLessEqual(metrics["accepted_errors"], metrics["accepted"])
                full, alone = row["metrics"]["full_router"], row["metrics"]["threshold_only"]
                if row["alarm"]:
                    self.assertEqual((full["accepted"], full["review"]), (0, 300))
                else:
                    self.assertEqual(full, alone)
        for cell in STUDY.aggregate(first):
            full, alone = cell["metrics"]["full_router"], cell["metrics"]["threshold_only"]
            self.assertEqual(full["count"], 600)
            self.assertEqual(
                cell["additional_reviews"],
                alone["accepted"] - full["accepted"] + alone["abstain"] - full["abstain"],
            )

    def test_zero_acceptance_has_undefined_risk(self):
        rows = STUDY.study_cell(PLAN, PLAN["profiles"][0], 150, 300, 1)
        row = rows[0]
        row["metrics"]["full_router"].update(accepted=0, accepted_errors=0, review=300, abstain=0)
        summary = STUDY.aggregate([row])[0]
        self.assertIsNone(summary["metrics"]["full_router"]["accepted_risk"])
