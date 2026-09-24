import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calibroute.adapters.shiftguard import read_shiftguard_records, write_shiftguard_csv
from calibroute.cli import main
from calibroute.io import read_records
from calibroute.policy import route_batch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "shiftguard"
SPEC = importlib.util.spec_from_file_location(
    "shiftguard_case_study", EXAMPLE / "run_case_study.py"
)
CASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CASE)


def trace(scenario="case-1", task="task-a", success=True, unsafe=False):
    return {
        "model": "example:model",
        "seed": 13,
        "scenario_id": scenario,
        "domain": "scheduler",
        "task_name": task,
        "telemetry_profile": "canonical",
        "scorer_version": "fixture-v1",
        "proposal": {"confidence": 0.95, "valid_json": True},
        "controllers": {
            "always_execute": {
                "task_success": success,
                "unsafe": unsafe,
                "safe_task_success": success and not unsafe,
            }
        },
    }


class ShiftGuardAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write(self, rows):
        path = self.directory / "traces.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path

    def test_labels_distinguish_success_safety_and_validity(self):
        rows = [trace("ok"), trace("no-op", success=False), trace("unsafe", unsafe=True)]
        invalid = trace("invalid-json")
        invalid["proposal"]["valid_json"] = False
        rows.append(invalid)
        records = read_shiftguard_records(self.write(rows))
        self.assertEqual([r.correct for r in records], [True, False, False, False])
        self.assertEqual([r.confidence for r in records], [0.95] * 4)

    def test_rejects_inconsistent_or_missing_outcome(self):
        for changes in ({"safe_task_success": False}, {"task_success": "true"}):
            row = trace()
            row["controllers"]["always_execute"].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "line 1:"):
                read_shiftguard_records(self.write([row]))
        row = trace()
        del row["controllers"]["always_execute"]
        with self.assertRaisesRegex(ValueError, "always_execute must be an object"):
            read_shiftguard_records(self.write([row]))

    def test_rejects_invalid_confidence_without_inventing_a_score(self):
        for value in (None, True, "0.95", -0.1, 1.1, float("nan"), float("inf"), 10**400):
            row = trace()
            row["proposal"]["confidence"] = value
            with (
                self.subTest(value=str(value)[:20]),
                self.assertRaisesRegex(ValueError, "confidence"),
            ):
                read_shiftguard_records(self.write([row]))

    def test_identity_separates_models_seeds_and_telemetry_but_rejects_duplicates(self):
        rows = [trace() for _ in range(4)]
        rows[1]["model"] = "other:model"
        rows[2]["seed"] = 42
        rows[3]["telemetry_profile"] = "degraded"
        records = read_shiftguard_records(self.write(rows))
        self.assertEqual(len({r.record_id for r in records}), 4)
        self.assertEqual(len({r.metadata["template_group"] for r in records}), 1)
        with self.assertRaisesRegex(ValueError, "duplicate proposal"):
            read_shiftguard_records(self.write([trace(), trace()]))

    def test_filters_and_schema_errors(self):
        row = trace()
        self.assertEqual(len(read_shiftguard_records(self.write([row]), seed=13)), 1)
        with self.assertRaisesRegex(ValueError, "no compatible"):
            read_shiftguard_records(self.write([row]), model="absent")
        for field, value in (("seed", True), ("seed", "13"), ("task_name", "")):
            bad = copy.deepcopy(row)
            bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                read_shiftguard_records(self.write([bad]))
        with self.assertRaisesRegex(ValueError, "record must be an object"):
            read_shiftguard_records(self.write([[]]))

    def test_oracle_fields_are_not_confidence_features(self):
        row = trace()
        row.update(hazardous=True, fault="hidden", authoritative_version=99, raw_content="private")
        row["controllers"]["some_other_controller"] = {"safe_task_success": False}
        record = read_shiftguard_records(self.write([row]))[0]
        self.assertTrue(record.correct)
        self.assertEqual(record.confidence, 0.95)
        for field in ("hazardous", "fault", "authoritative_version", "raw_content"):
            self.assertNotIn(field, record.metadata)

    def test_csv_and_cli_preserve_grouping_and_label_provenance(self):
        source = self.write([trace()])
        output = self.directory / "predictions.csv"
        self.assertEqual(
            main(["convert-shiftguard", "--input", str(source), "--output", str(output)]), 0
        )
        record = read_records(output)[0]
        original = read_shiftguard_records(source)[0]
        self.assertEqual(record.record_id, original.record_id)
        self.assertEqual(record.correct, original.correct)
        for field in (
            "template_group",
            "scenario_id",
            "task_name",
            "label_definition",
            "scorer_version",
        ):
            self.assertEqual(record.metadata[field], original.metadata[field])
        write_shiftguard_csv(output, [original])
        self.assertEqual(len(read_records(output)), 1)

    def test_split_groups_all_variants_and_is_order_independent(self):
        rows = []
        for task in ("task-a", "task-b"):
            for seed in (13, 42):
                for telemetry in ("canonical", "degraded"):
                    row = trace(task, task)
                    row.update(seed=seed, telemetry_profile=telemetry)
                    rows.append(row)
        records = read_shiftguard_records(self.write(rows))
        fitting, evaluation, split = CASE.split_templates(records)
        self.assertEqual((len(fitting), len(evaluation)), (4, 4))
        self.assertTrue(
            {r.metadata["template_group"] for r in fitting}.isdisjoint(
                {r.metadata["template_group"] for r in evaluation}
            )
        )
        self.assertEqual(CASE.split_templates(list(reversed(records)))[2], split)
        with self.assertRaisesRegex(ValueError, "at least two"):
            CASE.split_templates(fitting)

    def test_bundled_fixture_runs_offline_and_reports_infeasibility(self):
        report = CASE.run_case_study(EXAMPLE / "traces.jsonl", self.directory / "output")
        self.assertEqual(report["audit"]["overall"]["count"], 384)
        self.assertAlmostEqual(report["audit"]["overall"]["accuracy"], 224 / 384)
        self.assertEqual(report["split"]["fitting_count"], 288)
        self.assertEqual(report["split"]["evaluation_count"], 96)
        self.assertEqual(report["policy_attempt"]["status"], "no_feasible_policy")
        self.assertIsNone(report["policy_attempt"]["policy"])
        self.assertEqual(report["routing"]["status"], "not_run")
        self.assertEqual(
            len((self.directory / "output" / "decisions.csv").read_text().splitlines()), 1
        )

    def test_fitted_policy_never_sees_evaluation_labels(self):
        source = self.write([trace("a", "task-a"), trace("b", "task-b")])
        with patch.object(CASE, "route_batch", wraps=route_batch) as router:
            report = CASE.run_case_study(source, self.directory / "output")
        self.assertEqual(report["policy_attempt"]["status"], "fitted")
        supplied = router.call_args.args[0]
        self.assertTrue(all(r.correct is None and not r.metadata for r in supplied))
        self.assertEqual(report["routing"]["accepted_empirical_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
