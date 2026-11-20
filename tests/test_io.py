import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from calibroute.cli import main
from calibroute.io import read_policy, read_records, write_decisions
from calibroute.models import Action, Decision


class IOTests(unittest.TestCase):
    def test_csv_round_trip_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            source.write_text(
                "id,confidence,correct,domain\na,0.9,true,filings\nb,0.4,0,news\n",
                encoding="utf-8",
            )
            records = read_records(source)
            self.assertEqual(len(records), 2)
            self.assertTrue(records[0].correct)
            self.assertFalse(records[1].correct)

    def test_non_numeric_confidence_and_non_object_rows_are_cli_input_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            report = Path(directory) / "report.json"
            for invalid in ("null", '{"confidence":null}', '{"confidence":true,"correct":true}'):
                with self.subTest(invalid=invalid):
                    source.write_text("\n" + invalid + "\n", encoding="utf-8")
                    with (
                        contextlib.redirect_stderr(io.StringIO()) as stderr,
                        self.assertRaises(SystemExit) as error,
                    ):
                        main(["audit", "--input", str(source), "--output", str(report)])
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("line 2:", stderr.getvalue())
                    self.assertFalse(report.exists())

    def test_holdout_reader_preserves_zero_id_and_accepts_record_id_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text(
                '{"id":0,"confidence":0.9}\n{"record_id":"other","confidence":"0.8"}\n',
                encoding="utf-8",
            )
            records = read_records(source, require_id=True)
            self.assertEqual([row.record_id for row in records], ["0", "other"])

    def test_invalid_saved_policy_values_are_input_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            for content, message in (
                ("null", "policy must be a JSON object"),
                ('{"accept_threshold":null}', "invalid policy value"),
            ):
                with self.subTest(content=content):
                    policy.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        read_policy(policy)

    def test_write_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decisions.csv"
            write_decisions(
                output,
                [Decision("a", 0.9, Action.ACCEPT, "test", 0.01)],
            )
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["action"], "accept")


if __name__ == "__main__":
    unittest.main()
