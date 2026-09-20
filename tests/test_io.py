import csv
import tempfile
import unittest
from pathlib import Path

from calibroute.io import read_records, write_decisions
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
