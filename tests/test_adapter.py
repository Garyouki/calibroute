import json
import tempfile
import unittest
from pathlib import Path

from calibroute.adapters.financial_ner import read_financial_ner_records


class FinancialNerAdapterTests(unittest.TestCase):
    def _file(self, rows):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "records.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return path

    def test_encoder_conversion_and_seed_filter(self):
        path = self._file(
            [
                {
                    "seed": 13,
                    "domain": "fin_valid",
                    "sent_id": "a",
                    "sent_error": 0,
                    "sent_conf_msp": 0.8,
                },
                {
                    "seed": 42,
                    "domain": "fin_test",
                    "sent_id": "b",
                    "sent_error": 1,
                    "sent_conf_msp": 0.4,
                },
            ]
        )
        records = read_financial_ner_records(path, model="encoder", seed=42)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].correct)

    def test_generative_signal(self):
        path = self._file(
            [
                {
                    "seed": 42,
                    "domain": "tweetner7_test",
                    "sent_id": "a",
                    "sent_error": 0,
                    "conf_seq": 0.9,
                    "conf_min_sc": 0.6,
                }
            ]
        )
        record = read_financial_ner_records(path, model="generative", signal="conf_seq")[0]
        self.assertEqual(record.confidence, 0.9)
        self.assertTrue(record.correct)

    def test_rejects_wrong_signal(self):
        path = self._file([])
        with self.assertRaises(ValueError):
            read_financial_ner_records(path, model="encoder", signal="conf_seq")


if __name__ == "__main__":
    unittest.main()
