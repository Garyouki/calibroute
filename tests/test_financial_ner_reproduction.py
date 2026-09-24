"""Offline checks for the separately maintained NER data preparation entry point."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPRODUCTION = Path(__file__).resolve().parents[1] / "examples" / "financial_ner" / "reproduction"


class FinancialNerReproductionTests(unittest.TestCase):
    def test_recovered_source_hashes(self):
        import hashlib

        manifest = json.loads((REPRODUCTION / "source_manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["recovered_files"]:
            with self.subTest(path=entry["path"]):
                data = (REPRODUCTION / entry["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])

    def test_missing_offline_cache_fails_without_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_offline(directory)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Offline cache missing", result.stderr)
            self.assertFalse((Path(directory) / "verification.json").exists())

    def test_corrupt_cache_preserves_existing_normalized_data(self):
        manifest = json.loads((REPRODUCTION / "source_manifest.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "downloads").mkdir()
            (output / "downloads" / manifest["downloads"][0]["key"]).write_bytes(b"corrupt")
            (output / "raw").mkdir()
            sentinel = output / "raw" / "fin_train.jsonl"
            sentinel.write_bytes(b"existing data")
            result = self.run_offline(directory)
            self.assertEqual(result.returncode, 1)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertEqual(sentinel.read_bytes(), b"existing data")
            self.assertFalse((output / "verification.json").exists())

    @staticmethod
    def run_offline(directory):
        return subprocess.run(
            [
                sys.executable,
                str(REPRODUCTION / "prepare_data.py"),
                "--output",
                directory,
                "--offline",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
