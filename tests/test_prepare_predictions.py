import contextlib
import csv
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples/your_data/prepare_predictions.py"
SPEC = importlib.util.spec_from_file_location("prepare_predictions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreparePredictionsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.csv"
        self.output = Path(self.directory.name) / "canonical.csv"

    def convert(self, text, **kwargs):
        self.source.write_text(text, encoding="utf-8")
        MODULE.convert_csv(
            self.source, self.output, id_column="item", confidence_column="score", **kwargs
        )
        with self.output.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_boolean_mapping_and_bom(self):
        rows = self.convert(
            "\ufeffitem,score,ok,task\na,.95,true,invoice\nb,.4,incorrect,invoice\n",
            correct_column="ok",
            domain_column="task",
        )
        self.assertEqual([r["correct"] for r in rows], ["1", "0"])
        self.assertEqual(rows[0]["domain"], "invoice")
        self.assertEqual(rows[0]["confidence"], "0.95")

    def test_exact_match_is_literal(self):
        rows = self.convert(
            'item,score,pred,label\na,.9,USD,USD\nb,.9,usd,USD\nc,.9,"USD ",USD\n',
            prediction_column="pred",
            label_column="label",
        )
        self.assertEqual([r["correct"] for r in rows], ["1", "0", "0"])

    def test_unlabeled_production(self):
        rows = self.convert("item,score\na,.8\n", unlabeled=True)
        self.assertNotIn("correct", rows[0])
        self.assertEqual(rows[0]["domain"], "default")

    def test_rejects_invalid_rows_without_overwriting_output(self):
        cases = [
            "item,score,ok\na,.8,1\nb,NaN,1\n",
            "item,score,ok\na,inf,1\n",
            "item,score,ok\na,1.1,1\n",
            "item,score,ok\na,-.1,1\n",
            "item,score,ok\na,95%,1\n",
            "item,score,ok\na,.8,\n",
            "item,score,ok\na,.8,maybe\n",
            "item,score,ok\na,.8,1\na,.7,0\n",
            "item,score,ok\n,.8,1\n",
            "item,score,ok\na,.8\n",
            "item,score,ok\na,.8,1,extra\n",
            "item,score,score,ok\na,.8,.9,1\n",
            "item,score,other\na,.8,1\n",
            "item,score,ok\n",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.output.write_text("existing output", encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.convert(text, correct_column="ok")
                self.assertEqual(self.output.read_text(encoding="utf-8"), "existing output")

    def test_blank_prediction_is_not_a_correct_match(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.convert(
                "item,score,pred,label\na,.8,,\n", prediction_column="pred", label_column="label"
            )
        self.assertFalse(self.output.exists())

    def test_rejects_ambiguous_modes_and_column_reuse(self):
        for kwargs in (
            {},
            {"prediction_column": "pred"},
            {"label_column": "label"},
            {"correct_column": "ok", "unlabeled": True},
            {"prediction_column": "ok", "label_column": "ok"},
            {"correct_column": "score"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.convert("item,score,ok\na,.8,1\n", **kwargs)

    def test_input_cannot_be_overwritten(self):
        content = "item,score,ok\na,.8,1\n"
        self.source.write_text(content, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different files"):
            MODULE.convert_csv(
                self.source,
                self.source,
                id_column="item",
                confidence_column="score",
                correct_column="ok",
            )
        self.assertEqual(self.source.read_text(encoding="utf-8"), content)

    def test_cli_invalid_input_returns_two(self):
        self.source.write_text("item,score,ok\na,.8,\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = MODULE.main(
                [
                    "--input",
                    str(self.source),
                    "--output",
                    str(self.output),
                    "--id-column",
                    "item",
                    "--confidence-column",
                    "score",
                    "--correct-column",
                    "ok",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("correctness", stderr.getvalue())
