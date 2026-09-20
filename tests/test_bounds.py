import unittest

from calibroute.bounds import clopper_pearson_upper


class BoundTests(unittest.TestCase):
    def test_zero_error_closed_form(self):
        expected = 1 - 0.05 ** (1 / 20)
        self.assertAlmostEqual(clopper_pearson_upper(0, 20), expected, places=10)

    def test_all_errors_is_one(self):
        self.assertEqual(clopper_pearson_upper(4, 4), 1.0)

    def test_more_evidence_tightens_zero_error_bound(self):
        self.assertLess(clopper_pearson_upper(0, 100), clopper_pearson_upper(0, 20))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            clopper_pearson_upper(1, 0)


if __name__ == "__main__":
    unittest.main()
