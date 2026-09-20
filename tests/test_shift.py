import math
import unittest

from calibroute.shift import confidence_histogram, js_divergence


class ShiftTests(unittest.TestCase):
    def test_histogram_is_normalized(self):
        histogram = confidence_histogram([0.0, 0.1, 0.9, 1.0], bins=4)
        self.assertAlmostEqual(sum(histogram), 1.0)

    def test_identical_distributions_have_zero_divergence(self):
        self.assertAlmostEqual(js_divergence([0.2, 0.8], [0.2, 0.8]), 0.0)

    def test_disjoint_distributions_have_max_divergence(self):
        self.assertAlmostEqual(js_divergence([1.0, 0.0], [0.0, 1.0]), math.log(2))

    def test_empty_histogram_input_fails(self):
        with self.assertRaises(ValueError):
            confidence_histogram([], bins=10)


if __name__ == "__main__":
    unittest.main()
