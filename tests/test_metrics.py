import numpy as np
import unittest

from banking_intent.metrics import (
    confidence_threshold_table,
    select_confidence_threshold,
    top_two_margin,
)


class MetricsTests(unittest.TestCase):
    def test_top_two_margin(self):
        scores = np.array([[0.1, 0.8, 0.5], [0.9, -0.1, 0.2]])
        self.assertTrue(np.allclose(top_two_margin(scores), [0.3, 0.7]))

    def test_confidence_threshold_meets_target_when_possible(self):
        actual = np.array(["a", "b", "a", "b"])
        predicted = np.array(["a", "a", "a", "b"])
        confidence = np.array([0.9, 0.1, 0.8, 0.7])
        table = confidence_threshold_table(actual, predicted, confidence, target_accuracy=0.95)
        threshold = select_confidence_threshold(table)
        row = table.iloc[(table["threshold"] - threshold).abs().argmin()]
        self.assertGreaterEqual(row["accepted_accuracy"], 0.95)
        self.assertLess(row["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
