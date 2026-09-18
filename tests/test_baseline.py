import unittest

from banking_intent.baseline import build_baseline


class BaselineTests(unittest.TestCase):
    def test_baseline_fits_and_predicts(self):
        texts = [
            "my card has not arrived",
            "where is my replacement card",
            "cash machine kept my card",
            "the atm did not return my card",
            "my transfer is pending",
            "why is the bank transfer pending",
        ]
        labels = [
            "card_arrival",
            "card_arrival",
            "cash_withdrawal",
            "cash_withdrawal",
            "pending_transfer",
            "pending_transfer",
        ]
        model = build_baseline()
        model.fit(texts, labels)
        prediction = model.predict(["can I track the delivery of my card?"])[0]
        self.assertIn(prediction, set(labels))
        self.assertEqual(model.decision_function(["hello"]).shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
