from pathlib import Path
import unittest

from banking_intent.data import load_categories, load_split, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_official_dataset_passes_validation(self):
        train = load_split(ROOT / "data" / "raw" / "train.csv")
        test = load_split(ROOT / "data" / "raw" / "test.csv")
        categories = load_categories(ROOT / "data" / "raw" / "categories.json")
        report = validate_dataset(train, test, categories)
        self.assertEqual(report["train_rows"], 10_003)
        self.assertEqual(report["test_rows"], 3_080)
        self.assertEqual(report["categories"], 77)


if __name__ == "__main__":
    unittest.main()
