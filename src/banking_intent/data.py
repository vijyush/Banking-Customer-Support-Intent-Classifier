from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["text", "category"]


def load_split(path: Path | str) -> pd.DataFrame:
    """Load one BANKING77 split and normalize only superficial whitespace."""
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["text"] = frame["text"].astype("string").str.strip()
    frame["category"] = frame["category"].astype("string").str.strip()
    return frame


def load_categories(path: Path | str) -> list[str]:
    categories = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
        raise ValueError("categories.json must contain a JSON list of strings")
    return categories


def validate_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categories: list[str],
) -> dict:
    """Validate schema and label coverage and return an auditable quality report."""
    for split_name, frame in {"train": train, "test": test}.items():
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{split_name} is missing columns: {sorted(missing)}")
        if frame[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(f"{split_name} contains missing text or labels")
        if frame["text"].str.len().eq(0).any():
            raise ValueError(f"{split_name} contains empty utterances")

    category_set = set(categories)
    observed_train = set(train["category"].unique())
    observed_test = set(test["category"].unique())
    if len(categories) != 77 or len(category_set) != 77:
        raise ValueError("Expected exactly 77 unique BANKING77 categories")
    if observed_train != category_set:
        raise ValueError("Training labels do not match categories.json")
    if not observed_test.issubset(observed_train):
        raise ValueError("Test data contains labels absent from training data")

    combined = pd.concat(
        [train.assign(split="train"), test.assign(split="test")], ignore_index=True
    )
    conflicting = (
        combined.groupby("text")["category"].nunique().gt(1).sum()
    )
    train_test_overlap = len(set(train["text"]) & set(test["text"]))
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "categories": int(len(categories)),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "test_duplicate_rows": int(test.duplicated().sum()),
        "train_test_text_overlap": int(train_test_overlap),
        "texts_with_conflicting_labels": int(conflicting),
        "minimum_train_examples_per_intent": int(train["category"].value_counts().min()),
        "maximum_train_examples_per_intent": int(train["category"].value_counts().max()),
    }

