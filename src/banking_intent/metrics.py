from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score


@dataclass
class IntentMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    top_3_accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_multiclass(
    y_true,
    predictions,
    scores: np.ndarray,
    classes: np.ndarray | list[str],
) -> IntentMetrics:
    return IntentMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        macro_f1=float(f1_score(y_true, predictions, average="macro")),
        weighted_f1=float(f1_score(y_true, predictions, average="weighted")),
        top_3_accuracy=float(
            top_k_accuracy_score(y_true, scores, k=3, labels=np.asarray(classes))
        ),
    )


def top_two_margin(scores: np.ndarray) -> np.ndarray:
    """Return the gap between each example's largest and second-largest score."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must be a two-dimensional array with at least two classes")
    two_largest = np.partition(scores, kth=-2, axis=1)[:, -2:]
    return two_largest[:, 1] - two_largest[:, 0]


def confidence_threshold_table(
    y_true,
    predictions,
    confidence: np.ndarray,
    target_accuracy: float = 0.90,
) -> pd.DataFrame:
    """Trace the accuracy/coverage trade-off for a human-handoff threshold."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    confidence = np.asarray(confidence, dtype=float)
    candidates = np.unique(
        np.concatenate(
            ([0.0], np.quantile(confidence, np.linspace(0.0, 0.95, 40)))
        )
    )
    rows: list[dict] = []
    for threshold in candidates:
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        accepted_accuracy = (
            float((predictions[accepted] == y_true[accepted]).mean())
            if accepted_count
            else float("nan")
        )
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(accepted.mean()),
                "accepted_accuracy": accepted_accuracy,
                "accepted_examples": accepted_count,
                "target_accuracy_met": bool(accepted_accuracy >= target_accuracy),
            }
        )
    return pd.DataFrame(rows)


def select_confidence_threshold(table: pd.DataFrame) -> float:
    """Choose maximum coverage among thresholds meeting the accuracy target."""
    eligible = table[table["target_accuracy_met"]].sort_values(
        ["coverage", "threshold"], ascending=[False, True]
    )
    if not eligible.empty:
        return float(eligible.iloc[0]["threshold"])
    fallback = table.sort_values(
        ["accepted_accuracy", "coverage"], ascending=[False, False]
    ).iloc[0]
    return float(fallback["threshold"])


def per_class_report(y_true, predictions) -> pd.DataFrame:
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}:
            rows.append({"category": label, **values})
    return pd.DataFrame(rows).sort_values("f1-score")


def confusion_pairs(y_true, predictions) -> pd.DataFrame:
    errors = pd.DataFrame({"actual": y_true, "predicted": predictions})
    errors = errors[errors["actual"] != errors["predicted"]]
    return (
        errors.groupby(["actual", "predicted"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

