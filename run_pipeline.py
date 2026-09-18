from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from banking_intent.baseline import build_baseline  # noqa: E402
from banking_intent.config import (  # noqa: E402
    BASELINE_MODEL_PATH,
    CACHE_DIR,
    CATEGORIES_PATH,
    ENCODER_NAME,
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_ACCEPTED_ACCURACY,
    TEST_PATH,
    TRAIN_PATH,
    TRANSFORMER_MODEL_PATH,
    VALIDATION_SIZE,
)
from banking_intent.data import load_categories, load_split, validate_dataset  # noqa: E402
from banking_intent.metrics import (  # noqa: E402
    confidence_threshold_table,
    confusion_pairs,
    evaluate_multiclass,
    per_class_report,
    select_confidence_threshold,
    top_two_margin,
)
from banking_intent.transformer_model import (  # noqa: E402
    build_embedding_classifier,
    encode_texts,
    load_encoder,
)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


def metric_row(split: str, model: str, metrics) -> dict:
    return {"split": split, "model": model, **metrics.to_dict()}


def accepted_summary(table: pd.DataFrame, threshold: float) -> dict:
    row = table.iloc[(table["threshold"] - threshold).abs().argmin()]
    return {
        "threshold": float(row["threshold"]),
        "coverage": float(row["coverage"]),
        "accepted_accuracy": float(row["accepted_accuracy"]),
        "accepted_examples": int(row["accepted_examples"]),
    }


def evaluate_handoff(y_true, predictions, confidence, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    confidence = np.asarray(confidence, dtype=float)
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    return {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(
            (predictions[accepted] == y_true[accepted]).mean()
        ) if accepted_count else None,
        "accepted_examples": accepted_count,
        "reviewed_examples": int((~accepted).sum()),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    setup_style()

    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)
    categories = load_categories(CATEGORIES_PATH)
    quality = validate_dataset(train, test, categories)
    save_json(REPORTS_DIR / "data_validation.json", quality)

    all_indices = np.arange(len(train))
    development_indices, validation_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_SIZE,
        stratify=train["category"],
        random_state=RANDOM_STATE,
    )
    development = train.iloc[development_indices]
    validation = train.iloc[validation_indices]

    # Model 1: word + character TF-IDF with a linear support-vector classifier.
    baseline_validation_model = build_baseline()
    baseline_validation_model.fit(development["text"], development["category"])
    baseline_validation_scores = baseline_validation_model.decision_function(validation["text"])
    baseline_validation_predictions = baseline_validation_model.classes_[
        baseline_validation_scores.argmax(axis=1)
    ]
    baseline_validation_metrics = evaluate_multiclass(
        validation["category"],
        baseline_validation_predictions,
        baseline_validation_scores,
        baseline_validation_model.classes_,
    )
    baseline_validation_confidence = top_two_margin(baseline_validation_scores)
    baseline_thresholds = confidence_threshold_table(
        validation["category"],
        baseline_validation_predictions,
        baseline_validation_confidence,
        TARGET_ACCEPTED_ACCURACY,
    )
    baseline_handoff_threshold = select_confidence_threshold(baseline_thresholds)

    # Model 2: pretrained MiniLM sentence embeddings with a supervised classifier.
    encoder = load_encoder(ENCODER_NAME)
    train_embeddings = encode_texts(
        encoder, train["text"], CACHE_DIR / "banking77_train_minilm.npy"
    )
    test_embeddings = encode_texts(
        encoder, test["text"], CACHE_DIR / "banking77_test_minilm.npy"
    )
    transformer_validation_model = build_embedding_classifier()
    transformer_validation_model.fit(
        train_embeddings[development_indices], development["category"]
    )
    transformer_validation_scores = transformer_validation_model.predict_proba(
        train_embeddings[validation_indices]
    )
    transformer_validation_predictions = transformer_validation_model.classes_[
        transformer_validation_scores.argmax(axis=1)
    ]
    transformer_validation_metrics = evaluate_multiclass(
        validation["category"],
        transformer_validation_predictions,
        transformer_validation_scores,
        transformer_validation_model.classes_,
    )
    transformer_validation_confidence = transformer_validation_scores.max(axis=1)
    transformer_thresholds = confidence_threshold_table(
        validation["category"],
        transformer_validation_predictions,
        transformer_validation_confidence,
        TARGET_ACCEPTED_ACCURACY,
    )
    transformer_handoff_threshold = select_confidence_threshold(transformer_thresholds)

    validation_rows = [
        metric_row("validation", "tfidf_linear_svc", baseline_validation_metrics),
        metric_row("validation", "minilm_logistic_regression", transformer_validation_metrics),
    ]
    selected_model = max(validation_rows, key=lambda row: row["macro_f1"])["model"]

    # Freeze choices, retrain both models on the complete official training split,
    # and evaluate only once on the official test split.
    baseline_model = build_baseline()
    baseline_model.fit(train["text"], train["category"])
    baseline_test_scores = baseline_model.decision_function(test["text"])
    baseline_test_predictions = baseline_model.classes_[baseline_test_scores.argmax(axis=1)]
    baseline_test_metrics = evaluate_multiclass(
        test["category"],
        baseline_test_predictions,
        baseline_test_scores,
        baseline_model.classes_,
    )
    baseline_test_confidence = top_two_margin(baseline_test_scores)

    transformer_model = build_embedding_classifier()
    transformer_model.fit(train_embeddings, train["category"])
    transformer_test_scores = transformer_model.predict_proba(test_embeddings)
    transformer_test_predictions = transformer_model.classes_[
        transformer_test_scores.argmax(axis=1)
    ]
    transformer_test_metrics = evaluate_multiclass(
        test["category"],
        transformer_test_predictions,
        transformer_test_scores,
        transformer_model.classes_,
    )
    transformer_test_confidence = transformer_test_scores.max(axis=1)

    # The official files contain a very small number of exact text overlaps.
    # Preserve the benchmark result and also report a sensitivity check with
    # those examples removed from the test calculation.
    clean_test_mask = ~test["text"].isin(set(train["text"]))
    baseline_clean_test_metrics = evaluate_multiclass(
        test.loc[clean_test_mask, "category"],
        baseline_test_predictions[clean_test_mask],
        baseline_test_scores[clean_test_mask],
        baseline_model.classes_,
    )
    transformer_clean_test_metrics = evaluate_multiclass(
        test.loc[clean_test_mask, "category"],
        transformer_test_predictions[clean_test_mask],
        transformer_test_scores[clean_test_mask],
        transformer_model.classes_,
    )

    comparison_rows = validation_rows + [
        metric_row("test", "tfidf_linear_svc", baseline_test_metrics),
        metric_row("test", "minilm_logistic_regression", transformer_test_metrics),
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    baseline_thresholds.to_csv(
        REPORTS_DIR / "baseline_confidence_coverage.csv", index=False
    )
    transformer_thresholds.to_csv(
        REPORTS_DIR / "transformer_confidence_coverage.csv", index=False
    )

    joblib.dump(
        {
            "model": baseline_model,
            "handoff_threshold": baseline_handoff_threshold,
            "confidence_type": "top-two decision-score margin",
            "validation_metrics": baseline_validation_metrics.to_dict(),
            "test_metrics": baseline_test_metrics.to_dict(),
            "dataset": "BANKING77",
        },
        BASELINE_MODEL_PATH,
        compress=3,
    )
    joblib.dump(
        {
            "classifier": transformer_model,
            "encoder_name": ENCODER_NAME,
            "handoff_threshold": transformer_handoff_threshold,
            "confidence_type": "maximum class probability",
            "validation_metrics": transformer_validation_metrics.to_dict(),
            "test_metrics": transformer_test_metrics.to_dict(),
            "dataset": "BANKING77",
        },
        TRANSFORMER_MODEL_PATH,
        compress=3,
    )

    if selected_model == "minilm_logistic_regression":
        champion_predictions = transformer_test_predictions
        champion_scores = transformer_test_scores
        champion_confidence = transformer_test_confidence
        champion_classes = transformer_model.classes_
        champion_threshold = transformer_handoff_threshold
        champion_validation_table = transformer_thresholds
    else:
        champion_predictions = baseline_test_predictions
        champion_scores = baseline_test_scores
        champion_confidence = baseline_test_confidence
        champion_classes = baseline_model.classes_
        champion_threshold = baseline_handoff_threshold
        champion_validation_table = baseline_thresholds

    per_class = per_class_report(test["category"], champion_predictions)
    per_class.to_csv(REPORTS_DIR / "champion_per_class_metrics.csv", index=False)
    pairs = confusion_pairs(test["category"], champion_predictions)
    pairs.to_csv(REPORTS_DIR / "champion_confusion_pairs.csv", index=False)

    top_indices = np.argsort(champion_scores, axis=1)[:, ::-1][:, :3]
    predictions = test.copy()
    predictions["predicted_intent"] = champion_predictions
    predictions["confidence"] = champion_confidence
    predictions["needs_human_review"] = champion_confidence < champion_threshold
    predictions["correct"] = predictions["category"] == predictions["predicted_intent"]
    for rank in range(3):
        predictions[f"top_{rank + 1}_intent"] = champion_classes[top_indices[:, rank]]
        predictions[f"top_{rank + 1}_score"] = champion_scores[
            np.arange(len(test)), top_indices[:, rank]
        ]
    predictions.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    predictions.loc[~predictions["correct"]].sort_values("confidence").to_csv(
        REPORTS_DIR / "misclassified_examples.csv", index=False
    )

    metrics_payload = {
        "dataset": "BANKING77",
        "dataset_rows": {"train": int(len(train)), "test": int(len(test))},
        "categories": len(categories),
        "selection_metric": "validation macro F1",
        "selected_model": selected_model,
        "target_accepted_accuracy": TARGET_ACCEPTED_ACCURACY,
        "validation_metrics": {
            "tfidf_linear_svc": baseline_validation_metrics.to_dict(),
            "minilm_logistic_regression": transformer_validation_metrics.to_dict(),
        },
        "official_test_metrics": {
            "tfidf_linear_svc": baseline_test_metrics.to_dict(),
            "minilm_logistic_regression": transformer_test_metrics.to_dict(),
        },
        "overlap_removed_test_metrics": {
            "removed_exact_text_overlaps": int((~clean_test_mask).sum()),
            "remaining_test_rows": int(clean_test_mask.sum()),
            "tfidf_linear_svc": baseline_clean_test_metrics.to_dict(),
            "minilm_logistic_regression": transformer_clean_test_metrics.to_dict(),
        },
        "handoff_policy": {
            "tfidf_linear_svc": accepted_summary(
                baseline_thresholds, baseline_handoff_threshold
            ),
            "minilm_logistic_regression": accepted_summary(
                transformer_thresholds, transformer_handoff_threshold
            ),
        },
        "official_test_handoff": {
            "tfidf_linear_svc": evaluate_handoff(
                test["category"],
                baseline_test_predictions,
                baseline_test_confidence,
                baseline_handoff_threshold,
            ),
            "minilm_logistic_regression": evaluate_handoff(
                test["category"],
                transformer_test_predictions,
                transformer_test_confidence,
                transformer_handoff_threshold,
            ),
        },
    }
    save_json(REPORTS_DIR / "metrics.json", metrics_payload)

    # Figure 1: label distribution.
    counts = train["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.histplot(counts.values, bins=12, color="#2a9d8f", ax=ax)
    ax.set_title("BANKING77 training examples per intent")
    ax.set_xlabel("Training examples in an intent")
    ax.set_ylabel("Number of intents")
    fig.savefig(FIGURES_DIR / "intent_distribution.png")
    plt.close(fig)

    # Figure 2: validation and official test comparison.
    melted = comparison.melt(
        id_vars=["split", "model"],
        value_vars=["accuracy", "macro_f1", "top_3_accuracy"],
        var_name="metric",
        value_name="score",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, split in zip(axes, ["validation", "test"]):
        sns.barplot(data=melted[melted["split"] == split], x="metric", y="score", hue="model", ax=ax)
        ax.set_ylim(0.70, 1.0)
        ax.set_title(f"{split.title()} performance")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
    fig.savefig(FIGURES_DIR / "model_comparison.png")
    plt.close(fig)

    # Figure 3: validation accuracy/coverage trade-off for both models.
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for label, table in [
        ("TF-IDF + LinearSVC", baseline_thresholds),
        ("MiniLM + Logistic Regression", transformer_thresholds),
    ]:
        ax.plot(table["coverage"], table["accepted_accuracy"], marker=".", label=label)
    ax.axhline(
        TARGET_ACCEPTED_ACCURACY,
        color="#e76f51",
        linestyle="--",
        label=f"{TARGET_ACCEPTED_ACCURACY:.0%} target",
    )
    ax.set_title("Validation trade-off for automated routing")
    ax.set_xlabel("Coverage retained for automation")
    ax.set_ylabel("Accuracy on accepted messages")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.75, 1.01)
    ax.legend()
    fig.savefig(FIGURES_DIR / "confidence_coverage.png")
    plt.close(fig)

    # Figure 4: most frequent confusion pairs.
    top_pairs = pairs.head(15).copy()
    top_pairs["pair"] = top_pairs["actual"] + "  →  " + top_pairs["predicted"]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    sns.barplot(data=top_pairs, y="pair", x="count", color="#457b9d", ax=ax)
    ax.set_title(f"Most frequent errors — {selected_model}")
    ax.set_xlabel("Misclassified test messages")
    ax.set_ylabel("Actual → predicted")
    fig.savefig(FIGURES_DIR / "confusion_pairs.png")
    plt.close(fig)

    # Figure 5: lowest per-intent F1 scores.
    lowest = per_class.head(15).sort_values("f1-score")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    sns.barplot(data=lowest, y="category", x="f1-score", color="#f4a261", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title(f"Lowest held-out per-intent F1 — {selected_model}")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    fig.savefig(FIGURES_DIR / "lowest_per_intent_f1.png")
    plt.close(fig)

    # Figure 6: confidence separates correct and incorrect predictions.
    confidence_frame = pd.DataFrame(
        {
            "confidence": champion_confidence,
            "outcome": np.where(
                champion_predictions == test["category"].to_numpy(), "Correct", "Incorrect"
            ),
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.histplot(
        data=confidence_frame[confidence_frame["outcome"] == "Correct"],
        x="confidence",
        bins=30,
        stat="density",
        element="step",
        color="#e76f51",
        alpha=0.25,
        label="Correct",
        ax=ax,
    )
    sns.histplot(
        data=confidence_frame[confidence_frame["outcome"] == "Incorrect"],
        x="confidence",
        bins=30,
        stat="density",
        element="step",
        color="#457b9d",
        alpha=0.25,
        label="Incorrect",
        ax=ax,
    )
    ax.axvline(champion_threshold, color="#e76f51", linestyle="--", label="Handoff threshold")
    ax.set_title(f"Confidence by prediction outcome — {selected_model}")
    ax.legend()
    fig.savefig(FIGURES_DIR / "confidence_by_outcome.png")
    plt.close(fig)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
