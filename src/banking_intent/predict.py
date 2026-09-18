from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .config import BASELINE_MODEL_PATH, TRANSFORMER_MODEL_PATH
from .metrics import top_two_margin
from .transformer_model import load_encoder


@lru_cache(maxsize=2)
def _load_bundle(path: str) -> dict:
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_sentence_encoder(name: str):
    return load_encoder(name)


def predict_intent(
    text: str,
    model_name: str = "transformer",
    top_k: int = 3,
    model_path: Path | None = None,
) -> dict:
    """Predict an intent and return ranked suggestions plus handoff guidance."""
    if not text or not text.strip():
        raise ValueError("Enter a non-empty customer message")

    if model_name == "baseline":
        path = Path(model_path or BASELINE_MODEL_PATH)
        bundle = _load_bundle(str(path))
        estimator = bundle["model"]
        scores = estimator.decision_function([text])
        confidence = float(top_two_margin(scores)[0])
        classes = np.asarray(estimator.classes_)
    elif model_name == "transformer":
        path = Path(model_path or TRANSFORMER_MODEL_PATH)
        bundle = _load_bundle(str(path))
        classifier = bundle["classifier"]
        encoder = _load_sentence_encoder(bundle["encoder_name"])
        embedding = encoder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        scores = classifier.predict_proba(embedding)
        confidence = float(scores.max(axis=1)[0])
        classes = np.asarray(classifier.classes_)
    else:
        raise ValueError("model_name must be 'baseline' or 'transformer'")

    ranking = np.argsort(scores[0])[::-1][:top_k]
    suggestions = [
        {"intent": str(classes[index]), "score": float(scores[0, index])}
        for index in ranking
    ]
    threshold = float(bundle["handoff_threshold"])
    return {
        "predicted_intent": suggestions[0]["intent"],
        "confidence": confidence,
        "needs_human_review": bool(confidence < threshold),
        "handoff_threshold": threshold,
        "top_intents": suggestions,
        "model": model_name,
    }

