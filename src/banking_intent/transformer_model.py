from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .config import ENCODER_NAME, RANDOM_STATE


def load_encoder(name: str = ENCODER_NAME) -> SentenceTransformer:
    return SentenceTransformer(name)


def encode_texts(
    encoder: SentenceTransformer,
    texts,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Encode text with optional local caching for reproducible reruns."""
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    embeddings = encoder.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings


def build_embedding_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=5.0,
        max_iter=2_000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

