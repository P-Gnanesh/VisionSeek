from __future__ import annotations

import numpy as np

from src.image_encoder import encode_image
from src.text_encoder import encode_text


def _validate_text_weight(text_weight: float) -> float:
    weight = float(text_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("text_weight must be between 0 and 1")
    return weight


def encode_multimodal(image_source, text_input: str, text_weight: float = 0.5) -> np.ndarray:
    """Blend image and text embeddings into a single 512-dim vector.

    The returned vector is the weighted average of the normalized image and text
    embeddings, with ``text_weight`` controlling the contribution from text.
    """
    weight = _validate_text_weight(text_weight)

    text_embedding = np.asarray(encode_text(text_input), dtype=np.float32).reshape(-1)
    image_embedding = np.asarray(encode_image(image_source), dtype=np.float32).reshape(-1)

    if text_embedding.shape != image_embedding.shape:
        raise ValueError("image and text embeddings must have the same shape")

    combined = (weight * text_embedding) + ((1.0 - weight) * image_embedding)
    norm = float(np.linalg.norm(combined))
    if norm > 0.0:
        combined = combined / norm

    return combined.astype(np.float32)