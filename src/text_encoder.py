from __future__ import annotations

import numpy as np
import torch

from utils.helpers import load_clip_components

# Device selection (CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device for text encoding:", device)


def _normalize_text_inputs(text_inputs) -> list[str]:
    if isinstance(text_inputs, str):
        return [text_inputs.strip()]

    return [str(item).strip() for item in text_inputs]


def encode_text_batch(text_inputs) -> np.ndarray:
    """Encode one or more text prompts into normalized OpenCLIP embeddings.

    Ensures token tensors and model are on the same device, uses no_grad,
    normalizes outputs, and returns CPU numpy arrays.
    """
    texts = _normalize_text_inputs(text_inputs)
    if not texts or not any(texts):
        raise ValueError("text_inputs cannot be empty")

    model, _, tokenizer, _ = load_clip_components()
    # Ensure model is on the chosen device
    model = model.to(device)

    tokens = tokenizer(texts)
    tokens = tokens.to(device)

    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().numpy().astype(np.float32)


def encode_text(text_input) -> np.ndarray:
    """Encode a single text query and return a 2D array with shape (1, embedding_dim)."""
    return encode_text_batch([text_input])