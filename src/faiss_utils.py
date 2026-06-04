from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def _prepare_embeddings(embeddings) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    return np.ascontiguousarray(array)


def build_faiss_index(embeddings) -> faiss.IndexFlatIP:
    """
    Build a cosine similarity FAISS index.

    Assumes embeddings are already L2-normalized (from generate_embeddings.py).
    Applies normalization again as a safety check for idempotent consistency.

    Steps:
    - Normalize embeddings (L2) → idempotent if already normalized
    - Use inner product (IP) → becomes cosine similarity
    """
    prepared = _prepare_embeddings(embeddings)

    # Normalize vectors for cosine similarity (idempotent if already normalized)
    faiss.normalize_L2(prepared)

    # Inner product index (cosine similarity after normalization)
    index = faiss.IndexFlatIP(prepared.shape[1])

    if prepared.size:
        index.add(prepared)

    return index


def save_faiss_index(index: faiss.Index, index_path: str | Path) -> None:
    output_path = Path(index_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))


def load_faiss_index(index_path: str | Path):
    input_path = Path(index_path)

    if not input_path.exists():
        return None

    return faiss.read_index(str(input_path))