from __future__ import annotations

from pathlib import Path

import numpy as np
import faiss

from src.faiss_utils import build_faiss_index, load_faiss_index, save_faiss_index
from src.image_encoder import encode_image


def _load_existing_embeddings(existing_embeddings_path: str | Path) -> np.ndarray:
    path = Path(existing_embeddings_path)
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32)

    embeddings = np.asarray(np.load(path), dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError("Stored embeddings must be a 2D array")
    return embeddings


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Convert embeddings to float32 and normalize them for cosine similarity."""
    normalized = np.asarray(embeddings, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized.reshape(1, -1)
    faiss.normalize_L2(normalized)
    return normalized


def _load_existing_paths(existing_paths_path: str | Path) -> list[str]:
    path = Path(existing_paths_path)
    if not path.exists():
        return []

    loaded = np.load(path, allow_pickle=True).tolist()
    return [str(item) for item in loaded]


def add_images_to_index(
    new_image_paths,
    existing_embeddings_path,
    existing_paths_path,
    faiss_index_path,
):
    """Add new images to stored embeddings/path files and an existing FAISS flat index.

    Returns:
        Number of images that were successfully encoded and added.
    """
    embeddings_path = Path(existing_embeddings_path)
    paths_path = Path(existing_paths_path)
    index_path = Path(faiss_index_path)

    existing_embeddings = _load_existing_embeddings(embeddings_path)
    existing_paths = _load_existing_paths(paths_path)

    if existing_embeddings.shape[0] != len(existing_paths):
        raise ValueError("Existing embeddings and paths must have the same number of rows/items")

    new_embeddings_list: list[np.ndarray] = []
    new_paths_list: list[str] = []

    for image_path in new_image_paths:
        path_str = str(image_path)
        try:
            embedding = np.asarray(encode_image(path_str), dtype=np.float32)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            if embedding.ndim != 2 or embedding.shape[0] != 1:
                continue

            # Keep dimensions consistent with existing vectors.
            if existing_embeddings.size and embedding.shape[1] != existing_embeddings.shape[1]:
                continue
            if new_embeddings_list and embedding.shape[1] != new_embeddings_list[0].shape[1]:
                continue

            new_embeddings_list.append(embedding)
            new_paths_list.append(path_str)
        except Exception:
            continue

    if not new_embeddings_list:
        return 0

    new_embeddings = np.vstack(new_embeddings_list).astype(np.float32)
    new_embeddings = _normalize_embeddings(new_embeddings)

    if existing_embeddings.size:
        updated_embeddings = np.vstack([_normalize_embeddings(existing_embeddings), new_embeddings]).astype(np.float32)
    else:
        updated_embeddings = new_embeddings

    updated_paths = existing_paths + new_paths_list

    index = load_faiss_index(index_path)
    if index is None:
        index = build_faiss_index(updated_embeddings)
    else:
        # Add normalized vectors so the inner-product index behaves like cosine similarity.
        index.add(np.ascontiguousarray(new_embeddings, dtype=np.float32))

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    paths_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, updated_embeddings)
    np.save(paths_path, np.array(updated_paths, dtype=object))
    save_faiss_index(index, index_path)

    return len(new_paths_list)
