from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import faiss

from config import EMBEDDINGS_PATH, FAISS_INDEX_PATH, IMAGE_PATHS_PATH, TOP_K_RESULTS
from src.faiss_utils import build_faiss_index, load_faiss_index, save_faiss_index
from src.image_encoder import encode_image
from src.text_encoder import encode_text
from src.metadata_store import filter_paths_by_category
from utils.helpers import get_logger, to_relative_image_path


logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchAssets:
    image_paths: list[str]
    index: object | None


def _load_image_paths() -> list[str]:
    if not Path(IMAGE_PATHS_PATH).exists():
        return []
    return [str(path) for path in np.load(IMAGE_PATHS_PATH, allow_pickle=True).tolist()]


def _load_or_build_index():
    existing_index = load_faiss_index(FAISS_INDEX_PATH)

    if existing_index is not None:
        metric_type = getattr(existing_index, "metric_type", None)
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            return existing_index

    if not Path(EMBEDDINGS_PATH).exists():
        return existing_index

    if existing_index is not None:
        logger.info("Rebuilding stale FAISS index from %s", EMBEDDINGS_PATH)

    embeddings = np.load(EMBEDDINGS_PATH)
    if embeddings.size == 0:
        return None

    index = build_faiss_index(embeddings.astype(np.float32))
    save_faiss_index(index, FAISS_INDEX_PATH)
    logger.info("Rebuilt FAISS index from %s", EMBEDDINGS_PATH)
    return index


@lru_cache(maxsize=1)
def load_search_assets() -> SearchAssets:
    image_paths = _load_image_paths()
    index = _load_or_build_index()
    return SearchAssets(image_paths=image_paths, index=index)


def similarity_to_score(similarity: float) -> float:
    """Convert cosine similarity [0, 1] to percentage score [0, 100].
    
    Args:
        similarity: Cosine similarity from FAISS IndexFlatIP (range 0 to 1).
    
    Returns:
        Percentage score (0 to 100).
    """
    return float(max(0.0, similarity * 100.0))


def _build_result_record(image_path: str, similarity: float) -> dict:
    """Build a stable result payload for a single FAISS match.
    
    Args:
        image_path: Path to the matched image.
        similarity: Cosine similarity score from FAISS (0 to 1).
    
    Returns:
        Result dict with score (0–100), similarity (0–1), and metadata.
    """
    score = similarity_to_score(similarity)

    return {
        "image_path": image_path,
        "relative_path": to_relative_image_path(image_path),
        "score": score,               # 0–100 (percentage)
        "similarity": float(similarity),  # 0–1 (cosine similarity)
    }


def _search_by_embedding(
    query_embedding,
    top_k: int = TOP_K_RESULTS,
    category_filter: str | None = None,
) -> list[dict]:

    assets = load_search_assets()
    if assets.index is None or not assets.image_paths:
        return []

    query_array = np.asarray(query_embedding, dtype=np.float32)

    if query_array.ndim == 1:
        query_array = query_array.reshape(1, -1)

    # Normalize query for cosine similarity (matching stored normalized embeddings)
    faiss.normalize_L2(query_array)

    total_candidates = len(assets.image_paths)
    target_count = min(max(int(top_k), 1), total_candidates)

    # =========================
    # WITHOUT CATEGORY FILTER
    # =========================
    if not category_filter:
        similarities, indices = assets.index.search(query_array, target_count)

        results: list[dict] = []

        for sim, index_position in zip(similarities[0], indices[0]):
            if index_position < 0 or index_position >= total_candidates:
                continue

            image_path = str(assets.image_paths[index_position])
            results.append(_build_result_record(image_path, float(sim)))

        return results

    # =========================
    # WITH CATEGORY FILTER
    # =========================
    collected_results: list[dict] = []
    seen_paths: set[str] = set()
    search_size = target_count

    while search_size <= total_candidates and len(collected_results) < target_count:

        similarities, indices = assets.index.search(query_array, search_size)

        candidate_paths: list[str] = []
        candidate_records: list[tuple[str, float]] = []

        for sim, index_position in zip(similarities[0], indices[0]):
            if index_position < 0 or index_position >= total_candidates:
                continue

            image_path = str(assets.image_paths[index_position])
            candidate_paths.append(image_path)
            candidate_records.append((image_path, float(sim)))

        allowed_paths = set(filter_paths_by_category(candidate_paths, category_filter))

        for image_path, sim in candidate_records:
            if image_path not in allowed_paths or image_path in seen_paths:
                continue

            seen_paths.add(image_path)
            collected_results.append(_build_result_record(image_path, sim))

            if len(collected_results) >= target_count:
                break

        if search_size >= total_candidates:
            break

        search_size = min(total_candidates, search_size + target_count)

    return collected_results[:target_count]


def search_by_embedding(
    query_embedding,
    top_k: int = TOP_K_RESULTS,
    category_filter: str | None = None,
) -> list[dict]:
    return _search_by_embedding(query_embedding, top_k=top_k, category_filter=category_filter)


def text_to_image_search(
    query: str,
    top_k: int = TOP_K_RESULTS,
    category_filter: str | None = None,
) -> list[dict]:

    query_text = (query or "").strip()
    if not query_text:
        return []

    query_embedding = encode_text(query_text)
    return _search_by_embedding(query_embedding, top_k=top_k, category_filter=category_filter)


def image_to_image_search(
    image_source,
    top_k: int = TOP_K_RESULTS,
    category_filter: str | None = None,
) -> list[dict]:

    query_embedding = encode_image(image_source)
    return _search_by_embedding(query_embedding, top_k=top_k, category_filter=category_filter)