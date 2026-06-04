from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from config import BASE_DIR, DATASET_IMAGES_DIR, IMAGE_PATHS_PATH
from utils.helpers import list_image_files


METADATA_PATH = BASE_DIR / "metadata.json"


def _basename_key(path_value: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    # Support both slash styles regardless of platform.
    return os.path.basename(raw.replace("\\", "/")).strip()


def _candidate_image_paths() -> list[str]:
    """Collect image paths only from flattened dataset/images folder."""
    return list_image_files(DATASET_IMAGES_DIR, recursive=False)


def _infer_category_from_path(image_path: str) -> str:
    filename = Path(str(image_path)).name
    if not filename:
        return "unknown"

    category = filename.split("_")[0]

    ITALIAN_TO_ENGLISH = {
        "cane": "dog",
        "gatto": "cat",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "chicken",
        "mucca": "cow",
        "pecora": "sheep",
        "ragno": "spider",
        "scoiattolo": "squirrel"
    }

    return ITALIAN_TO_ENGLISH.get(category, category)


def _generate_metadata() -> dict[str, str]:
    """Build filename -> category mapping and persist to metadata.json."""
    metadata: dict[str, str] = {}

    for image_path in _candidate_image_paths():
        filename = Path(str(image_path)).name
        if not filename:
            continue
        metadata[filename] = _infer_category_from_path(str(image_path))

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    return metadata


def _load_metadata() -> dict[str, str]:
    """Load metadata from disk, auto-generating it when missing or invalid."""
    if not METADATA_PATH.exists():
        return _generate_metadata()

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return _generate_metadata()

    if not isinstance(metadata, dict):
        return _generate_metadata()

    normalized: dict[str, str] = {}
    for key, value in metadata.items():
        normalized[str(key)] = str(value)
    return normalized


def get_category(image_path) -> str | None:
    """Return category label for an image path by looking up its filename."""
    filename = _basename_key(str(image_path))
    if not filename:
        return None

    metadata = _load_metadata()
    for stored_key, category in metadata.items():
        if _basename_key(str(stored_key)) == filename:
            return category
    return None


def get_all_categories() -> list[str]:
    """Return sorted unique category labels from metadata.json."""
    metadata = _load_metadata()
    categories = {str(category) for category in metadata.values() if str(category).strip()}
    return sorted(categories)


def filter_paths_by_category(paths, category) -> list[str]:
    """Filter input paths by category label."""
    target = str(category or "").strip().lower()
    if not target:
        return [str(path) for path in paths]

    metadata = _load_metadata()
    category_by_filename: dict[str, str] = {}
    for stored_key, stored_category in metadata.items():
        key = _basename_key(str(stored_key))
        if key and key not in category_by_filename:
            category_by_filename[key] = str(stored_category or "").strip().lower()

    filtered: list[str] = []

    for image_path in paths:
        path_str = str(image_path)
        normalized_name = _basename_key(path_str)
        if not normalized_name:
            continue
        if category_by_filename.get(normalized_name) == target:
            filtered.append(path_str)

    return filtered
