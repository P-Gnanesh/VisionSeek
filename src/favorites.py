from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR


FAVORITES_PATH = BASE_DIR / "favorites.json"


def _load_favorites() -> list[dict]:
    if not FAVORITES_PATH.exists():
        return []

    try:
        with FAVORITES_PATH.open("r", encoding="utf-8") as file_handle:
            favorites = json.load(file_handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if isinstance(favorites, list):
        return [item for item in favorites if isinstance(item, dict)]

    return []


def _save_favorites(favorites: list[dict]) -> None:
    with FAVORITES_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(favorites, file_handle, indent=2)


def add_favorite(image_path):
    favorites = _load_favorites()
    image_path = str(image_path)

    favorites = [item for item in favorites if str(item.get("image_path", "")) != image_path]
    favorites.append(
        {
            "image_path": image_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_favorites(favorites)
    return favorites


def remove_favorite(image_path):
    favorites = _load_favorites()
    image_path = str(image_path)
    favorites = [item for item in favorites if str(item.get("image_path", "")) != image_path]
    _save_favorites(favorites)
    return favorites


def get_favorites():
    return _load_favorites()