from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import open_clip
import torch
from PIL import Image

from config import ALLOWED_IMAGE_EXTENSIONS, BASE_DIR, DATASET_IMAGES_DIR, MODEL_NAME, PRETRAINED_TAG


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_clip_components(
    model_name: str = MODEL_NAME,
    pretrained_tag: str = PRETRAINED_TAG,
    device: str | None = None,
):
    """Load OpenCLIP once and reuse the model, preprocess transform, and tokenizer."""
    resolved_device = device or get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(resolved_device)
    model.eval()
    return model, preprocess, tokenizer, resolved_device


def load_pil_image(image_source) -> Image.Image:
    """Load an uploaded file, path, or PIL image into RGB mode."""
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if hasattr(image_source, "read"):
        return Image.open(image_source).convert("RGB")

    return Image.open(str(image_source)).convert("RGB")


def list_image_files(images_dir: str | Path, recursive: bool = True) -> list[str]:
    images_path = Path(images_dir)
    if not images_path.exists():
        return []

    iterator: Iterable[Path]
    if recursive:
        iterator = images_path.rglob("*")
    else:
        iterator = images_path.iterdir()

    image_files: list[str] = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            image_files.append(str(path))

    return sorted(image_files)


def is_allowed_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def to_relative_image_path(path_value: str | Path, images_root: str | Path = DATASET_IMAGES_DIR) -> str:
    """Convert any stored image path into a Flask-friendly path relative to dataset/images."""
    raw_value = str(path_value).replace("\\", "/")
    normalized_root = Path(images_root).resolve()

    try:
        candidate = Path(raw_value).resolve()
        relative_candidate = candidate.relative_to(normalized_root)
        return relative_candidate.as_posix()
    except Exception:
        pass

    prefix = "dataset/images/"
    if raw_value.startswith(prefix):
        return raw_value[len(prefix) :]

    return Path(raw_value).name
