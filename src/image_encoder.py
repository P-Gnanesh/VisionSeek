from __future__ import annotations

import numpy as np
import torch

from utils.helpers import load_clip_components, load_pil_image

# Device selection (CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device for image encoding:", device)


def encode_image_batch(image_sources) -> np.ndarray:
    """Encode one or more images into normalized OpenCLIP embeddings.

    Ensures model and tensors are on the same device, uses no_grad for speed,
    and returns CPU numpy arrays of dtype float32.
    """
    image_list = list(image_sources)
    if not image_list:
        raise ValueError("image_sources cannot be empty")

    model, preprocess, _, _ = load_clip_components()
    # Ensure model is on the chosen device (idempotent)
    model = model.to(device)

    tensors = [preprocess(load_pil_image(image_source)) for image_source in image_list]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(batch)
        # L2 normalize per-vector
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().numpy().astype(np.float32)


def encode_image(image_source) -> np.ndarray:
    """Encode a single image and return a 2D array with shape (1, embedding_dim)."""
    return encode_image_batch([image_source])
