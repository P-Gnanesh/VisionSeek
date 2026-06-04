import argparse
from pathlib import Path

import numpy as np
from config import BATCH_SIZE, DATASET_IMAGES_DIR, EMBEDDINGS_PATH, FAISS_INDEX_PATH, IMAGE_PATHS_PATH
from src.faiss_utils import build_faiss_index, save_faiss_index
from src.image_encoder import encode_image_batch
from utils.helpers import get_logger, list_image_files


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate image embeddings and FAISS index.")
    parser.add_argument(
        "--images-dir",
        default=str(DATASET_IMAGES_DIR),
        help="Root image directory to scan (default: dataset/images)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan subfolders for images (default: enabled)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level image folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = args.images_dir
    recursive = bool(args.recursive)

    images_dir_path = Path(images_dir)

    if not images_dir_path.is_dir():
        logger.error("Image folder not found: %s", images_dir)
        return

    image_files = list_image_files(images_dir_path, recursive=recursive)
    if not image_files:
        logger.warning("No supported images found in: %s", images_dir)
        return

    logger.info("Found %d images to embed (recursive=%s).", len(image_files), recursive)

    embeddings_list: list[np.ndarray] = []
    valid_image_paths: list[str] = []

    for start_index in range(0, len(image_files), BATCH_SIZE):
        batch_paths = image_files[start_index : start_index + BATCH_SIZE]
        try:
            batch_embeddings = encode_image_batch(batch_paths)
            embeddings_list.append(batch_embeddings)
            valid_image_paths.extend(batch_paths)
            logger.info("Processed %d/%d images", len(valid_image_paths), len(image_files))
        except Exception as exc:
            logger.warning("Batch starting at %d failed, falling back to single-image processing: %s", start_index, exc)

            for image_path in batch_paths:
                try:
                    single_embedding = encode_image_batch([image_path])
                    embeddings_list.append(single_embedding)
                    valid_image_paths.append(image_path)
                    logger.info("Processed %d/%d images", len(valid_image_paths), len(image_files))
                except Exception as image_exc:
                    logger.exception("Skipping unreadable image %s: %s", image_path, image_exc)

    if not embeddings_list:
        logger.error("No embeddings were generated.")
        return

    image_embeddings = np.vstack(embeddings_list).astype(np.float32)
    # Note: encode_image_batch already returns L2-normalized embeddings.
    # Do NOT normalize again here — build_faiss_index also normalizes before
    # adding to the index, causing a triple-normalization mismatch with query embeddings.

    np.save(EMBEDDINGS_PATH, image_embeddings)
    np.save(IMAGE_PATHS_PATH, np.array(valid_image_paths, dtype=object))

    index = build_faiss_index(image_embeddings)
    save_faiss_index(index, FAISS_INDEX_PATH)

    logger.info("Final embedding shape: %s", image_embeddings.shape)
    logger.info("Saved embeddings to: %s", EMBEDDINGS_PATH)
    logger.info("Saved image paths to: %s", IMAGE_PATHS_PATH)
    logger.info("Saved FAISS index to: %s", FAISS_INDEX_PATH)
    logger.info("Embedding generation completed successfully.")


if __name__ == "__main__":
    main()