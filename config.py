from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
EMBEDDINGS_PATH = BASE_DIR / "image_embeddings.npy"
IMAGE_PATHS_PATH = BASE_DIR / "image_paths.npy"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index.bin"
UPLOAD_DIR = BASE_DIR / "uploads"

MODEL_NAME = "ViT-B-32"
PRETRAINED_TAG = "openai"
TOP_K_RESULTS = 5
BATCH_SIZE = 32
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}
