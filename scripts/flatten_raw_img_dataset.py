from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_supported_images(source_dir: Path):
    """Yield supported image files recursively from source_dir."""
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def unique_destination_path(destination_dir: Path, base_name: str, suffix: str) -> Path:
    """Create a destination path that does not overwrite an existing file."""
    candidate = destination_dir / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = destination_dir / f"{base_name}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def flatten_dataset(source_dir: Path, destination_dir: Path) -> int:
    """Copy images recursively into a flat destination folder.

    Filenames are generated as: <category>_<original_filename>
    where category is inferred from the immediate parent folder name.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    for image_path in iter_supported_images(source_dir):
        category_name = image_path.parent.name.strip() or "unknown"
        original_name = image_path.name
        output_base = f"{category_name}_{Path(original_name).stem}"
        output_suffix = image_path.suffix.lower()

        output_path = unique_destination_path(destination_dir, output_base, output_suffix)
        shutil.copy2(image_path, output_path)
        processed_count += 1

    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten dataset/images/raw-img/<category>/*.jpg into dataset/images/",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset") / "images" / "raw-img",
        help="Source directory to scan recursively (default: dataset/images/raw-img)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("dataset") / "images",
        help="Destination directory for flattened files (default: dataset/images)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = flatten_dataset(args.source, args.destination)
    print(f"Total images processed: {count}")


if __name__ == "__main__":
    main()
