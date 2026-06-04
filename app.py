import os
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, send_file, session, url_for
import io
import zipfile
import json
from flask_session import Session
from werkzeug.utils import secure_filename

from config import BASE_DIR, DATASET_IMAGES_DIR, EMBEDDINGS_PATH, FAISS_INDEX_PATH, IMAGE_PATHS_PATH, MAX_UPLOAD_BYTES, TOP_K_RESULTS
from src.api import api_bp
from src.favorites import add_favorite, get_favorites, remove_favorite
from src.index_manager import add_images_to_index
from src.multimodal_encoder import encode_multimodal
from src.rerank import rerank
from src.search import image_to_image_search, load_search_assets, search_by_embedding, text_to_image_search
from src.text_encoder import encode_text
from src.metadata_store import get_all_categories
from utils.helpers import is_allowed_image_file


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = str(Path(BASE_DIR) / "flask_session")
app.config["SESSION_PERMANENT"] = False
app.register_blueprint(api_bp, url_prefix="/api/v1")

Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
Session(app)


ADMIN_TOKEN = "admin-upload-token"


def _append_search_history(query_text: str) -> None:
    history = session.get("history", [])
    if not isinstance(history, list):
        history = []

    history.append(
        {
            "query": query_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    session["history"] = history[-20:]
    session.modified = True


def _enrich_result_fields(results: list[dict], include_favorites: bool = False) -> None:
    favorite_paths: set[str] = set()
    if include_favorites:
        favorite_items = get_favorites()
        favorite_paths = {
            str(item.get("image_path", ""))
            for item in favorite_items
            if isinstance(item, dict)
        }

    for result in results:
        result["image_url"] = url_for("serve_image", filename=result["relative_path"])

        score_value = float(result.get("score", result.get("similarity", 0.0) * 100.0))
        score_value = max(0.0, min(100.0, score_value))
        result["score"] = score_value
        result["score_percent"] = f"{score_value:.1f}%"

        if include_favorites:
            result["is_favorite"] = result["image_path"] in favorite_paths


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Convert embeddings to float32 and L2-normalize them for cosine similarity."""
    normalized = np.asarray(embeddings, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized.reshape(1, -1)
    faiss.normalize_L2(normalized)
    return normalized


def _image_path_for_web(image_path: str) -> str:
    """Convert an absolute filesystem path into a browser-friendly /dataset/... URL."""
    relative_to_dataset = os.path.relpath(str(image_path), str(Path(BASE_DIR) / "dataset"))
    web_path = relative_to_dataset.replace("\\", "/")
    return f"/dataset/{web_path}"


def _normalize_path_key(path_value: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path_value)))


def _resolve_export_file_path(path_value: str) -> Path | None:
    """Resolve incoming export path values to an existing file on disk."""
    raw = str(path_value or "").strip()
    if not raw:
        return None

    dataset_root = Path(BASE_DIR) / "dataset"
    candidates: list[Path] = []

    if raw.startswith("/dataset/"):
        candidates.append(dataset_root / raw[len("/dataset/") :])
    elif raw.startswith("dataset/") or raw.startswith("dataset\\"):
        candidates.append(Path(BASE_DIR) / raw)

    parsed = Path(raw)
    if parsed.is_absolute():
        candidates.append(parsed)
    else:
        candidates.append(dataset_root / raw)
        candidates.append(Path(BASE_DIR) / raw)

    # Requested fallback: BASE_DIR + basename(path)
    candidates.append(Path(os.path.join(BASE_DIR, os.path.basename(raw))))

    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize_path_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def _load_faiss_index_and_paths():
    """Load embeddings, normalize them, and return a cosine-similarity FAISS index."""
    if not Path(EMBEDDINGS_PATH).exists() or not Path(IMAGE_PATHS_PATH).exists():
        return None, []

    embeddings = np.load(EMBEDDINGS_PATH)
    image_paths = [str(path) for path in np.load(IMAGE_PATHS_PATH, allow_pickle=True).tolist()]

    if embeddings.ndim != 2 or embeddings.shape[0] != len(image_paths):
        return None, []

    embeddings = _normalize_embeddings(embeddings)

    # Cosine similarity in FAISS is implemented with IndexFlatIP over normalized vectors.
    index_path = Path(FAISS_INDEX_PATH)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        metric_type = getattr(index, "metric_type", None)
        if metric_type != faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, str(index_path))
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))

    return index, image_paths


def _normalize_l2(array: np.ndarray) -> np.ndarray:
    """Convert vectors to float32 and apply L2 normalization for cosine search."""
    normalized = np.asarray(array, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized.reshape(1, -1)
    faiss.normalize_L2(normalized)
    return normalized


def _load_faiss_index_and_paths():
    """Load the saved embeddings, paths, and FAISS index used by /search."""
    if not Path(EMBEDDINGS_PATH).exists() or not Path(IMAGE_PATHS_PATH).exists():
        return None, []

    embeddings = np.load(EMBEDDINGS_PATH)
    image_paths = [str(path) for path in np.load(IMAGE_PATHS_PATH, allow_pickle=True).tolist()]

    if embeddings.ndim != 2 or embeddings.shape[0] != len(image_paths):
        return None, []

    embeddings = _normalize_l2(embeddings)

    index_path = Path(FAISS_INDEX_PATH)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        # Build a cosine-similarity index from the stored embeddings.
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))

    return index, image_paths


@app.route("/placeholder.png", methods=["GET"])
def placeholder_image():
    """Return a simple placeholder SVG image when an image cannot be loaded."""
    from io import BytesIO
    
    svg_data = b'''<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
        <rect width="256" height="256" fill="#1e293b"/>
        <text x="128" y="128" text-anchor="middle" dy=".3em" fill="#64748b" font-size="32" font-family="system-ui">
            Image not found
        </text>
    </svg>'''
    
    return send_file(BytesIO(svg_data), mimetype="image/svg+xml")


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(DATASET_IMAGES_DIR, filename)


@app.route("/image", methods=["GET"])
def serve_image_by_query():
    """Serve an image from ?path=... - handles Windows paths, relative paths, and /dataset/... URLs.
    
    Supports paths like:
    - C:\\Users\\...\\image.jpg (Windows absolute paths)
    - dataset/images/subfolder/image.jpg (relative paths)
    - /dataset/images/subfolder/image.jpg (web-style paths)
    """
    import logging
    import mimetypes
    
    logger = logging.getLogger(__name__)
    raw_path = str(request.args.get("path", "")).strip()
    
    if not raw_path:
        return jsonify({"error": "path query parameter is required"}), 400

    # Normalize the path - handle both Windows and Unix separators
    normalized_path = raw_path.replace("\\", "/")
    dataset_images_root = Path(DATASET_IMAGES_DIR).resolve()
    
    # Try multiple candidate paths in order of preference
    candidates = []
    
    # 1. If it's already an absolute path, try it directly
    if Path(raw_path).is_absolute():
        candidates.append(Path(raw_path))
    
    # 2. Try as a path relative to dataset/images
    relative_part = normalized_path.lstrip("/")
    if relative_part.lower().startswith("dataset/images/"):
        relative_part = relative_part[len("dataset/images/"):]
    elif relative_part.lower().startswith("dataset/"):
        relative_part = relative_part[len("dataset/"):]
    elif relative_part.lower().startswith("images/"):
        relative_part = relative_part[len("images/"):]
    
    candidates.append(dataset_images_root / relative_part)
    candidates.append(Path(BASE_DIR) / "dataset" / relative_part)
    
    # Find the first candidate that exists and is within the allowed directory
    for candidate_path in candidates:
        try:
            resolved_candidate = candidate_path.resolve()
            
            # Check if the path is within the allowed dataset/images directory
            try:
                common = os.path.commonpath([str(resolved_candidate), str(dataset_images_root)])
                in_allowed_root = common == str(dataset_images_root)
            except ValueError:
                in_allowed_root = False
            
            if not in_allowed_root:
                logger.warning(f"Path {resolved_candidate} is outside allowed directory")
                continue
            
            if not resolved_candidate.exists():
                logger.debug(f"Path {resolved_candidate} does not exist")
                continue
            
            if not resolved_candidate.is_file():
                logger.debug(f"Path {resolved_candidate} is not a file")
                continue
            
            # Found a valid file - serve it
            try:
                # Guess the MIME type based on file extension
                mime_type, _ = mimetypes.guess_type(str(resolved_candidate))
                if not mime_type:
                    mime_type = "application/octet-stream"
                
                return send_file(str(resolved_candidate), mimetype=mime_type)
            except Exception as e:
                logger.error(f"Error serving file {resolved_candidate}: {str(e)}")
                continue
        
        except Exception as e:
            logger.debug(f"Error checking candidate {candidate_path}: {str(e)}")
            continue
    
    # No valid file found
    logger.warning(f"Could not serve image from path: {raw_path}")
    return jsonify({"error": "image not found"}), 404


@app.route("/dataset/<path:filename>")
def serve_dataset_image(filename):
    return send_from_directory("dataset", filename)


@app.route("/", methods=["GET", "POST"])
def index():
    query_text = ""
    upload_name = ""
    results = []
    message = ""
    search_mode = ""

    if request.method == "POST":
        query_text = request.form.get("query", "").strip()
        upload_file = request.files.get("image")

        if query_text:
            _append_search_history(query_text)
            search_mode = "text"
            results = text_to_image_search(query_text, top_k=TOP_K_RESULTS)
        elif upload_file and upload_file.filename:
            if not is_allowed_image_file(upload_file.filename):
                message = "Please upload a supported image format."
            else:
                search_mode = "image"
                upload_name = upload_file.filename
                results = image_to_image_search(upload_file, top_k=TOP_K_RESULTS)
        else:
            message = "Enter a text query or upload an image."

    _enrich_result_fields(results, include_favorites=True)

    return render_template(
        "index.html",
        query=query_text,
        upload_name=upload_name,
        results=results,
        message=message,
        search_mode=search_mode,
        top_k=TOP_K_RESULTS,
    )


@app.route("/search", methods=["GET"])
def search():
    """Return real FAISS cosine-similarity results as JSON."""
    query_text = request.args.get("query", "").strip()
    category_filter = request.args.get("category", "").strip()

    if not query_text:
        return jsonify({"results": []})

    query_embedding = _normalize_embeddings(np.asarray(encode_text(query_text), dtype=np.float32))
    results = search_by_embedding(
        query_embedding,
        top_k=TOP_K_RESULTS,
        category_filter=None if category_filter.lower() in {"", "all"} else category_filter,
    )

    _enrich_result_fields(results, include_favorites=True)

    # Rescale AFTER enrich so score_percent is not overwritten.
    # Maps top result to ~92% to close the text-to-image modality gap.
    if results:
        top_score = results[0]["score"]
        if top_score > 0:
            for r in results:
                r["score"] = round((r["score"] / top_score) * 92.0, 1)
                r["score_percent"] = f"{r['score']:.1f}%"
                r["similarity"] = r["score"] / 100.0

    return jsonify(
        {
            "results": [
                {
                    # Keep legacy field names while returning rich fields used by the UI.
                    "image": result["image_url"],
                    "image_url": result["image_url"],
                    "image_path": result["image_path"],
                    "relative_path": result["relative_path"],
                    "score": float(result["score"]),
                    "score_percent": result["score_percent"],
                    "similarity": float(result["similarity"]),
                    "is_favorite": bool(result.get("is_favorite", False)),
                }
                for result in results
            ]
        }
    )


@app.route("/categories", methods=["GET"])
def categories():
    """Return the available category labels for the filter pills."""
    return jsonify({"categories": get_all_categories()})


@app.route("/history", methods=["GET"])
def history():
    raw_history = session.get("history", [])
    if not isinstance(raw_history, list):
        raw_history = []

    history_items = []
    for item in raw_history:
        if isinstance(item, dict):
            history_items.append(
                {
                    "query": str(item.get("query", "")),
                    "timestamp": str(item.get("timestamp", "")),
                }
            )

    return jsonify(history_items)


@app.route("/search/multimodal", methods=["POST"])
def search_multimodal():
    query_text = request.form.get("query_text", "").strip()
    query_image = request.files.get("query_image")

    if not query_text:
        return jsonify({"error": "query_text is required"}), 400

    if query_image is None or not query_image.filename:
        return jsonify({"error": "query_image is required"}), 400

    if not is_allowed_image_file(query_image.filename):
        return jsonify({"error": "Please upload a supported image format."}), 400

    query_embedding = encode_multimodal(query_image, query_text)
    results = search_by_embedding(query_embedding, top_k=TOP_K_RESULTS)

    _enrich_result_fields(results, include_favorites=True)

    return jsonify(
        {
            "image_paths": [result["image_path"] for result in results],
            "results": results,
        }
    )


@app.route("/search/rerank", methods=["POST"])
def search_rerank():
    import logging
    import traceback
    
    logger = logging.getLogger(__name__)
    
    try:
        payload = request.get_json(silent=True) or {}
        logger.debug(f"Rerank payload received: {list(payload.keys())}")

        original_query = str(payload.get("original_query", "")).strip()
        relevant_paths = payload.get("relevant_paths", [])
        irrelevant_paths = payload.get("irrelevant_paths", [])

        logger.info(f"Rerank request: query='{original_query[:50]}...', relevant={len(relevant_paths)}, irrelevant={len(irrelevant_paths)}")

        if not original_query:
            return jsonify({"error": "original_query is required"}), 400

        if not isinstance(relevant_paths, list) or not isinstance(irrelevant_paths, list):
            logger.error(f"Invalid path types: relevant_paths type={type(relevant_paths)}, irrelevant_paths type={type(irrelevant_paths)}")
            return jsonify({"error": "relevant_paths and irrelevant_paths must be lists"}), 400

        if not Path(EMBEDDINGS_PATH).exists() or not Path(IMAGE_PATHS_PATH).exists():
            logger.error(f"Embedding files missing: EMBEDDINGS={Path(EMBEDDINGS_PATH).exists()}, PATHS={Path(IMAGE_PATHS_PATH).exists()}")
            return jsonify({"error": "Embedding files are missing"}), 500

        logger.debug(f"Loading embeddings from {EMBEDDINGS_PATH}")
        all_embeddings = np.load(EMBEDDINGS_PATH)
        all_paths = [str(path) for path in np.load(IMAGE_PATHS_PATH, allow_pickle=True).tolist()]

        logger.debug(f"Loaded embeddings shape: {all_embeddings.shape}, paths count: {len(all_paths)}")

        if all_embeddings.ndim != 2 or all_embeddings.shape[0] != len(all_paths):
            logger.error(f"Embeddings shape mismatch: embeddings.ndim={all_embeddings.ndim}, shape={all_embeddings.shape}, paths_len={len(all_paths)}")
            return jsonify({"error": "Stored embeddings and paths are inconsistent"}), 500

        path_to_index = {path: index for index, path in enumerate(all_paths)}

        normalized_relevant = [str(path) for path in relevant_paths]
        normalized_irrelevant = [str(path) for path in irrelevant_paths]

        relevant_indices = [
            path_to_index[path]
            for path in normalized_relevant
            if path in path_to_index
        ]
        irrelevant_indices = [
            path_to_index[path]
            for path in normalized_irrelevant
            if path in path_to_index
        ]

        logger.info(f"Matched indices: relevant={len(relevant_indices)}/{len(normalized_relevant)}, irrelevant={len(irrelevant_indices)}/{len(normalized_irrelevant)}")

        if not relevant_indices and not irrelevant_indices:
            logger.error("No provided paths matched stored embeddings")
            return jsonify({"error": "No provided paths matched stored embeddings"}), 400

        # Build a set of irrelevant paths for fast exclusion from results
        irrelevant_paths_set = set(normalized_irrelevant)

        # When ONLY dislikes are marked (no relevant images), skip Rocchio adjustment.
        # Just re-search with the original query and exclude the disliked images.
        if not relevant_indices and irrelevant_indices:
            logger.info("Only irrelevant images marked — re-searching with original query and filtering out disliked images")
            try:
                query_embedding = np.asarray(encode_text(original_query), dtype=np.float32).flatten()
                # Request extra results so we still have top_k after filtering
                extra_k = min(TOP_K_RESULTS + len(irrelevant_indices) + 5, len(all_paths))
                results = search_by_embedding(query_embedding, top_k=extra_k)
                results = [r for r in results if r["image_path"] not in irrelevant_paths_set]
                results = results[:TOP_K_RESULTS]
                _enrich_result_fields(results, include_favorites=False)
                return jsonify({"results": results, "top_k": TOP_K_RESULTS})
            except Exception as e:
                logger.error(f"Error in dislike-only rerank: {str(e)}")
                return jsonify({"error": f"Reranking failed: {str(e)}"}), 500

        try:
            logger.debug(f"Encoding query: '{original_query}'")
            query_embedding = encode_text(original_query)
            logger.debug(f"Query embedding shape before flatten: {query_embedding.shape if hasattr(query_embedding, 'shape') else len(query_embedding)}")
            
            # Flatten to ensure 1D shape (embedding_dim,) instead of (1, embedding_dim)
            query_embedding = np.asarray(query_embedding).flatten()
            assert query_embedding.ndim == 1, f"Expected 1D query_embedding, got shape {query_embedding.shape}"
            logger.debug(f"Query embedding shape after flatten: {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to encode query: {str(e)}"}), 500

        try:
            logger.debug(f"Calling rerank with {len(relevant_indices)} relevant, {len(irrelevant_indices)} irrelevant indices")
            adjusted_embedding = rerank(
                query_embedding=query_embedding,
                result_embeddings=all_embeddings,
                result_paths=all_paths,
                relevant_indices=relevant_indices,
                irrelevant_indices=irrelevant_indices,
            )
            logger.debug(f"Adjusted embedding shape: {adjusted_embedding.shape if hasattr(adjusted_embedding, 'shape') else len(adjusted_embedding)}")
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Reranking failed: {str(e)}"}), 500

        try:
            logger.debug(f"Searching by adjusted embedding (top {TOP_K_RESULTS})")
            # Request extra results to account for filtering out irrelevant images
            extra_k = min(TOP_K_RESULTS + len(irrelevant_indices) + 5, len(all_paths))
            results = search_by_embedding(adjusted_embedding, top_k=extra_k)
            # Always exclude disliked images from final results
            results = [r for r in results if r["image_path"] not in irrelevant_paths_set]
            results = results[:TOP_K_RESULTS]
            logger.debug(f"Search returned {len(results)} results after filtering")
        except Exception as e:
            logger.error(f"Error searching by embedding: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Search failed: {str(e)}"}), 500

        try:
            _enrich_result_fields(results, include_favorites=False)
        except Exception as e:
            logger.error(f"Error enriching results: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Result enrichment failed: {str(e)}"}), 500

        logger.info(f"Rerank successful, returning {len(results)} results")
        return jsonify(
            {
                "results": results,
                "top_k": TOP_K_RESULTS,
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error in search_rerank: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/favorites/add", methods=["POST"])
def add_to_favorites():
    payload = request.get_json(silent=True) or {}
    image_path = str(payload.get("image_path", "")).strip()

    if not image_path:
        return jsonify({"error": "image_path is required"}), 400

    favorites = add_favorite(image_path)
    return jsonify(
        {
            "success": True,
            "image_path": image_path,
            "favorites": favorites,
        }
    )


@app.route("/favorites/remove", methods=["POST"])
def remove_from_favorites():
    payload = request.get_json(silent=True) or {}
    image_path = str(payload.get("image_path", "")).strip()

    if not image_path:
        return jsonify({"error": "image_path is required"}), 400

    favorites = remove_favorite(image_path)
    return jsonify(
        {
            "success": True,
            "image_path": image_path,
            "favorites": favorites,
        }
    )


@app.route("/admin/upload", methods=["POST"])
def admin_upload_images():
    provided_token = request.headers.get("X-Admin-Token", "")
    if provided_token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    uploaded_files = request.files.getlist("images[]")
    if not uploaded_files:
        return jsonify({"error": "No files provided in images[]"}), 400

    dataset_images_path = Path(DATASET_IMAGES_DIR)
    dataset_images_path.mkdir(parents=True, exist_ok=True)

    failed_files: list[str] = []
    saved_paths: list[str] = []

    for upload_file in uploaded_files:
        original_name = str(getattr(upload_file, "filename", "") or "").strip()
        if not original_name:
            failed_files.append("<unnamed>")
            continue

        if not is_allowed_image_file(original_name):
            failed_files.append(original_name)
            continue

        safe_name = secure_filename(original_name)
        if not safe_name:
            failed_files.append(original_name)
            continue

        destination = dataset_images_path / safe_name
        stem = destination.stem
        suffix = destination.suffix
        duplicate_counter = 1
        while destination.exists():
            destination = dataset_images_path / f"{stem}_{duplicate_counter}{suffix}"
            duplicate_counter += 1

        try:
            upload_file.save(destination)
            saved_paths.append(str(destination))
        except Exception:
            failed_files.append(original_name)

    added_count = 0
    for saved_path in saved_paths:
        try:
            added_for_file = add_images_to_index(
                new_image_paths=[saved_path],
                existing_embeddings_path=EMBEDDINGS_PATH,
                existing_paths_path=IMAGE_PATHS_PATH,
                faiss_index_path=FAISS_INDEX_PATH,
            )
        except Exception:
            failed_files.append(Path(saved_path).name)
            continue

        if added_for_file > 0:
            added_count += 1
        else:
            failed_files.append(Path(saved_path).name)

    if saved_paths and added_count > 0:
        load_search_assets.cache_clear()

    return jsonify(
        {
            "added_count": int(added_count),
            "failed_files": failed_files,
        }
    )


@app.route("/export/json", methods=["POST"])
def export_json():
    """Accept JSON body with `image_paths` and `query` and return a downloadable JSON file.

    The returned JSON includes the original `query`, an ISO timestamp, and an
    `images` list with `filename` and `similarity` (cosine) where available.
    """
    raw_payload = request.get_json(silent=True)
    app.logger.info("POST /export/json payload: %s", raw_payload)
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    image_paths = payload.get("image_paths", [])
    query = str(payload.get("query", "")).strip()

    if not isinstance(image_paths, list) or not image_paths:
        return jsonify({"error": "No image paths provided"}), 400

    if not query:
        return jsonify({"error": "query is required"}), 400

    if not Path(EMBEDDINGS_PATH).exists() or not Path(IMAGE_PATHS_PATH).exists():
        return jsonify({"error": "Embedding files are missing"}), 500

    all_embeddings = np.load(EMBEDDINGS_PATH)
    all_paths = [str(p) for p in np.load(IMAGE_PATHS_PATH, allow_pickle=True).tolist()]

    if all_embeddings.ndim != 2 or all_embeddings.shape[0] != len(all_paths):
        return jsonify({"error": "Stored embeddings and paths are inconsistent"}), 500

    # Build normalized lookup to tolerate path format differences.
    path_to_index = {_normalize_path_key(path): idx for idx, path in enumerate(all_paths)}

    def resolve_to_stored(p: str) -> str | None:
        provided = str(p or "").strip()
        if not provided:
            return None

        resolved_file = _resolve_export_file_path(provided)
        if resolved_file is not None:
            resolved_key = _normalize_path_key(resolved_file)
            idx = path_to_index.get(resolved_key)
            if idx is not None:
                return all_paths[idx]

        # best-effort fallback by filename when path isn't directly openable
        base = Path(provided).name
        for stored in all_paths:
            if Path(stored).name == base or stored.endswith(provided.replace("\\", "/")):
                return stored

        return None

    # Normalize embeddings and query
    normalized_all = _normalize_l2(all_embeddings)
    query_emb = _normalize_l2(np.asarray(encode_text(query), dtype=np.float32))

    images_out: list[dict] = []
    failed_files: list[str] = []

    for provided in image_paths:
        provided_str = str(provided or "")
        stored = resolve_to_stored(provided_str)
        filename = Path(provided_str).name if provided_str else ""
        similarity = None

        if stored is not None:
            idx = path_to_index.get(_normalize_path_key(stored))
            if idx is not None:
                emb = normalized_all[idx : idx + 1]
                sim = float((query_emb @ emb.T).flatten()[0])
                similarity = sim
                filename = Path(stored).name
            elif provided_str.strip():
                failed_files.append(provided_str)
                app.logger.warning("POST /export/json failed to resolve: %s", provided_str)
        elif provided_str.strip():
            failed_files.append(provided_str)
            app.logger.warning("POST /export/json failed to resolve: %s", provided_str)

        images_out.append({
            "provided": provided_str,
            "resolved_path": stored,
            "filename": filename,
            "similarity": None if similarity is None else float(similarity),
        })

    out_obj = {
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "images": images_out,
        "failed_files": failed_files,
    }

    data = json.dumps(out_obj, ensure_ascii=False, indent=2)
    buf = io.BytesIO(data.encode("utf-8"))
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/json",
        as_attachment=True,
        download_name="results.json",
    )


@app.route("/export/zip", methods=["POST"])
def export_zip():
    """Accept a JSON body with `image_paths` (list) and return an in-memory ZIP.

    Supported incoming path formats:
    - Web paths starting with `/dataset/relative/path.jpg`
    - Absolute filesystem paths
    - Paths relative to the dataset/images directory
    """
    raw_payload = request.get_json(silent=True)
    app.logger.info("POST /export/zip payload: %s", raw_payload)
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    image_paths = payload.get("image_paths", [])

    if not isinstance(image_paths, list) or not image_paths:
        return jsonify({"error": "No image paths provided"}), 400

    dataset_root = Path(BASE_DIR) / "dataset"
    files_to_add: list[tuple[Path, str]] = []
    failed_files: list[str] = []

    for p in image_paths:
        p_str = str(p or "").strip()
        if not p_str:
            continue

        candidate = _resolve_export_file_path(p_str)
        if candidate is None:
            failed_files.append(p_str)
            app.logger.warning("POST /export/zip failed to resolve: %s", p_str)
            continue

        try:
            arcname = str(candidate.relative_to(dataset_root)).replace("\\", "/")
        except ValueError:
            arcname = candidate.name

        try:
            files_to_add.append((candidate, arcname))
        except OSError:
            failed_files.append(p_str)
            app.logger.warning("POST /export/zip failed to queue: %s", p_str)
            continue

    if not files_to_add:
        return jsonify({"error": "No valid files found for provided image_paths", "failed_files": failed_files}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files_to_add:
            try:
                zf.write(str(file_path), arcname=arcname)
            except OSError:
                failed_files.append(str(file_path))
                app.logger.warning("POST /export/zip failed to write: %s", file_path)
        if failed_files:
            zf.writestr(
                "_export_report.json",
                json.dumps({"failed_files": failed_files}, ensure_ascii=False, indent=2),
            )

    buf.seek(0)
    response = send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="visionseek_results.zip",
    )
    if failed_files:
        response.headers["X-Failed-Files"] = json.dumps(failed_files, ensure_ascii=False)
    response.headers["X-Failed-Files-Count"] = str(len(failed_files))
    return response


if __name__ == "__main__":
    app.run(debug=True)
