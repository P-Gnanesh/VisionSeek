from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from flask import Blueprint, jsonify, request

from src.search import load_search_assets, search_by_embedding
from src.image_encoder import encode_image
from src.text_encoder import encode_text


api_bp = Blueprint("api", __name__)


def _json_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


@api_bp.before_request
def require_api_key():
    expected_key = os.environ.get("VISIONSEEK_API_KEY", "").strip()
    if not expected_key:
        return None

    provided_key = str(request.headers.get("X-API-Key", "") or "").strip()
    if provided_key != expected_key:
        return _json_error("Unauthorized", 401)

    return None


def _parse_top_k(payload) -> int | tuple[dict, int]:
    top_k_value = payload.get("top_k", 5)
    if isinstance(top_k_value, bool):
        return _json_error("top_k must be an integer")

    try:
        top_k = int(top_k_value)
    except (TypeError, ValueError):
        return _json_error("top_k must be an integer")

    if top_k < 1:
        return _json_error("top_k must be greater than 0")

    return top_k


def _format_search_results(results: list[dict]) -> list[dict]:
    return [
        {
            "image_path": str(result.get("image_path", "")),
            "filename": Path(str(result.get("image_path", ""))).name,
            "score": float(result.get("score", 0.0)),
        }
        for result in results
    ]


@api_bp.route("/search/text", methods=["POST"])
def search_text_api():
    payload = request.get_json(silent=True)
    if payload is None:
        return _json_error("JSON body is required")

    query = payload.get("query")
    if query is None:
        return _json_error("query is required")

    if not isinstance(query, str) or not query.strip():
        return _json_error("query must be a non-empty string")

    top_k = _parse_top_k(payload)
    if isinstance(top_k, tuple):
        return top_k

    started_at = perf_counter()
    query_embedding = encode_text(query.strip())
    results = search_by_embedding(query_embedding, top_k=top_k)
    elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)

    return jsonify(
        {
            "query": query.strip(),
            "results": _format_search_results(results),
            "elapsed_ms": elapsed_ms,
        }
    )


@api_bp.route("/search/image", methods=["POST"])
def search_image_api():
    image_file = request.files.get("image")
    if image_file is None or not str(getattr(image_file, "filename", "") or "").strip():
        return _json_error("image is required")

    payload = request.form
    top_k = _parse_top_k(payload)
    if isinstance(top_k, tuple):
        return top_k

    started_at = perf_counter()
    query_embedding = encode_image(image_file)
    results = search_by_embedding(query_embedding, top_k=top_k)
    elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)

    return jsonify(
        {
            "query": str(getattr(image_file, "filename", "") or ""),
            "results": _format_search_results(results),
            "elapsed_ms": elapsed_ms,
        }
    )


@api_bp.route("/health", methods=["GET"])
def health_api():
    assets = load_search_assets()
    return jsonify({"status": "ok", "index_size": len(assets.image_paths)})