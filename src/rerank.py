from __future__ import annotations

import numpy as np
import faiss


def rerank_by_feedback(
    query_embedding,
    result_embeddings,
    result_paths,
    relevant_indices,
    irrelevant_indices,
):
    """Adjust a query embedding using Rocchio relevance feedback.

    Args:
        query_embedding: Query vector of shape (d,).
        result_embeddings: Retrieved result vectors of shape (n, d).
        result_paths: Retrieved result paths of length n (used for size validation).
        relevant_indices: Indices of relevant results in result_embeddings.
        irrelevant_indices: Indices of irrelevant results in result_embeddings.

    Returns:
        A unit-normalized query embedding after Rocchio adjustment.
    """
    query = np.asarray(query_embedding, dtype=np.float64)
    embeddings = np.asarray(result_embeddings, dtype=np.float64)

    if query.ndim != 1:
        raise ValueError("query_embedding must be a 1D vector")

    if embeddings.ndim != 2:
        raise ValueError("result_embeddings must be a 2D array")

    if embeddings.shape[0] != len(result_paths):
        raise ValueError("result_embeddings and result_paths must have matching lengths")

    if embeddings.shape[1] != query.shape[0]:
        raise ValueError("Embedding dimensionality mismatch between query and results")

    # Rocchio parameters
    alpha = 0.75  # positive feedback weight
    beta = 0.3  # negative feedback weight (reduced to avoid over-correction)

    adjusted_query = query.copy()

    rel_idx = np.asarray(relevant_indices, dtype=np.int64)
    irr_idx = np.asarray(irrelevant_indices, dtype=np.int64)

    # If there are no relevant images selected, do NOT apply a negative
    # subtraction — subtracting a negative centroid without a positive
    # anchor can destroy the query direction. Just return the original
    # (clamped + normalized) query so the caller can re-search and
    # optionally filter out the irrelevant paths from results.
    if rel_idx.size == 0:
        # clamp then normalize using faiss for consistency with the index
        out = np.clip(adjusted_query, -1.0, 1.0).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(out)
        return out.reshape(-1).astype(np.float64)

    # There is at least one relevant image: compute positive centroid
    pos_centroid = embeddings[rel_idx].mean(axis=0)
    adjusted_query = adjusted_query + alpha * pos_centroid

    # Apply negative penalty only when there are negative examples
    if irr_idx.size > 0:
        neg_centroid = embeddings[irr_idx].mean(axis=0)
        adjusted_query = adjusted_query - beta * neg_centroid

    # Clamp to reasonable range and re-normalize before returning
    adjusted_query = np.clip(adjusted_query, -1.0, 1.0).astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(adjusted_query)
    return adjusted_query.reshape(-1).astype(np.float64)


def rerank(
    query_embedding,
    result_embeddings,
    result_paths,
    relevant_indices,
    irrelevant_indices,
):
    """Convenience wrapper around rerank_by_feedback."""
    return rerank_by_feedback(
        query_embedding=query_embedding,
        result_embeddings=result_embeddings,
        result_paths=result_paths,
        relevant_indices=relevant_indices,
        irrelevant_indices=irrelevant_indices,
    )
