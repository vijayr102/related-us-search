"""BM25-based MongoDB Atlas retrieval helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import settings
from .db import get_collection
from .normalize import normalize_query_text, normalize_acceptance_metadata, sanitize_metadata


async def search_with_atlas_pipeline(query: str, limit: int) -> Tuple[List[Dict[str, Any]], int]:
    """Perform BM25 style full-text search using MongoDB Atlas $search stage."""
    query = normalize_query_text(query)
    if not query:
        return [], 0

    coll = get_collection()
    search_fields = settings.search_fields or []
    if not search_fields:
        search_fields = ["text"]

    search_stage: Dict[str, Any] = {
        "$search": {
            "index": settings.bm25_index_name,
            "text": {
                "query": query,
                "path": search_fields,
            },
        }
    }

    pipeline: List[Dict[str, Any]] = [
        search_stage,
        {"$project": {"score": {"$meta": "searchScore"}, "document": "$$ROOT"}},
        {"$limit": limit},
    ]

    docs: List[Dict[str, Any]] = []
    cursor = coll.aggregate(pipeline)
    async for doc in cursor:
        root = doc.get("document", {}) or {}
        content = root.get("content") or root.get("text") or root.get("summary") or ""
        metadata = {k: v for k, v in root.items() if k not in {"content", "text", "summary", "embedding"}}
        metadata = normalize_acceptance_metadata(sanitize_metadata(metadata))
        docs.append({
            "content": content,
            "score": float(doc.get("score", 0.0) or 0.0),
            "metadata": metadata,
        })

    return docs, len(docs)


__all__ = ["search_with_atlas_pipeline"]
