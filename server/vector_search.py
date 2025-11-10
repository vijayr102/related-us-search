"""Embedding retrieval and MongoDB Atlas vector search helpers."""
from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests import HTTPError

from .config import settings
from .db import get_collection
from .normalize import normalize_query_text, normalize_acceptance_metadata, sanitize_metadata


logger = logging.getLogger("uvicorn.error")


_EMBEDDING_ENDPOINT: Optional[Tuple[str, Dict[str, str]]] = None


def _resolve_embedding_endpoint() -> Tuple[str, Dict[str, str]]:
    global _EMBEDDING_ENDPOINT
    if _EMBEDDING_ENDPOINT is not None:
        return _EMBEDDING_ENDPOINT

    base = (settings.embedding_api_base or "").strip()
    if not base:
        raise RuntimeError("EMBEDDING_API_BASE is required")
    base = base.rstrip("/")
    url = f"{base}/embeddings"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    token = settings.embedding_auth_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if settings.embedding_user_email:
        headers.setdefault("X-User-Email", settings.embedding_user_email)

    _EMBEDDING_ENDPOINT = (url, headers)
    return _EMBEDDING_ENDPOINT


@lru_cache(maxsize=256)
def get_embedding(text: str) -> List[float]:
    """Fetch an embedding vector from the configured provider, caching by normalized text."""
    normalized = normalize_query_text(text)
    if not normalized:
        raise ValueError("cannot embed empty text")

    url, headers = _resolve_embedding_endpoint()
    payload = {"input": normalized, "model": settings.embedding_model}

    for attempt in range(2):
        start = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            body = resp.json()
            data = body.get("data") if isinstance(body, dict) else None
            if isinstance(data, list) and data:
                embedding = data[0].get("embedding")
                if isinstance(embedding, list):
                    logger.info("Embedding call ok url=%s ms=%.1f", url, elapsed)
                    print(f"[embedding] using url={url} header=bearer ({elapsed:.1f} ms)")
                    return embedding
            if isinstance(body, dict) and isinstance(body.get("embedding"), list):
                logger.info("Embedding call ok url=%s ms=%.1f (top-level)", url, elapsed)
                print(f"[embedding] using url={url} header=bearer (top-level, {elapsed:.1f} ms)")
                return body["embedding"]
            if isinstance(body, list):
                logger.info("Embedding call ok url=%s ms=%.1f (list)", url, elapsed)
                print(f"[embedding] using url={url} header=bearer (list, {elapsed:.1f} ms)")
                return body
            raise RuntimeError(f"Unexpected embedding response format from {url}: {type(body)}")
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 and attempt == 0:
                time.sleep(1.0)
                continue
        raise

    raise RuntimeError("Embedding request failed after retries")


async def search_vector(query: str, k: int) -> Tuple[List[Dict[str, Any]], int, str]:
    """Perform vector search using MongoDB Atlas $vectorSearch operator."""
    query = normalize_query_text(query)
    if k <= 0 or not query:
        return [], 0, "$vectorSearch"

    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(None, get_embedding, query)

    coll = get_collection()
    stage = {
        "$vectorSearch": {
            "index": settings.vector_index_name,
            "path": "embedding",
            "queryVector": embedding,
            "numCandidates": max(100, k * 10),
            "limit": k,
        }
    }
    pipeline = [stage, {"$project": {"score": {"$meta": "vectorSearchScore"}, "document": "$$ROOT"}}]

    docs: List[Dict[str, Any]] = []
    try:
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
        logger.info("vector search operator=$vectorSearch returned %d", len(docs))
        print(f"[vector-search] operator=$vectorSearch results={len(docs)}")
        return docs, len(docs), "$vectorSearch"
    except Exception as exc:
        logger.error("$vectorSearch failed: %s", exc)
        print(f"[vector-search] operator=$vectorSearch failed: {exc}")
        raise


__all__ = [
    "get_embedding",
    "search_vector",
]
