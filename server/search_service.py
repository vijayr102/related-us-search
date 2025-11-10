"""Search service module providing orchestration across normalization, BM25, vector, and rerank helpers."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .dedup import identifier_for_doc, prepare_document
from .normalize import (
    normalize_query_text as _normalize_query_text,
    normalize_acceptance_metadata as _normalize_acceptance_metadata_impl,
    sanitize_metadata as _sanitize_metadata_impl,
)
from .rerank import groq_available as _groq_available_impl, groq_rerank as _groq_rerank_impl, normalize_scores
from .vector_search import get_embedding as _get_embedding_impl, search_vector as _search_vector_impl
from .Bm25 import search_with_atlas_pipeline as _search_with_atlas_pipeline_impl
from .logging_utils import (
    clear_request_context,
    log_stage,
    new_request_id,
    set_request_context,
)


logger = logging.getLogger("uvicorn.error")


def normalize_query_text(text: str) -> str:
    return _normalize_query_text(text)


def _normalize_acceptance_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_acceptance_metadata_impl(metadata)


def _sanitize_metadata(obj: Any) -> Any:
    return _sanitize_metadata_impl(obj)


def _groq_available() -> bool:
    return _groq_available_impl()


def _groq_rerank(query: str, docs: List[Dict[str, Any]], top_k: int, group: str) -> List[Dict[str, Any]]:
    return _groq_rerank_impl(query, docs, top_k, group)


def _norm_scores(results: List[Dict[str, Any]]) -> None:
    normalize_scores(results)


def validate_limit(limit: Optional[int]) -> int:
    if limit is None:
        return settings.default_limit
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if limit > settings.max_limit:
        raise ValueError(f"limit must be <= {settings.max_limit}")
    return limit


async def search_with_langchain(query: str, limit: int) -> List[Dict[str, Any]]:
    """Optional LangChain integration. Currently raises so callers fall back to Atlas automatically."""
    raise RuntimeError("LangChain retriever not configured")


async def search_with_atlas_pipeline(query: str, limit: int) -> Tuple[List[Dict[str, Any]], int]:
    return await _search_with_atlas_pipeline_impl(query, limit)


async def search_vector(query: str, k: int) -> Tuple[List[Dict[str, Any]], int, str]:
    return await _search_vector_impl(query, k)


async def hybrid_search(query: str, limit: int, bm25_ratio: float = 0.5) -> Dict[str, Any]:
    """Perform hybrid search with double-fetch, Groq reranking, and deduplication."""
    raw_query = query
    request_id = new_request_id()
    set_request_context(request_id, raw_query, limit)
    try:
        query = normalize_query_text(query)
        if limit <= 0:
            return {"results": [], "total_count": 0, "params": {"query": query, "limit": limit}, "timings": {}}

        desired_bm25 = int(round(limit * bm25_ratio))
        desired_bm25 = max(0, min(desired_bm25, limit))
        desired_vector = max(0, limit - desired_bm25)

        fetch_bm25 = desired_bm25 * 2 if desired_bm25 > 0 else 0
        fetch_vector = desired_vector * 2 if desired_vector > 0 else 0

        timings: Dict[str, float] = {}

        async def _timed(label: str, coro):
            start = time.perf_counter()
            result = await coro
            timings[f"{label}_ms"] = (time.perf_counter() - start) * 1000
            return result

        tasks: List[asyncio.Task] = []
        labels: List[str] = []
        if fetch_bm25 > 0:
            tasks.append(asyncio.create_task(_timed("bm25", search_with_atlas_pipeline(query, fetch_bm25))))
            labels.append("bm25")
        if fetch_vector > 0:
            tasks.append(asyncio.create_task(_timed("vector", search_vector(query, fetch_vector))))
            labels.append("vector")

        fetch_start = time.perf_counter()
        results: List[Any] = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        fetch_end = time.perf_counter()

        bm25_docs: List[Dict[str, Any]] = []
        bm25_total = 0
        vec_docs: List[Dict[str, Any]] = []
        vec_total = 0
        vec_operator: Optional[str] = None

        for idx, result in enumerate(results):
            label = labels[idx]
            if isinstance(result, Exception):
                logger.warning("Hybrid search sub-task failed (%s): %s", label, result)
                continue
            if label == "bm25" and isinstance(result, tuple) and len(result) == 2:
                bm25_docs, bm25_total = result
            elif label == "vector" and isinstance(result, tuple):
                if len(result) == 3:
                    vec_docs, vec_total, vec_operator = result
                elif len(result) == 2:
                    vec_docs, vec_total = result

        docs_combined = list(bm25_docs) + list(vec_docs)
        docs_duration_ms = (fetch_end - fetch_start) * 1000 if tasks else 0.0
        log_stage("docs_fetched", docs_combined, duration_ms=docs_duration_ms)
        log_stage("bm25", bm25_docs, duration_ms=timings.get("bm25_ms"))
        log_stage("vector", vec_docs, duration_ms=timings.get("vector_ms"))

        norm_start = time.perf_counter()
        _norm_scores(bm25_docs)
        _norm_scores(vec_docs)
        timings["normalize_ms"] = (time.perf_counter() - norm_start) * 1000

        dedup_start = time.perf_counter()
        seen_ids = set()
        bm25_unique: List[Dict[str, Any]] = []
        for doc in bm25_docs:
            identifier = identifier_for_doc(doc)
            if identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            bm25_unique.append(doc)

        vector_unique: List[Dict[str, Any]] = []
        for doc in vec_docs:
            identifier = identifier_for_doc(doc)
            if identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            vector_unique.append(doc)

        dedup_duration = (time.perf_counter() - dedup_start) * 1000
        timings["dedup_ms"] = dedup_duration
        log_stage("dedup", bm25_unique + vector_unique, duration_ms=dedup_duration)

        rerank_start = time.perf_counter()
        bm25_final = _groq_rerank(query, bm25_unique, desired_bm25, "bm25") if desired_bm25 > 0 else []
        vector_final = _groq_rerank(query, vector_unique, desired_vector, "vector") if desired_vector > 0 else []
        timings["groq_ms"] = (time.perf_counter() - rerank_start) * 1000

        def _prepare(doc: Dict[str, Any], source: str) -> Dict[str, Any]:
            prepared = prepare_document(doc, source)
            prepared["final_score"] = float(prepared.get("groq_score", prepared.get("score", 0.0)))
            return prepared

        combined = [_prepare(doc, "bm25") for doc in bm25_final] + [_prepare(doc, "vector") for doc in vector_final]
        combined.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        log_stage("rerank", combined, duration_ms=timings.get("groq_ms"))

        timings.setdefault("bm25_ms", 0.0)
        timings.setdefault("vector_ms", 0.0)
        timings.setdefault("normalize_ms", 0.0)
        timings.setdefault("groq_ms", 0.0)
        timings.setdefault("dedup_ms", 0.0)
        timings["total_ms"] = (
            timings.get("bm25_ms", 0.0)
            + timings.get("vector_ms", 0.0)
            + timings.get("normalize_ms", 0.0)
            + timings.get("groq_ms", 0.0)
            + timings.get("dedup_ms", 0.0)
        )

        logger.info(
            "hybrid summary: query=%s bm25_final=%d vector_final=%d timings(ms)=%s",
            query,
            len(bm25_final),
            len(vector_final),
            {k: round(v, 1) for k, v in timings.items()},
        )

        return {
            "results": combined[:limit],
            "total_count": max(bm25_total, vec_total),
            "params": {
                "query": query,
                "limit": limit,
                "bm25_ratio": bm25_ratio,
                "bm25_final": len(bm25_final),
                "vector_final": len(vector_final),
                "bm25_fetch": fetch_bm25,
                "vector_fetch": fetch_vector,
                "vector_operator": vec_operator,
                "groq_model": settings.groq_model if _groq_available() else None,
            },
            "timings": timings,
        }
    finally:
        clear_request_context()


def _get_embedding(text: str) -> List[float]:
    return _get_embedding_impl(text)


__all__ = [
    "normalize_query_text",
    "_normalize_acceptance_metadata",
    "_groq_available",
    "_groq_rerank",
    "_sanitize_metadata",
    "validate_limit",
    "search_with_langchain",
    "search_with_atlas_pipeline",
    "search_vector",
    "hybrid_search",
    "_get_embedding",
]
