"""Groq-based reranking and score normalization utilities."""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, List

import requests

from .config import settings
from .normalize import normalize_acceptance_metadata, sanitize_metadata


logger = logging.getLogger("uvicorn.error")


def groq_available() -> bool:
    return bool(settings.groq_api_key and settings.groq_model)


def _truncate(text: str, limit: int = 500) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def groq_rerank(query: str, docs: List[Dict[str, Any]], top_k: int, group: str) -> List[Dict[str, Any]]:
    if top_k <= 0 or not docs:
        return []

    fallback = sorted(docs, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]

    if not groq_available():
        return _enrich_with_local_scores(fallback)

    base_url = (settings.groq_api_base or "https://api.groq.com/openai/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    payload_candidates = docs[: min(len(docs), max(top_k * 2, top_k))]
    payload = {
        "model": settings.groq_model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You score search results for relevance on a scale from 0 to 1. "
                    "Respond with JSON:{\"scores\":[{\"idx\":int,\"score\":float}]}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Query: "
                    + query
                    + "\nGroup: "
                    + group
                    + "\nCandidates:\n"
                    + "\n".join(
                        f"{idx}: "
                        + _truncate(candidate.get("content"))
                        + (
                            " | metadata: "
                            + _truncate(json.dumps(candidate.get("metadata", {})))
                            if candidate.get("metadata")
                            else ""
                        )
                        for idx, candidate in enumerate(payload_candidates)
                    )
                    + "\nReturn JSON with an entry for each candidate."
                ),
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    try:
        start = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        elapsed = (time.perf_counter() - start) * 1000
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content)
        scores = parsed.get("scores") if isinstance(parsed, dict) else parsed
        score_map: Dict[int, float] = {}
        for entry in scores or []:
            idx = entry.get("idx")
            score = entry.get("score")
            if isinstance(idx, int) and isinstance(score, (int, float)):
                score_map[idx] = float(score)
        reranked: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(payload_candidates):
            groq_score = score_map.get(idx)
            if groq_score is None:
                groq_score = float(candidate.get("score", 0.0))
            copy = {**candidate}
            copy["groq_score"] = groq_score
            copy.setdefault("metadata", {})
            copy["metadata"] = normalize_acceptance_metadata(sanitize_metadata(copy["metadata"]))
            copy["metadata"]["groq_score"] = groq_score
            copy["metadata"]["groq_response_ms"] = round(elapsed, 1)
            reranked.append(copy)
        reranked.sort(key=lambda x: x.get("groq_score", x.get("score", 0.0)), reverse=True)
        return reranked[:top_k]
    except Exception as exc:  # broad to ensure fallback path
        logger.warning("Groq rerank failed, falling back to intrinsic scores: %s", exc)
        return _enrich_with_local_scores(fallback)


def _enrich_with_local_scores(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for doc in docs:
        copy = {**doc}
        copy["groq_score"] = float(doc.get("score", 0.0))
        copy.setdefault("metadata", {})
        copy["metadata"] = normalize_acceptance_metadata(sanitize_metadata(copy["metadata"]))
        copy["metadata"]["groq_score"] = copy["groq_score"]
        enriched.append(copy)
    return enriched


def normalize_scores(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    max_score = max((r.get("score", 0.0) for r in results), default=0.0)
    if max_score <= 0 or math.isclose(max_score, 0.0):
        for r in results:
            r["norm_score"] = 0.0
    else:
        for r in results:
            r["norm_score"] = float(r.get("score", 0.0)) / float(max_score)


__all__ = [
    "groq_available",
    "groq_rerank",
    "normalize_scores",
]
