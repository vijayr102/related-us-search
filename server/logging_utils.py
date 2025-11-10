"""Logging helpers and request context for structured stage instrumentation."""
from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import re
import uuid
from typing import Any, Dict, Iterable, List, Tuple

from .dedup import identifier_for_doc

_logger = logging.getLogger("uvicorn.error")

_request_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("request_context", default={})

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b")


def new_request_id() -> str:
    return str(uuid.uuid4())


def set_request_context(request_id: str, query: str, limit: int) -> None:
    _request_context.set({"request_id": request_id, "query": query, "limit": limit})


def clear_request_context() -> None:
    _request_context.set({})


def get_request_context() -> Dict[str, Any]:
    return _request_context.get() or {}


def _redact_query(query: str) -> Tuple[str, bool]:
    if not query:
        return "", False
    if _EMAIL_RE.search(query) or _PHONE_RE.search(query):
        truncated = (query[:50] + "…") if len(query) > 50 else query
        return f"[REDACTED] {truncated}", True
    if len(query) > 200:
        return query[:200] + "…", False
    return query, False


def _extract_top_info(docs: Iterable[Dict[str, Any]], max_items: int = 10) -> Tuple[List[str], List[float]]:
    ids: List[str] = []
    scores: List[float] = []
    for doc in docs:
        if len(ids) >= max_items:
            break
        doc_id = identifier_for_doc(doc)
        ids.append(doc_id)
        score = None
        for key in ("final_score", "groq_score", "score"):
            val = doc.get(key)
            if isinstance(val, (int, float)):
                score = float(val)
                break
        scores.append(score if score is not None else None)
    return ids, scores


def log_stage(stage: str, docs: Iterable[Dict[str, Any]], *, duration_ms: float | None = None, note: str | None = None) -> None:
    ctx = get_request_context()
    request_id = ctx.get("request_id") or new_request_id()
    query = ctx.get("query", "")
    limit = ctx.get("limit")

    display_query, redacted = _redact_query(query)
    docs_list = list(docs)
    count = len(docs_list)
    top_ids, top_scores = _extract_top_info(docs_list)

    entry: Dict[str, Any] = {
        "timestamp": _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "request_id": request_id,
        "stage": stage,
        "query": display_query,
        "limit": limit,
        "count": count,
        "top_ids": top_ids,
        "top_scores": top_scores,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
    }
    if note:
        entry["note"] = note
    if redacted:
        entry["redacted"] = True

    message = json.dumps(entry, default=str)
    _logger.info("%s", message)

    if limit is not None and count < limit:
        warn_entry = dict(entry)
        warn_entry.setdefault("note", "results < limit")
        warn_message = json.dumps(warn_entry, default=str)
        _logger.warning("%s", warn_message)


__all__ = [
    "new_request_id",
    "set_request_context",
    "clear_request_context",
    "get_request_context",
    "log_stage",
]
