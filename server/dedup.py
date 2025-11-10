"""Deduplication helpers for hybrid search results."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from .normalize import normalize_acceptance_metadata, sanitize_metadata


def identifier_for_doc(doc: Dict[str, Any]) -> str:
    metadata = doc.get("metadata", {}) or {}
    identifier = metadata.get("_id")
    if identifier:
        return str(identifier)
    content = doc.get("content") or ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def prepare_document(doc: Dict[str, Any], source: str) -> Dict[str, Any]:
    prepared = {**doc, "source": source}
    prepared.pop("norm_score", None)
    prepared.setdefault("metadata", {})
    prepared["metadata"] = normalize_acceptance_metadata(sanitize_metadata(prepared["metadata"]))
    return prepared


def deduplicate_and_trim(results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for doc in results:
        identifier = identifier_for_doc(doc)
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(doc)
        if len(deduped) == limit:
            break
    return deduped


__all__ = ["prepare_document", "deduplicate_and_trim", "identifier_for_doc"]
