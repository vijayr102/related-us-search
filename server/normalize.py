"""Utilities for normalizing query text and MongoDB records."""
from __future__ import annotations

import re
from typing import Any, Dict

from bson import ObjectId


def normalize_query_text(text: str) -> str:
    """Normalize free-text queries by flattening whitespace and common phrases."""
    if not text:
        return ""
    cleaned = text.replace("\t", " ").replace("•", " ")
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"[\r\n]+", " ", cleaned)
    cleaned = re.sub(r'["“”]+', " ", cleaned)
    cleaned = re.sub(r"Acceptance\s+Criteria['’]s?", "Acceptance Criteria", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Acceptance\s+Criteria[:]+", "Acceptance Criteria", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_acceptance_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize acceptance-criteria-like fields within a metadata mapping."""
    if not isinstance(metadata, dict):
        return metadata
    for key in ("acceptanceCriteria", "acceptance_criteria", "acceptance criteria"):
        value = metadata.get(key)
        if isinstance(value, str):
            metadata[key] = normalize_query_text(value)
    return metadata


def sanitize_metadata(obj: Any) -> Any:
    """Recursively convert Mongo-specific types to JSON-safe forms while dropping embeddings."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "embedding":
                continue
            out[key] = sanitize_metadata(value)
        return out
    if isinstance(obj, list):
        return [sanitize_metadata(item) for item in obj]
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return str(obj)
    except Exception:  # pragma: no cover - defensive
        return None


__all__ = [
    "normalize_query_text",
    "normalize_acceptance_metadata",
    "sanitize_metadata",
]
