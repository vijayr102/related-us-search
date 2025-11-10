from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    limit: Optional[int] = Field(None, ge=1, le=100, description="Maximum number of results to return")


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
