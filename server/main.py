import asyncio
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .db import connect_to_mongo, close_mongo_connection, get_collection
from .models import SearchRequest, SearchResponse, SearchResult
from .search_service import (
        validate_limit,
        search_with_langchain,
        search_with_atlas_pipeline,
        hybrid_search,
        _get_embedding,
    )

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="BM25 Search API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await connect_to_mongo(app)
    logger.info("Connected to MongoDB")


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection(app)
    logger.info("Closed MongoDB connection")


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        limit = validate_limit(req.limit if req.limit is not None else settings.default_limit)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)) from ve

    try:
        try:
            try:
                results = await asyncio.wait_for(
                    search_with_langchain(req.query, limit), timeout=settings.langchain_timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "LangChain retriever timed out after %.2fs, falling back to Atlas",
                    settings.langchain_timeout_seconds,
                )
                raise

            total_count = len(results)
            return SearchResponse(results=[SearchResult(**r) for r in results], total_count=total_count)
        except Exception as exc:  # LangChain fallback
            logger.debug("LangChain retriever not available, timed out or failed: %s", exc)
            docs, total = await search_with_atlas_pipeline(req.query, limit)
            return SearchResponse(results=[SearchResult(**d) for d in docs], total_count=total)
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Internal search error")


@app.post("/hybrid_search")
async def hybrid(req: SearchRequest, bm25_ratio: float = 0.5):
    try:
        limit = validate_limit(req.limit if req.limit is not None else settings.default_limit)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)) from ve

    if not 0.0 <= bm25_ratio <= 1.0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="bm25_ratio must be between 0.0 and 1.0")

    try:
        res = await hybrid_search(req.query, limit, bm25_ratio)
        return res
    except Exception:
        logger.exception("Hybrid search failed")
        raise HTTPException(status_code=500, detail="Hybrid search error")


@app.get("/health")
async def health():
    try:
        coll = get_collection()
        await coll.database.list_collection_names()
        return JSONResponse({"status": "ok"})
    except Exception:
        raise HTTPException(status_code=503, detail="MongoDB unreachable")


@app.get("/embedding_test")
async def embedding_test(text: str = "hello world"):
    try:
        loop = asyncio.get_running_loop()
        t0 = asyncio.get_event_loop().time()
        vec = await loop.run_in_executor(None, _get_embedding, text)
        t1 = asyncio.get_event_loop().time()
        return JSONResponse({
            "ok": True,
            "model": settings.embedding_model,
            "base": settings.embedding_api_base,
            "length": len(vec) if isinstance(vec, list) else None,
            "timing_ms": (t1 - t0) * 1000,
        })
    except Exception as exc:
        logger.exception("embedding_test failed")
        raise HTTPException(status_code=500, detail=f"embedding_test error: {exc}")
