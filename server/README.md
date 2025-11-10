# Hybrid Search Server

Python FastAPI service that exposes the `/hybrid_search` endpoint plus supporting health and diagnostic routes. The implementation lives inside the `server/` package; top-level modules remain as thin compatibility shims so existing tooling can still import `config`, `search_service`, etc.

## Prerequisites

- Python 3.10 or newer (tested with 3.11)
- A running MongoDB Atlas cluster with the expected collections and indexes
- Required environment variables set (see `.env` in the repository root)

## Setup (PowerShell)

```powershell
# From the repository root
cd server

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run migrations or diagnostics here if applicable
```

## Running the API locally

```powershell
# Option 1: launch from the repository root (recommended)
server\.venv\Scripts\python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

```

> `uvicorn server.main:app ...` only works when the working directory is the repository root. Running that command from inside `server/` causes `ModuleNotFoundError: No module named 'server'` because Python can no longer see the package one level up.

### Health check

```powershell
curl.exe -sS http://127.0.0.1:8000/health
```

A successful response will return `{ "status": "ok" }`.

## Project structure

```
server/
  Bm25.py             # BM25 retrieval helpers
  config.py           # Pydantic settings loader
  db.py               # MongoDB connection helpers
  dedup.py            # Document deduplication utilities
  logging_utils.py    # Structured logging helpers
  main.py             # FastAPI application definition
  models.py           # Pydantic request/response models
  normalize.py        # Text/metadata normalization
  rerank.py           # Groq reranking helpers
  search_service.py   # Hybrid search orchestration
  vector_search.py    # Embedding + vector search helpers
```

## Manual verification checklist

1. Start the server as shown above.
2. Hit `http://127.0.0.1:8000/health` and confirm `{ "status": "ok" }`.
3. Issue a sample hybrid search:
   ```powershell
   curl.exe -sS -X POST "http://127.0.0.1:8000/hybrid_search?bm25_ratio=0.5" ^
     -H "Content-Type: application/json" ^
     -d '{"query":"As a user...","limit":10}'
   ```
4. Confirm a JSON payload with `results`, `total_count`, `params`, and `timings` is returned.
