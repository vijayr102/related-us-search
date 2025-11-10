# Hybrid Search Monorepo

Hybrid search demo combining a FastAPI backend with a React + TypeScript frontend. The backend orchestrates BM25, vector search, deduplication, and Groq reranking for a `/hybrid_search` API; the frontend offers a simple UI for experimenting with tuning knobs.

## Repository layout

```
.
├── server/   # FastAPI application and hybrid search pipeline
├── client/   # React + Vite single-page app to query the API
├── .env      # Local secrets (ignored); copy from .env.example
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB Atlas cluster with appropriate collections and indexes
- Groq API key (optional, for reranking)

## Quick start (PowerShell)

```powershell
# Clone and enter the repo
 git clone <your-fork-url>
 cd HybridSearch

# Copy environment template
 copy .env.example .env
 # populate .env with real values for Mongo, embeddings, etc.
```

### Backend setup

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run FastAPI (from repo root)
cd ..
server\.venv\Scripts\python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend setup

```powershell
cd client
npm install
npm run dev -- --host
```

Visit `http://localhost:5173` to load the UI. The app expects the API at `http://localhost:8000` by default; adjust `VITE_API_BASE` in `.env` if needed.

## Testing the API

```powershell
curl.exe -sS http://127.0.0.1:8000/health
curl.exe -sS -X POST "http://127.0.0.1:8000/hybrid_search?bm25_ratio=0.5" `
  -H "Content-Type: application/json" `
  -d '{"query":"Explain hybrid search","limit":5}'
```

## Development notes

- `server/main.py` includes CORS support for the local Vite dev server.
- Deduplication happens before reranking; timing metrics (including `dedup_ms`) surface in `/hybrid_search` responses.
- Root-level modules (`config.py`, `search_service.py`, etc.) remain as shims to maintain backwards compatibility for existing imports.

## License

Add licensing information here.
