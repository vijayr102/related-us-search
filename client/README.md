# Hybrid Search Client

Minimal React + TypeScript UI for exercising the `/hybrid_search` API. The client is scaffolded with Vite and lives entirely inside `client/`.

## Prerequisites

- Node.js 18.x (or 16.x LTS) with npm
- Server running locally on `http://localhost:8000` (default). Override via `VITE_API_BASE`.

## Setup (PowerShell)

```powershell
# From the repository root
cd client

# Install dependencies
npm install

# Optional: set API base (defaults to http://localhost:8000)
# $env:VITE_API_BASE = "http://localhost:8000"
```

## Local development

```powershell
# From client/
npm run dev
```

The development server starts on `http://localhost:5173`. Open the URL in a browser and use the UI to submit hybrid searches.

## Manual acceptance checklist

1. Ensure the Python server is running on port 8000 (see `server/README.md`).
2. Start the client with `npm run dev` and open the URL printed in the console.
3. Paste a user story, adjust the BM25 ratio slider (vector ratio updates automatically), set a limit, and click **Search**.
4. Confirm the results table renders rows with expected columns (Rank, storyId, content, acceptanceCriteria, Priority, risk, FinalScore).
5. Inspect the params/timings panel to verify it matches the server response.
6. Trigger an error by stopping the server and re-running **Search**; an error banner should appear.
