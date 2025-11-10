import { ChangeEvent, FormEvent, useMemo, useState } from "react";

type HybridResult = {
  content: string;
  final_score?: number;
  score?: number;
  metadata?: Record<string, any>;
};

type HybridResponse = {
  results: HybridResult[];
  total_count: number;
  params?: Record<string, any>;
  timings?: Record<string, number>;
};

const DEFAULT_QUERY = "As a user, I want to doctor dashboard patient history so that the related functionality works as expected.";

export default function App() {
  const [query, setQuery] = useState<string>(DEFAULT_QUERY);
  const [bm25Ratio, setBm25Ratio] = useState<number>(0.5);
  const [limit, setLimit] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [response, setResponse] = useState<HybridResponse | null>(null);

  const apiBase = useMemo(() => {
    return (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
  }, []);

  const vectorRatio = useMemo(() => {
    return Number((1 - bm25Ratio).toFixed(4));
  }, [bm25Ratio]);

  const handleBm25Slider = (value: number) => {
    const ratio = Math.min(1, Math.max(0, value / 100));
    setBm25Ratio(Number(ratio.toFixed(4)));
  };

  const handleBm25Input = (value: number) => {
    const ratio = Math.min(1, Math.max(0, value));
    setBm25Ratio(Number(ratio.toFixed(4)));
  };

  const handleVectorInput = (value: number) => {
    const ratio = 1 - Math.min(1, Math.max(0, value));
    setBm25Ratio(Number(ratio.toFixed(4)));
  };

  const handleLimitInput = (value: number) => {
    if (Number.isNaN(value)) {
      return;
    }
    setLimit(Math.min(100, Math.max(1, Math.trunc(value))));
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const payload = {
        query,
        limit,
      };
      const ratioParam = bm25Ratio.toFixed(4);
      const res = await fetch(`${apiBase}/hybrid_search?bm25_ratio=${ratioParam}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || `Request failed with status ${res.status}`);
      }

      const data: HybridResponse = await res.json();
      setResponse(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  const results = response?.results ?? [];

  return (
    <main>
      <h1>Related User Stories Search</h1>
      <p>
        Submit a user story and adjust the ratio between BM25 and vector results.
      </p>

      <form onSubmit={handleSubmit}>
        <fieldset>
          <label htmlFor="query">UserStory</label>
          <textarea
            id="query"
            value={query}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setQuery(event.target.value)}
            placeholder="Enter the user story text..."
            required
          />
        </fieldset>

        <fieldset className="controls">
          <div>
            <label htmlFor="bm25-slider">BM25 Ratio ({Math.round(bm25Ratio * 100)}%)</label>
            <div className="slider-row">
              <input
                id="bm25-slider"
                type="range"
                min={0}
                max={100}
                value={Math.round(bm25Ratio * 100)}
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  handleBm25Slider(Number(event.target.value))
                }
              />
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={bm25Ratio}
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  handleBm25Input(Number(event.target.value))
                }
              />
            </div>
          </div>
          <div>
            <label htmlFor="vector-ratio">Vector Ratio ({Math.round(vectorRatio * 100)}%)</label>
            <div className="slider-row">
              <input
                id="vector-ratio"
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={vectorRatio}
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  handleVectorInput(Number(event.target.value))
                }
              />
              <span>Auto-calculated</span>
            </div>
          </div>
          <div>
            <label htmlFor="limit">Limit</label>
            <input
              id="limit"
              type="number"
              min={1}
              max={100}
              value={limit}
              onChange={(event: ChangeEvent<HTMLInputElement>) =>
                handleLimitInput(Number(event.target.value))
              }
            />
          </div>
        </fieldset>

        <div className="button-row">
          <button type="submit" disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
          <button
            type="button"
            onClick={() => {
              setQuery(DEFAULT_QUERY);
              setBm25Ratio(0.5);
              setLimit(10);
              setResponse(null);
              setError("");
            }}
          >
            Reset
          </button>
        </div>
      </form>

      {error && <div className="error-banner">{error}</div>}

      {!loading && results.length === 0 && !error && (
        <p>No results yet. Submit a search to see matches.</p>
      )}

      {results.length > 0 && (
        <section>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>storyId</th>
                  <th>content</th>
                  <th>acceptanceCriteria</th>
                  <th>Priority</th>
                  <th>risk</th>
                  <th>source</th>
                  <th>FinalScore</th>
                </tr>
              </thead>
              <tbody>
                {results.map((item, index) => {
                  const metadata = item.metadata ?? {};
                  const finalScore = item.final_score ?? item.score ?? 0;
                  return (
                    <tr key={`${metadata?.storyId ?? index}-${index}`}>
                      <td>{index + 1}</td>
                      <td>{metadata?.storyId ?? "—"}</td>
                      <td>{item.content ?? ""}</td>
                      <td>{metadata?.acceptanceCriteria ?? "—"}</td>
                      <td>{metadata?.priority ?? "—"}</td>
                      <td>{metadata?.risk ?? "—"}</td>
                      <td>{item.source ?? "—"}</td>
                      <td>{finalScore.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p> The request is issued to
        <code> {apiBase}/hybrid_search</code> with the parameters you choose.</p>
        </section>
      )}
    </main>
  );
}
