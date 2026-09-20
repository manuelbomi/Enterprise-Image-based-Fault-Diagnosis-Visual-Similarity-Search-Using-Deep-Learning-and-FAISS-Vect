import { useEffect, useState } from "react";
import { ImageUploader } from "./components/ImageUploader";
import { ImageGrid } from "./components/ImageGrid";
import { ConfidenceMeter } from "./components/ConfidenceMeter";
import { HealthBanner } from "./components/HealthBanner";
import { classify, fetchHealth, searchSimilarImages } from "./lib/api";
import type { ClassifyResponse, HealthResponse, SearchResponse } from "./types";
import "./App.css";

type Tab = "classify" | "search";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("classify");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [classifyResult, setClassifyResult] = useState<ClassifyResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err: Error) => setHealthError(err.message));
  }, []);

  async function run() {
    if (!file) return;
    setLoading(true);
    setRunError(null);
    try {
      if (tab === "search") {
        setSearchResult(await searchSimilarImages(file, 5));
      } else {
        setClassifyResult(await classify(file, 5));
      }
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <h1>Visual Fault Similarity Search</h1>
        <p className="subtitle">
          Upload a fault image to classify it and find visually similar historical
          faults using VGG16 embeddings + FAISS. React/TypeScript frontend for{" "}
          <a
            href="https://github.com/manuelbomi/Image-based-Fault-Diagnosis-Visual-Similarity-Search-Using-Deep-Learning-and-FAISS-Vector-Indexing"
            target="_blank"
            rel="noreferrer"
          >
            Image-based-Fault-Diagnosis-Visual-Similarity-Search-Using-Deep-Learning-and-FAISS-Vector-Indexing
          </a>
          .
        </p>
      </header>

      <HealthBanner health={health} error={healthError} />

      <section className="panel">
        <ImageUploader onFileSelected={setFile} disabled={loading} />
      </section>

      <section className="panel">
        <div className="tabs" role="tablist">
          <button
            role="tab"
            aria-selected={tab === "classify"}
            className={tab === "classify" ? "tab active" : "tab"}
            onClick={() => setTab("classify")}
          >
            Classify
          </button>
          <button
            role="tab"
            aria-selected={tab === "search"}
            className={tab === "search" ? "tab active" : "tab"}
            onClick={() => setTab("search")}
          >
            Visual Search
          </button>
        </div>

        <button className="run-button" onClick={run} disabled={!file || loading}>
          {loading ? "Running..." : tab === "classify" ? "Classify fault" : "Find similar images"}
        </button>

        {runError && <p className="error">{runError}</p>}

        {tab === "search" && searchResult && (
          <div className="results">
            <h3>Similar images to {searchResult.query_filename}</h3>
            <ImageGrid items={searchResult.results} />
          </div>
        )}

        {tab === "classify" && classifyResult && (
          <div className="results">
            {classifyResult.predicted_label && classifyResult.predicted_confidence !== null ? (
              <ConfidenceMeter
                label={classifyResult.predicted_label}
                confidence={classifyResult.predicted_confidence}
              />
            ) : (
              <p className="muted">
                No trained classifier available — run{" "}
                <code>python -m src.train_classifier</code> on the API server to enable
                predicted labels.
              </p>
            )}

            <h3>Similar historical faults</h3>
            <ImageGrid items={classifyResult.similar_images} />
          </div>
        )}
      </section>

      <footer className="page-footer">
        <p>
          Backend: FastAPI wrapping <code>src/query_similar_images.py</code> and the{" "}
          <code>src/train_classifier.py</code> classifier. See <code>api/README.md</code>{" "}
          and <code>frontend/README.md</code>.
        </p>
      </footer>
    </div>
  );
}
