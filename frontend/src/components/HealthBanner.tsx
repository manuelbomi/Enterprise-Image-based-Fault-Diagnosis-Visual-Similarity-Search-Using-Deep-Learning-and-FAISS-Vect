import type { HealthResponse } from "../types";
import "./HealthBanner.css";

interface HealthBannerProps {
  health: HealthResponse | null;
  error: string | null;
}

export function HealthBanner({ health, error }: HealthBannerProps) {
  if (error) {
    return (
      <div className="health-banner critical">
        Can't reach the API at the configured <code>VITE_API_BASE_URL</code>. Is it
        running? (<code>uvicorn api.main:app --reload --port 8000</code>) — {error}
      </div>
    );
  }

  if (!health) {
    return <div className="health-banner">Checking API status...</div>;
  }

  if (!health.image_search_ready) {
    return (
      <div className="health-banner warning">
        Image index not built yet. Run <code>python -m src.extract_embeddings</code> then{" "}
        <code>python -m src.build_faiss_index</code>, then restart the API.
      </div>
    );
  }

  if (!health.classifier_ready) {
    return (
      <div className="health-banner warning">
        No trained classifier found — Visual Search will work, but Classify won't
        show a predicted label. Run <code>python -m src.train_classifier</code>, then
        restart the API.
      </div>
    );
  }

  return null;
}
