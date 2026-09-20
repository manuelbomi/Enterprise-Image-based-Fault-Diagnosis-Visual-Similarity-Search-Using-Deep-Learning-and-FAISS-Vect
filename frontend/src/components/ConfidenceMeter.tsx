import "./ConfidenceMeter.css";

interface ConfidenceMeterProps {
  label: string;
  confidence: number;
}

export function ConfidenceMeter({ label, confidence }: ConfidenceMeterProps) {
  const pct = Math.round(confidence * 100);
  const severity = confidence >= 0.66 ? "good" : confidence >= 0.4 ? "warning" : "critical";

  return (
    <div className="confidence-meter">
      <div className="confidence-meter-label">
        <span className="predicted-label">{label}</span>
        <span className="predicted-pct">{pct}% confidence</span>
      </div>
      <div className="confidence-meter-track">
        <div className={`confidence-meter-fill ${severity}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
