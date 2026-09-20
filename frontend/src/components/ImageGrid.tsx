import { imageUrl } from "../lib/api";
import "./ImageGrid.css";

interface GridItem {
  name: string;
  url: string;
  score?: number;
}

export function ImageGrid({ items }: { items: GridItem[] }) {
  if (items.length === 0) {
    return <p className="muted">No results.</p>;
  }
  return (
    <div className="image-grid">
      {items.map((item) => (
        <figure key={item.name} className="image-grid-item">
          <img src={imageUrl(item.url)} alt={item.name} loading="lazy" />
          <figcaption>
            <span className="name" title={item.name}>
              {item.name}
            </span>
            {item.score !== undefined && (
              <span className="score">{(item.score * 100).toFixed(0)}%</span>
            )}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
