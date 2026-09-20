"""
generate_sample_dataset.py
============================

Synthesizes a small sample fault-image dataset (data/raw/*.jpg) and its
label file (data/processed/labels.csv), so the rest of this pipeline
(extract_embeddings.py, build_faiss_index.py, query_similar_images.py,
train_classifier.py) can be run end to end without a real customer image
dataset. Swap in your own images for a real deployment -- see README >
How to Run.

Two synthetic fault classes are generated, each as a textured metal-like
surface so VGG16 embeddings actually differ between and within classes
(unlike flat color swatches):

  - "crack":     a light gray metal surface with a few random jagged dark
                 fracture lines drawn across it.
  - "corrosion": a metal surface with overlapping rust-colored blotches of
                 varying size/opacity.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG_SIZE = 300
METAL_BASE = (176, 178, 182)


def _metal_background(rng):
    noise = rng.normal(0, 6, size=(IMG_SIZE, IMG_SIZE, 3))
    base = np.array(METAL_BASE, dtype=np.float32) + noise
    return np.clip(base, 0, 255).astype(np.uint8)


def make_crack_image(rng, n_cracks=None):
    arr = _metal_background(rng)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    n_cracks = n_cracks or rng.integers(2, 4)
    for _ in range(n_cracks):
        x, y = rng.uniform(40, IMG_SIZE - 40, size=2)
        length = rng.integers(25, 45)
        gray = int(rng.uniform(30, 70))
        width = int(rng.integers(1, 3))
        angle = rng.uniform(0, 2 * np.pi)
        for _ in range(length):
            angle += rng.uniform(-0.5, 0.5)  # jagged but continuous, not a random walk star-burst
            nx, ny = x + np.cos(angle) * 8, y + np.sin(angle) * 8
            draw.line([(x, y), (nx, ny)], fill=(gray, gray, gray), width=width)
            x, y = np.clip(nx, 0, IMG_SIZE - 1), np.clip(ny, 0, IMG_SIZE - 1)

    return img.filter(ImageFilter.GaussianBlur(0.4))


def make_corrosion_image(rng, n_blotches=None):
    arr = _metal_background(rng)
    img = Image.fromarray(arr).convert("RGBA")

    n_blotches = n_blotches or rng.integers(6, 12)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rust_palette = [(120, 62, 20), (150, 80, 30), (110, 45, 15), (170, 100, 40)]

    for _ in range(n_blotches):
        cx, cy = rng.uniform(0, IMG_SIZE, size=2)
        r = rng.uniform(10, 35)
        color = rust_palette[rng.integers(0, len(rust_palette))]
        alpha = int(rng.uniform(90, 180))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, alpha))

    overlay = overlay.filter(ImageFilter.GaussianBlur(3))
    return Image.alpha_composite(img, overlay).convert("RGB")


def generate(out_raw_dir, out_labels_csv, n_per_class=5, seed=11):
    rng = np.random.default_rng(seed)
    out_raw_dir = Path(out_raw_dir)
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    Path(out_labels_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_per_class):
        name = f"crack_{i}.jpg"
        make_crack_image(rng).save(out_raw_dir / name, quality=92)
        rows.append((name, "crack"))

    for i in range(n_per_class):
        name = f"corrosion_{i}.jpg"
        make_corrosion_image(rng).save(out_raw_dir / name, quality=92)
        rows.append((name, "corrosion"))

    with open(out_labels_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} images -> {out_raw_dir}")
    print(f"Wrote labels -> {out_labels_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--labels-csv", default="data/processed/labels.csv")
    parser.add_argument("--n-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    generate(args.raw_dir, args.labels_csv, n_per_class=args.n_per_class, seed=args.seed)
