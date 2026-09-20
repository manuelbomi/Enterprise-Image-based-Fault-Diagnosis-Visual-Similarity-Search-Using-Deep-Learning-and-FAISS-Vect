# API

A thin FastAPI layer over the existing `src/query_similar_images.py` pipeline
and the optional `src/train_classifier.py` classifier — the "Deploy as REST
API" item from the main README's Enterprise Expansion Ideas table. It
doesn't reimplement any vision logic; it loads the same VGG16 model, FAISS
index, and classifier those CLI scripts use once at startup
(`api/models.py`) and calls the same functions per request.

## Prerequisites

Build the embeddings, index, and (optionally) the classifier first — from
the repo root, see the main [README's Quickstart](../README.md#quickstart):

```bash
python generate_sample_dataset.py   # or supply your own images in data/raw/
python -m src.extract_embeddings
python -m src.build_faiss_index
python -m src.train_classifier      # optional -- needs data/processed/labels.csv
```

## Running

From the repo root (so `data/`, `src/`, and `api/` all resolve):

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | — | Whether the image index and classifier are loaded |
| POST | `/api/search?top_k=5` | multipart `file` (jpg/png) | Top-`k` visually similar images from `data/raw/`, with similarity scores |
| POST | `/api/classify?top_k=5` | multipart `file` (jpg/png) | Classifier prediction (if a classifier was trained) plus the same similar-images list as `/api/search` |

Images referenced in responses are served at `/images/<filename>` (mounted
from `data/raw/`). If no classifier was trained, `/api/classify` still
returns the similar-images list, with `predicted_label`/`predicted_confidence`
as `null`.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated origins allowed to call this API (the frontend's dev server by default) |

## A note on ports

Port 8000 is a common default and can already be in use by other local
services (Docker Desktop, other dev servers, etc.). If `/api/health`
requests from the frontend fail with a CORS or connection error but `curl
http://127.0.0.1:8000/api/health` works, something else is likely also
listening on that port — pick a different one (`--port 8010`, etc.) and
update `frontend/.env.local`'s `VITE_API_BASE_URL` to match.
