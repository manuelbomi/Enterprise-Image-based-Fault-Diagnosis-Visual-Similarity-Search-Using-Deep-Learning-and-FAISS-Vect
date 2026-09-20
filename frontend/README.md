# Visual Fault Similarity Search — Frontend

A React + TypeScript app for the [visual similarity search API](../api/README.md):
upload a fault image, then either run **Classify** (predicted fault label +
confidence, plus similar historical faults) or **Visual Search** (just the
similar-image lookup, no classifier involved).

## Running

The [API](../api/README.md) must be running first (from the repo root):

```bash
uvicorn api.main:app --reload --port 8000
```

Then, in this directory:

```bash
npm install
npm run dev
```

Open the URL Vite prints (typically http://localhost:5173).

## Configuring the API URL

Defaults to `http://localhost:8000`. To point at a different host/port
(e.g. if 8000 was already taken on your machine — see the note in
[`api/README.md`](../api/README.md#a-note-on-ports)), create `.env.local`:

```bash
cp .env.example .env.local
# edit VITE_API_BASE_URL in .env.local
```

Restart `npm run dev` after changing it — Vite only reads `.env*` files at
startup.

## Project layout

```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUploader.tsx    # drag-and-drop / click-to-browse file input
│   │   ├── ImageGrid.tsx         # thumbnail grid with similarity score
│   │   ├── ConfidenceMeter.tsx   # classifier prediction + confidence bar
│   │   └── HealthBanner.tsx      # surfaces "API unreachable" / "index not built yet"
│   ├── lib/api.ts                # typed fetch wrapper for /api/health, /search, /classify
│   ├── types.ts                  # mirrors api/schemas.py field-for-field
│   └── App.tsx
```

## Stack

Vite + React 19 + TypeScript. No UI framework, no state-management or
data-fetching library, no generated API client — two endpoints and a
handful of components don't need any of that.

## Building for deployment

```bash
npm run build
```

Outputs a static site to `dist/`, deployable to any static host. Set
`VITE_API_BASE_URL` at build time to point at your deployed API:

```bash
VITE_API_BASE_URL=https://your-api.example.com npm run build
```
