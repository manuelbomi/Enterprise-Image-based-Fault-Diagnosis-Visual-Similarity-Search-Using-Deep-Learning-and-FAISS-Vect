import type { ClassifyResponse, HealthResponse, SearchResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json();
}

export function imageUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/health`);
  return handle(response);
}

export async function searchSimilarImages(file: File, topK = 5): Promise<SearchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/api/search?top_k=${topK}`, {
    method: "POST",
    body: formData,
  });
  return handle(response);
}

export async function classify(file: File, topK = 5): Promise<ClassifyResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/api/classify?top_k=${topK}`, {
    method: "POST",
    body: formData,
  });
  return handle(response);
}
