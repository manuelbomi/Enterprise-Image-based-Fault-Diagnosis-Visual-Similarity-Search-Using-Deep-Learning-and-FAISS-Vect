from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    image_search_ready: bool
    classifier_ready: bool


class SimilarImageResult(BaseModel):
    name: str
    score: float
    url: str


class SearchResponse(BaseModel):
    query_filename: str
    results: list[SimilarImageResult]


class ClassifyResponse(BaseModel):
    query_filename: str
    predicted_label: Optional[str] = None
    predicted_confidence: Optional[float] = None
    similar_images: list[SimilarImageResult]
