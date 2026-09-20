"""Loads the embedding model, FAISS index, and classifier exactly once, at
process startup, and hands out the cached instances to request handlers.

query_similar_images.py was written as a CLI script that reloads VGG16 on
every invocation -- fine for a one-shot script, far too slow per HTTP
request. api/main.py calls extract_feature() with the model cached here
instead.
"""
from pathlib import Path

import faiss
import numpy as np
from joblib import load as joblib_load

from src.config import CLASSIFIER_FILE, EMBEDDINGS_FILE, FAISS_INDEX_FILE, NAMES_FILE, RAW_DIR
from src.query_similar_images import load_model as load_image_model


class ModelStore:
    """Populated once by `load_all()` at FastAPI startup; read-only after that."""

    def __init__(self):
        self.image_model = None
        self.image_index = None
        self.image_names = None
        self.classifier = None

    @property
    def image_search_ready(self) -> bool:
        return self.image_index is not None

    @property
    def classifier_ready(self) -> bool:
        return self.classifier is not None

    def load_all(self):
        if EMBEDDINGS_FILE.exists() and NAMES_FILE.exists() and FAISS_INDEX_FILE.exists():
            self.image_model = load_image_model()
            self.image_index = faiss.read_index(str(FAISS_INDEX_FILE))
            self.image_names = np.load(NAMES_FILE, allow_pickle=True)

        if Path(CLASSIFIER_FILE).exists():
            self.classifier = joblib_load(CLASSIFIER_FILE)

    def image_path(self, name: str) -> Path:
        return RAW_DIR / name


store = ModelStore()
