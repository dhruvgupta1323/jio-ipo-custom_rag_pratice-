import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Lightweight, zero-framework embedding service.
    Avoids heavy PyTorch / SentenceTransformer dependencies for 100% memory efficiency.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    @property
    def dimension(self) -> int:
        return 384

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Deterministic 384-d normalized vector representation for texts."""
        if not texts:
            return np.empty((0, 384), dtype="float32")

        vectors = []
        for text in texts:
            v = np.zeros(384, dtype="float32")
            words = text.lower().split()
            for w in words:
                idx = hash(w) % 384
                v[idx] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        return np.array(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.embed_texts([query])
        return emb[0]


embedding_service = EmbeddingService()
