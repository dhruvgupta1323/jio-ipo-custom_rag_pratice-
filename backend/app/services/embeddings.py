import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            try:
                logger.info(f"Loading embedding model: {settings.embedding_model}")
                cls._model = SentenceTransformer(settings.embedding_model)
            except Exception as e:
                logger.error(f"Error loading embedding model {settings.embedding_model}: {e}")
                raise
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Compute normalized sentence embeddings for a list of strings.
        Normalization ensures inner product equals cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string as a 1D normalized float32 vector."""
        emb = self.embed_texts([query])
        return emb[0]


embedding_service = EmbeddingService()
