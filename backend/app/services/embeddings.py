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
            cls._model = None
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass
            logger.info(f"Lazy-loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is not None:
            if hasattr(self._model, "get_embedding_dimension"):
                return self._model.get_embedding_dimension()
            return self._model.get_sentence_embedding_dimension()
        return 384

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
