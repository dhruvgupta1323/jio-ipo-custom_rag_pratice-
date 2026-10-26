import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0 and len(self.chunks) > 0

    def build(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Build FAISS IndexFlatIP from normalized embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        dim = embeddings.shape[1]
        self.dimension = dim
        # IndexFlatIP computes exact inner product (equivalent to cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))
        self.chunks = chunks
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors of dimension {dim}.")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search top-k most similar chunks for a query vector."""
        if not self.is_ready:
            raise RuntimeError("FAISS index is not initialized or empty.")

        query_vec = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype("float32")
        )
        # Ensure query vector is normalized
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def save(self, directory: Path) -> None:
        """Persist index and chunk metadata to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not self.is_ready:
            raise RuntimeError("Cannot save an uninitialized vector store.")

        index_file = directory / "index.faiss"
        chunks_file = directory / "chunks.json"

        faiss.write_index(self.index, str(index_file))
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index and {len(self.chunks)} chunks to {directory}")

    def load(self, directory: Path) -> bool:
        """Load index and chunk metadata from directory. Returns True on success."""
        directory = Path(directory)
        index_file = directory / "index.faiss"
        chunks_file = directory / "chunks.json"

        if not index_file.exists() or not chunks_file.exists():
            logger.info(f"Index files not found in {directory}")
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info(
                f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.chunks)} chunks."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from {directory}: {e}")
            return False

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return list(self.chunks)
