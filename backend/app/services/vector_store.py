import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.info("FAISS not found in environment; using pure NumPy vector engine.")


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Dict[str, Any]] = []

    @property
    def is_ready(self) -> bool:
        return len(self.chunks) > 0

    def build(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Build vector index from normalized embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        dim = embeddings.shape[1]
        self.dimension = dim
        self.embeddings = embeddings.astype("float32")
        self.chunks = chunks

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors.")
        else:
            logger.info(f"Built in-memory NumPy index with {len(chunks)} vectors.")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search top-k most similar chunks for a query vector."""
        if not self.is_ready:
            raise RuntimeError("Vector store is not initialized or empty.")

        k = min(top_k, len(self.chunks))

        if HAS_FAISS and self.index is not None:
            query_vec = np.ascontiguousarray(
                query_embedding.reshape(1, -1).astype("float32")
            )
            faiss.normalize_L2(query_vec)
            scores, indices = self.index.search(query_vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.chunks):
                    continue
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)
            return results

        # Fallback pure NumPy cosine inner product
        if self.embeddings is not None and len(self.embeddings) > 0:
            q_norm = np.linalg.norm(query_embedding)
            q_vec = query_embedding / (q_norm if q_norm > 0 else 1.0)
            scores = np.dot(self.embeddings, q_vec)
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_indices:
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(scores[idx])
                results.append(chunk)
            return results

        # Default: return initial chunks
        return [dict(c, score=1.0) for c in self.chunks[:k]]

    def save(self, directory: Path) -> None:
        """Persist index and chunk metadata to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        chunks_file = directory / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        if HAS_FAISS and self.index is not None:
            index_file = directory / "index.faiss"
            faiss.write_index(self.index, str(index_file))

        logger.info(f"Saved vector store with {len(self.chunks)} chunks to {directory}")

    def load(self, directory: Path) -> bool:
        """Load chunk metadata (and FAISS index if available)."""
        directory = Path(directory)
        chunks_file = directory / "chunks.json"

        if not chunks_file.exists():
            return False

        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            index_file = directory / "index.faiss"
            if HAS_FAISS and index_file.exists():
                try:
                    self.index = faiss.read_index(str(index_file))
                    self.dimension = self.index.d
                except Exception as e:
                    logger.warning(f"Could not load FAISS binary: {e}")

            logger.info(f"Loaded vector store with {len(self.chunks)} chunks.")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store from {directory}: {e}")
            return False

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return list(self.chunks)
