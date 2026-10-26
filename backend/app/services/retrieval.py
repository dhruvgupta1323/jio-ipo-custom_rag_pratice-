import logging
import re
from typing import List, Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.services.embeddings import embedding_service
from backend.app.services.vector_store import FAISSVectorStore
from backend.app.services.ingestion import extract_pdf_from_file, chunk_pages

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "what", "are", "the", "of", "in", "to", "and", "is", "for", "with",
    "from", "this", "that", "how", "much", "did", "was", "were", "when",
    "which", "who", "whom", "whose", "why", "where", "about", "can", "could",
    "should", "would", "does", "do", "done", "tell", "give", "show", "me",
    "a", "an", "as", "by", "at", "be", "it", "its", "their", "our", "all",
    "jio", "platforms", "limited", "capstocks", "equity", "research", "ipo", "company"
}


class RAGRetriever:
    def __init__(self):
        self.vector_store = FAISSVectorStore(dimension=embedding_service.dimension)
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Attempt to load existing index or build from the fixed Jio IPO PDF."""
        index_dir = settings.index_dir_abs
        if self.vector_store.load(index_dir):
            logger.info("Fixed Jio IPO document index loaded successfully.")
            return

        logger.info("Index not found on disk. Building index for fixed Jio IPO PDF...")
        self.rebuild_fixed_document_index()

    def rebuild_fixed_document_index(self) -> int:
        """Extract, chunk, embed, and index the fixed Jio IPO PDF."""
        pdf_path = settings.source_pdf_abs
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fixed source PDF not found at {pdf_path}")

        pages = extract_pdf_from_file(pdf_path)
        chunks = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            source_file=pdf_path.name,
            document_id="jio_platforms_ipo",
        )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.embed_texts(texts)

        self.vector_store.build(chunks, embeddings)
        self.vector_store.save(settings.index_dir_abs)
        logger.info(f"Fixed document indexed with {len(chunks)} chunks.")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks using dense semantic similarity
        augmented with salient entity keyword matching for high financial precision.
        """
        if not self.vector_store.is_ready:
            self._initialize_index()

        k = top_k or settings.top_k
        min_thresh = threshold if threshold is not None else settings.relevance_threshold

        query_emb = embedding_service.embed_query(query)
        total_chunks = len(self.vector_store.chunks)
        candidates = self.vector_store.search(query_emb, top_k=min(total_chunks, 25))

        # Extract only salient terms, excluding common English stop words
        raw_terms = set(re.findall(r"\b[a-zA-Z0-9₹%-]+\b", query.lower()))
        salient_terms = [t for t in raw_terms if t not in STOP_WORDS and len(t) >= 3]

        boosted_results = []
        for cand in candidates:
            score = cand["score"]
            text_lower = cand["text"].lower()

            exact_hits = sum(1 for term in salient_terms if term in text_lower)

            # Proportional boost for salient domain terms (e.g. objects, pat, prepayment, lot, rating)
            bonus = min(0.35, exact_hits * 0.08)
            final_score = score + bonus

            cand_copy = dict(cand)
            cand_copy["score"] = round(float(final_score), 4)

            # Keep candidate if semantic score meets threshold or strong salient keyword hit
            if score >= min_thresh or exact_hits >= 1:
                boosted_results.append(cand_copy)

        # Sort descending by score
        boosted_results.sort(key=lambda x: x["score"], reverse=True)
        return boosted_results[:k]

    def get_document_info(self) -> Dict[str, Any]:
        """Return information regarding the fixed Jio IPO document and its index status."""
        pdf_path = settings.source_pdf_abs
        return {
            "document_title": settings.document_title,
            "file_name": pdf_path.name if pdf_path.exists() else "Unknown",
            "file_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
            "total_pages": 3,
            "total_chunks": len(self.vector_store.chunks),
            "is_indexed": self.vector_store.is_ready,
            "company_name": "Jio Platforms Limited",
            "issue_date": "21 Oct 26 – 23 Oct 26",
            "price_band": "₹1065 to ₹1119",
            "market_lot": 12,
            "face_value": 10,
            "issue_size": "₹30,213 Crore",
            "sector": "Mobile & Digital Services",
            "capstocks_rating": "18 / 30 (Good)",
            "recommendation": "Subscribe (Medium-to-long-term view)",
        }


retriever = RAGRetriever()
