import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.services.retrieval import retriever
from backend.app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("secure_rag")

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SecureRAG with fixed Jio Platforms Limited IPO document...")
    try:
        if not retriever.vector_store.is_ready:
            logger.info("Building FAISS index on startup...")
            retriever.rebuild_fixed_document_index()
        logger.info(
            f"SecureRAG ready with {len(retriever.vector_store.chunks)} indexed chunks."
        )
    except Exception as e:
        logger.error(f"Startup initialization warning: {e}")
    yield
    logger.info("Shutting down SecureRAG.")


app = FastAPI(
    title="SecureRAG API",
    description="Fixed Document RAG with Guardrails and Evaluation for Jio Platforms Limited IPO",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "SecureRAG API",
        "fixed_document": settings.document_title,
        "is_indexed": retriever.vector_store.is_ready,
        "chunks_indexed": len(retriever.vector_store.chunks),
        "embedding_model": settings.embedding_model,
        "gemini_model": settings.gemini_model,
    }
