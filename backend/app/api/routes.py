import time
import logging
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, Query, status

from backend.app.core.config import settings
from backend.app.core.security import sanitize_input
from backend.app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentInfo,
    GuardrailAudit,
    RetrievedChunk,
    ChunksResponse,
    ChunkItem,
    EvaluationRunResponse,
)
from backend.app.services.ingestion import extract_pdf, chunk_pages, extract_pdf_from_file
from backend.app.services.retrieval import retriever
from backend.app.services.generation import generator
from backend.app.services.guardrails import check_input, check_output
from backend.app.services.evaluation import run_evaluation_suite

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["SecureRAG"])

MAX_FILE_SIZE = 10 * 1024 * 1024


@router.get("/document-info", response_model=DocumentInfo)
def get_document_info():
    """Get metadata and index status for the fixed Jio Platforms Limited IPO document."""
    info = retriever.get_document_info()
    return DocumentInfo(**info)


@router.get("/document/pages")
def get_document_pages():
    """Retrieve full text and metadata for each page of the fixed Jio IPO report."""
    pdf_path = settings.source_pdf_abs
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source PDF file not found at {pdf_path}",
        )
    pages = extract_pdf_from_file(pdf_path)
    return {
        "title": settings.document_title,
        "total_pages": len(pages),
        "pages": [
            {
                "page": p["page"],
                "text": p["text"],
                "char_count": len(p["text"]),
            }
            for p in pages
        ],
    }


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Query the fixed Jio Platforms IPO document with input/output guardrails,
    dense vector retrieval, and grounded answer generation with citations.
    """
    t0 = time.perf_counter()
    raw_question = request.question
    clean_question = sanitize_input(raw_question)

    # 1. Input Guardrails
    input_audit = check_input(clean_question)
    if not input_audit["allowed"]:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return QueryResponse(
            question=raw_question,
            answer=(
                f"Your query was flagged by the security guardrail: {input_audit['reason']} "
                "Please submit a valid question related to the Jio Platforms Limited IPO report."
            ),
            citations=[],
            sources=[],
            guardrails=GuardrailAudit(
                allowed=False,
                risk=input_audit["risk"],
                reason=input_audit["reason"],
                output_passed=False,
                output_warnings=["blocked_by_input_guardrail"],
                cited_pages=[],
                available_pages=[],
            ),
            latency_ms=latency_ms,
            model="guardrail_blocked",
            document_title=settings.document_title,
        )

    # 2. Dense Vector Retrieval
    top_k = request.top_k or settings.top_k
    retrieved_chunks = retriever.retrieve(clean_question, top_k=top_k)

    # 3. Grounded Answer Generation
    gen_result = generator.generate_answer(
        question=clean_question,
        passages=retrieved_chunks,
        require_citations=request.require_citations,
    )
    answer = gen_result["answer"]
    model_used = gen_result["model"]

    # 4. Output Guardrails & Citation Verification
    output_audit = check_output(answer, retrieved_chunks)

    # Extract integer citations
    citations = [
        int(p) for p in output_audit["cited_pages"] if p.isdigit()
    ]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return QueryResponse(
        question=raw_question,
        answer=answer,
        citations=citations,
        sources=[
            RetrievedChunk(
                chunk_id=c.get("chunk_id", 0),
                page=c["page"],
                score=c.get("score", 0.0),
                text=c["text"],
                source_file=c.get("source_file", "847332133Jio Platforms Limited IPO.pdf"),
            )
            for c in retrieved_chunks
        ],
        guardrails=GuardrailAudit(
            allowed=True,
            risk="low",
            reason=input_audit["reason"],
            output_passed=output_audit["passed"],
            output_warnings=output_audit["warnings"],
            cited_pages=output_audit["cited_pages"],
            available_pages=output_audit["available_pages"],
        ),
        latency_ms=latency_ms,
        model=model_used,
        document_title=settings.document_title,
    )


@router.get("/chunks", response_model=ChunksResponse)
def get_chunks():
    """Retrieve all indexed chunks of the fixed Jio IPO report."""
    raw_chunks = retriever.vector_store.get_all_chunks()
    return ChunksResponse(
        total_chunks=len(raw_chunks),
        chunks=[
            ChunkItem(
                chunk_id=c.get("chunk_id", i),
                page=c.get("page", 1),
                text=c.get("text", ""),
                source_file=c.get("source_file", "847332133Jio Platforms Limited IPO.pdf"),
                document_id=c.get("document_id", "jio_platforms_ipo"),
            )
            for i, c in enumerate(raw_chunks)
        ],
    )


@router.post("/reindex")
def reindex_document():
    """Force re-indexation of the fixed Jio Platforms IPO PDF."""
    try:
        count = retriever.rebuild_fixed_document_index()
        return {
            "status": "success",
            "message": f"Successfully re-indexed fixed Jio IPO document with {count} chunks.",
            "chunks_count": count,
        }
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-indexing failed: {str(e)}",
        )


@router.get("/evaluation/run", response_model=EvaluationRunResponse)
def run_evaluation(k: int = Query(default=3, ge=1, le=5)):
    """Run automated benchmark evaluation on Jio IPO ground-truth questions."""
    try:
        summary = run_evaluation_suite(retriever, generator, k=k)
        return EvaluationRunResponse(**summary)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and inspect an alternative PDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a PDF file.",
        )

    pdf_bytes = await file.read(MAX_FILE_SIZE + 1)
    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF must be 10 MB or smaller.",
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file does not appear to be a valid PDF.",
        )

    try:
        pages = extract_pdf(pdf_bytes)
        chunks = chunk_pages(pages, source_file=file.filename)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not process this PDF.",
        ) from error

    return {
        "filename": file.filename,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "sample_chunks": chunks[:5],
    }
