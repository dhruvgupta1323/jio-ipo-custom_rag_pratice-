from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question regarding the Jio IPO document")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of document chunks to retrieve")
    require_citations: bool = Field(default=True, description="Enforce [Page X] citation formatting in answer")


class RetrievedChunk(BaseModel):
    chunk_id: int
    page: int
    score: float
    text: str
    source_file: str


class GuardrailAudit(BaseModel):
    allowed: bool
    risk: str
    reason: str
    output_passed: bool
    output_warnings: List[str]
    cited_pages: List[str]
    available_pages: List[str]


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[int]
    sources: List[RetrievedChunk]
    guardrails: GuardrailAudit
    latency_ms: float
    model: str
    document_title: str


class DocumentInfo(BaseModel):
    document_title: str
    file_name: str
    file_size_bytes: int
    total_pages: int
    total_chunks: int
    is_indexed: bool
    company_name: str = "Jio Platforms Limited"
    issue_date: str = "21 Oct 26 – 23 Oct 26"
    price_band: str = "₹1065 to ₹1119"
    market_lot: int = 12
    face_value: int = 10
    issue_size: str = "₹30,213 Crore"
    sector: str = "Mobile & Digital Services"
    capstocks_rating: str = "18 / 30 (Good)"
    recommendation: str = "Subscribe (Medium-to-long-term view)"


class EvaluationItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    relevant_pages: List[int]
    expected_keywords: List[str]


class EvaluationResultItem(BaseModel):
    id: str
    question: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    keyword_coverage: float
    retrieved_pages: List[int]
    relevant_pages: List[int]
    guardrail_passed: bool
    generated_answer: str
    latency_ms: float


class EvaluationRunResponse(BaseModel):
    total_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_keyword_coverage: float
    guardrail_pass_rate: float
    evaluation_time_seconds: float
    results: List[EvaluationResultItem]


class ChunkItem(BaseModel):
    chunk_id: int
    page: int
    text: str
    source_file: str
    document_id: str


class ChunksResponse(BaseModel):
    total_chunks: int
    chunks: List[ChunkItem]
