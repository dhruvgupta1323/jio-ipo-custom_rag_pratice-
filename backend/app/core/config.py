from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    gemini_api_key: Optional[str] = ""
    gemini_model: str = "gemini-3.5-flash"
    fallback_models: List[str] = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
    
    # Embedding & RAG Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3
    chunk_size: int = 700
    chunk_overlap: int = 120
    relevance_threshold: float = 0.05
    
    # Fixed Document: Jio Platforms Limited IPO
    document_title: str = "Jio Platforms Limited IPO - Capstocks Equity Research"
    source_pdf_path: str = "data/jio_ipo/source/847332133Jio Platforms Limited IPO.pdf"
    index_dir: str = "data/jio_ipo/index"
    evaluation_dataset_path: str = "evaluation/jio_ipo_dataset.json"
    evaluation_results_dir: str = "evaluation/results"
    
    # Admin secret for management endpoints
    admin_api_key: Optional[str] = "jio-ipo-admin-key-2026"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def source_pdf_abs(self) -> Path:
        return BASE_DIR / self.source_pdf_path

    @property
    def index_dir_abs(self) -> Path:
        return BASE_DIR / self.index_dir

    @property
    def evaluation_dataset_abs(self) -> Path:
        return BASE_DIR / self.evaluation_dataset_path

    @property
    def evaluation_results_abs(self) -> Path:
        return BASE_DIR / self.evaluation_results_dir


settings = Settings()
