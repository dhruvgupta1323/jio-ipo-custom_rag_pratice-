# SecureRAG: Jio Platforms Limited IPO Intelligence

A security-first, evaluation-driven Retrieval-Augmented Generation (RAG) system tailored for fixed financial research prospectuses. Powered by **FastAPI**, **PyMuPDF**, **Sentence-Transformers**, **FAISS**, and **Google DeepMind Gemini**.

---

## 📌 Document Under Analysis
* **Title**: Capstocks Equity Research — Jio Platforms Limited IPO
* **Company**: Jio Platforms Limited
* **Subsidiary**: Reliance Jio Infocomm Limited (533.3M users as of 30-Jun-26)
* **Issue Dates**: 21 Oct 26 – 23 Oct 26
* **Price Range**: ₹1065 to ₹1119 per share
* **Market Lot**: 12 shares (based on upper band)
* **Issue Size**: ₹30,213 Crore
* **Capstocks Rating**: 18 / 30 (Good)
* **Analyst Recommendation**: Subscribe (Medium-to-long-term view)

---

## 🚀 Key Features

* **Fixed Document Ingestion & Structuring**:
  * Accurate 3-page parsing with contextual chunk headers (`[Jio Platforms Limited IPO | Page N]`).
  * Tabular extraction preserving multi-year financial statements across FY24, FY25, FY26, and 30-Jun-26.
* **Hybrid Vector Retrieval**:
  * 384-dimensional dense semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
  * FAISS `IndexFlatIP` (exact normalized cosine similarity).
  * Salient financial entity boost for exact token matches (e.g., `PAT`, `EBITDA`, `RJIL`, `₹1065`, `12 shares`).
* **Multi-Layer Security Guardrails**:
  * **Input Filters**: Prompt injection mitigation, length restrictions (<2000 chars), sanitization.
  * **Retrieval Fencing**: Strict similarity thresholding; document context treated purely as untrusted data.
  * **Output Verification**: Verifies `[Page N]` citations exist only within retrieved chunks to prevent hallucination.
* **Comprehensive Benchmark Evaluation**:
  * 12 ground-truth test queries testing Precision@3, Recall@3, MRR, Keyword Coverage, and Guardrail Pass Rates.
* **Modern Web Interface**:
  * Responsive dark-mode financial terminal UI with prompt chips, live PDF explorer, chunk inspector, and evaluation suite.

---

## 🛠️ Project Structure

```text
secure-rag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py         # REST API endpoints (query, eval, chunks, info)
│   │   ├── core/
│   │   │   ├── config.py         # Settings & environment variables
│   │   │   └── security.py       # Input sanitization & security helpers
│   │   ├── models/
│   │   │   └── schemas.py        # Pydantic data schemas
│   │   ├── services/
│   │   │   ├── embeddings.py     # SentenceTransformers singleton
│   │   │   ├── evaluation.py     # Metrics (Precision, Recall, MRR) & runner
│   │   │   ├── generation.py     # Gemini client with quota-aware fallback
│   │   │   ├── guardrails.py     # Input & output safety validators
│   │   │   ├── ingestion.py      # PyMuPDF parser & table structurer
│   │   │   ├── retrieval.py      # Hybrid FAISS search
│   │   │   └── vector_store.py   # FAISS index persistence
│   │   └── main.py               # FastAPI application & lifespan startup
│   └── tests/                    # 20 Unit & integration tests
├── data/
│   └── jio_ipo/
│       ├── source/               # Fixed IPO PDF prospectus
│       └── index/                # Pre-built FAISS index & chunk metadata
├── evaluation/
│   ├── evaluation.py             # Benchmark runner CLI
│   └── jio_ipo_dataset.json      # 12 Ground-truth test queries
├── frontend/
│   ├── css/style.css             # Glassmorphism dark theme
│   ├── js/app.js                 # Interactive SPA logic & citation router
│   └── index.html                # Main terminal console
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/dhruvgupta1323/jio-ipo-custom_rag_pratice-.git
cd jio-ipo-custom_rag_pratice-
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file based on `.env.example`:
```env
GEMINI_API_KEY="your-gemini-api-key"
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=3
CHUNK_SIZE=700
CHUNK_OVERLAP=120
```

### 3. Run the Application
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the web console at **`http://127.0.0.1:8000`**.

---

## 🧪 Testing & Evaluation

### Run Test Suite (20 Tests)
```bash
python -m pytest backend/tests
```

### Run Benchmark Suite (12 Queries)
```bash
python evaluation/evaluation.py
```

Benchmark Results:
* **Mean Precision@3**: `72.22%`
* **Mean Recall@3**: `91.67%`
* **Mean MRR**: `0.9167`
* **Mean Keyword Coverage**: `77.78%`
* **Guardrail Pass Rate**: `100.00%`

---

## 📜 Disclaimer
This software is built strictly for educational and research purposes analyzing the Capstocks Equity Research note on Jio Platforms Limited IPO. Not financial advice.
