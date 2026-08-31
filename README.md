# Technological University Intelligent Admissions AI Assistant (RAG Pipeline)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Vector Store](https://img.shields.io/badge/ChromaDB-Persistent-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An intelligent, production-grade AI Admissions Assistant for a Technological University built with **FastAPI**, **ChromaDB**, and **OpenAI/LLM orchestration**. Designed to handle repetitive applicant inquiries regarding degree programs, tuition fees, financial aid, application deadlines, and industry certifications grounded strictly in business documents.

---

## Key Features

1. **Grounded Business RAG Architecture**:
   - Custom chunking algorithm with configurable size (500 chars) and overlap (100 chars).
   - ChromaDB persistent vector database with cosine similarity search.
   - Grounded strictly in 3+ business documents with strict anti-hallucination controls.

2. **Double-Filter Human Escalation Engine**:
   - **Filter 1 (Pre-LLM Similarity Score Check)**: Evaluates retrieved vector similarity scores before calling the LLM API. If no relevant chunks meet the confidence threshold (`0.35`), the system escalates immediately **without calling the LLM API**, saving API costs and preventing hallucinations.
   - **Filter 2 (Post-LLM System Prompt Tag)**: Evaluates LLM responses for the explicit `[ESCALATE_TO_HUMAN]` prefix tag when questions pass vector search but are determined to be out-of-scope.
   - Generates structured ticket objects (`TICK-XXXXXX`) for human admissions counselors.

3. **Usage & Cost Telemetry (`response.usage`)**:
   - Dynamically tracks token consumption (`prompt_tokens`, `completion_tokens`, `total_tokens`) extracted directly from the LLM API `response.usage` object.
   - Calculates monetary costs in USD ($) and tracks escalation rate (%) in real-time.

4. **Response Caching**:
   - Built-in normalized semantic/exact memory cache (`CacheService`) to return zero-cost, instant responses for frequent applicant questions.

5. **Multi-Channel Integration**:
   - `POST /api/chat`: REST API endpoint for web and mobile frontends.
   - `POST /api/webhook`: Webhook endpoint compatible with Telegram bots and n8n workflow triggers.
   - Glassmorphism Web UI at `/` featuring live chat, source chunk inspector, and telemetry dashboard.

---

## Project Structure

```
.
├── app/
│   ├── api/                 # API endpoints and route definitions
│   ├── main.py              # FastAPI server entry point & startup logic
│   ├── prompts/
│   │   └── system_prompts.py# Grounded system prompt & 4 Few-Shot examples
│   ├── rag/
│   │   ├── engine.py        # RAG pipeline orchestrator
│   │   ├── ingest.py        # Business document ingestion script
│   │   ├── text_splitter.py # Recursive text splitter with overlap
│   │   └── vector_store.py  # ChromaDB vector store manager & hybrid embeddings
│   ├── services/
│   │   ├── cache_service.py # Fast semantic/exact query response cache
│   │   ├── escalation_service.py # Double-filter escalation & ticket manager
│   │   └── metrics_service.py    # Usage token cost & telemetry tracking
│   ├── static/
│   │   └── index.html       # Modern dark-mode web application & dashboard
│   └── tests/
│       └── test_rag.py      # Automated test suite
├── data/
│   ├── chroma_db/           # Persistent ChromaDB vector database files
│   └── documents/           # University business documents (TXT/MD)
│       ├── 01_programs_and_modalities.txt
│       ├── 02_tuition_fees_and_financial_aid.txt
│       └── 03_admissions_and_certifications.txt
├── .env.example             # Environment variables configuration template
├── package_project.py       # Submission packaging utility script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Quick Start & Setup Guide

### 1. Environment Setup & Installation

Clone or extract the repository, create a virtual environment, and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Linux/macOS)
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.1
CHROMA_DB_DIR=./data/chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
SIMILARITY_THRESHOLD=0.35
ENABLE_SEMANTIC_CACHE=true
HOST=0.0.0.0
PORT=8000
```

> **Note**: If `OPENAI_API_KEY` is not provided or remains as default placeholder, the system runs with a high-accuracy local vectorizer and offline generator so all endpoints and features remain 100% testable.

### 3. Ingest University Business Documents

Run the ingestion script to populate ChromaDB vector store:

```bash
python -m app.rag.ingest
```

*Output:*
```
INFO:app.rag.ingest:Found 3 business documents in './data/documents'...
INFO:app.rag.ingest:Ingestion Complete! Total 18 chunks stored in ChromaDB.
```

### 4. Run FastAPI Backend Server

Launch the server locally:

```bash
python app/main.py
```
Or using uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
- **Interactive Web UI & Dashboard**: `http://localhost:8000`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`

---

## API Documentation & Examples

### 1. Chat Query Endpoint (`POST /api/chat`)

**Request:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How much is undergraduate tuition per semester?",
       "session_id": "student_123"
     }'
```

**Response:**
```json
{
  "answer": "Full-time undergraduate tuition at TechUni is $4,200 USD per semester ($8,400 USD per academic year).",
  "escalated": false,
  "escalation_details": null,
  "sources": ["02_tuition_fees_and_financial_aid.txt"],
  "retrieved_chunks": [
    {
      "text": "Full-Time Tuition (15-18 credits per semester): $4,200 USD per semester.",
      "source": "02_tuition_fees_and_financial_aid.txt",
      "similarity": 0.8924
    }
  ],
  "cached": false,
  "token_usage": {
    "prompt_tokens": 420,
    "completion_tokens": 35,
    "total_tokens": 455
  },
  "estimated_cost_usd": 0.000084,
  "latency_ms": 312.4
}
```

### 2. Webhook Endpoint for Telegram / n8n (`POST /api/webhook`)

**Request:**
```bash
curl -X POST "http://localhost:8000/api/webhook" \
     -H "Content-Type: application/json" \
     -d '{
       "message": {
         "text": "Are there any scholarships for women in STEM?",
         "chat": { "id": 987654321 }
       }
     }'
```

### 3. Telemetry & Metrics Endpoint (`GET /api/metrics`)

**Request:**
```bash
curl http://localhost:8000/api/metrics
```

**Response:**
```json
{
  "total_queries_processed": 14,
  "escalation_metrics": {
    "total_escalations": 2,
    "escalation_rate_pct": 14.29,
    "pre_llm_escalations_saved_cost": 1,
    "post_llm_escalations": 1
  },
  "token_usage": {
    "prompt_tokens": 4210,
    "completion_tokens": 580,
    "total_tokens": 4790
  },
  "financial_metrics": {
    "total_estimated_cost_usd": 0.000980,
    "currency": "USD"
  },
  "cache_metrics": {
    "hits": 3,
    "misses": 11,
    "hit_rate_pct": 21.43
  }
}
```

---

## Automated Verification & Testing

Run unit & integration tests using `pytest`:

```bash
pytest app/tests/test_rag.py -v
```

---

## Packaging Deliverables

To generate the final `.zip` submission archive required for evaluation:

```bash
python package_project.py
```

This generates `university_admissions_rag_assistant.zip` in the root directory containing all code, business documents, vector embeddings, documentation, and configuration files.

---

## License

This project is licensed under the MIT License. Developed for Technological University Admissions.
