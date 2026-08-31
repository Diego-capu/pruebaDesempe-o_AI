# Technological University Intelligent Admissions AI Assistant (Miku AI — RAG Pipeline)

[![Render Deployment](https://img.shields.io/badge/Render-Live%20Production-46E3B7?logo=render&logoColor=white)](https://miku-ai-admissions-assistant.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Vector Store](https://img.shields.io/badge/ChromaDB-Persistent-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An intelligent, production-grade AI Admissions Assistant for a Technological University built with **FastAPI**, **ChromaDB**, and **Google Gemini / OpenAI orchestration**. Designed to handle repetitive applicant inquiries regarding degree programs, tuition fees, financial aid, application deadlines, and industry certifications grounded strictly in business documents.

🌐 **Live Production URL:** [https://miku-ai-admissions-assistant.onrender.com](https://miku-ai-admissions-assistant.onrender.com)  
📖 **Interactive API Documentation:** [https://miku-ai-admissions-assistant.onrender.com/docs](https://miku-ai-admissions-assistant.onrender.com/docs)

---

## Key Features

1. **Grounded Business RAG Architecture**:
   - Custom chunking algorithm with configurable size (500 chars) and overlap (100 chars).
   - ChromaDB persistent vector database with cosine similarity search.
   - Grounded strictly in 3+ business documents with strict anti-hallucination controls.

2. **Double-Filter Human Escalation Engine**:
   - **Filter 1 (Context Evaluation)**: Filters vector chunks based on cosine similarity threshold before LLM synthesis.
   - **Filter 2 (Post-LLM System Prompt Tag)**: Detects explicit `[ESCALATE_TO_HUMAN]` prefix tags when questions require human advisor assistance, while politely rejecting non-academic spam without ticket creation.
   - Generates structured ticket objects (`TICK-XXXXXX`) for human admissions counselors.

3. **Usage & Cost Telemetry (`response.usage`)**:
   - Dynamically tracks token consumption (`prompt_tokens`, `completion_tokens`, `total_tokens`) extracted directly from the LLM API `response.usage` object.
   - Calculates monetary costs in USD ($) and monthly token quota progress (%) in real-time.

4. **Response Caching**:
   - Built-in normalized semantic/exact memory cache (`CacheService`) to return zero-cost, instant responses for frequent applicant questions.

5. **Multi-Channel Integration & Real-time Streaming**:
   - `POST /api/chat/stream`: Real-time Server-Sent Events (SSE) streaming endpoint with progressive token delivery and interactive Quick Reply Chips.
   - `POST /api/chat`: REST API endpoint for web and mobile frontends.
   - `POST /api/webhook`: Webhook endpoint compatible with Telegram bots and n8n workflow triggers.
   - `n8n_workflow.json`: Ready-to-import n8n workflow connecting Telegram & Webhook to the RAG backend with automated human escalation routing.
   - Glassmorphism Web UI at `/` featuring live chat, quick reply chips, source chunk inspector, and telemetry dashboard.

---

## Project Structure

```
.
├── app/
│   ├── api/                 # API endpoints and route definitions
│   ├── main.py              # FastAPI server entry point, streaming & webhook logic
│   ├── prompts/
│   │   └── system_prompts.py# Grounded system prompt & 7 Few-Shot examples
│   ├── rag/
│   │   ├── engine.py        # RAG pipeline orchestrator (Sync & SSE Stream)
│   │   ├── ingest.py        # Business document ingestion script
│   │   ├── text_splitter.py # Recursive text splitter with overlap
│   │   └── vector_store.py  # ChromaDB vector store manager & embeddings
│   ├── services/
│   │   ├── cache_service.py # Fast semantic/exact query response cache
│   │   ├── escalation_service.py # Escalation evaluation & ticket manager
│   │   └── metrics_service.py    # Usage token cost, quota & telemetry tracking
│   ├── static/
│   │   └── index.html       # Modern dark-mode web application & dashboard
│   └── tests/
│       └── test_rag.py      # Automated test suite (8 unit & integration tests)
├── data/
│   ├── chroma_db/           # Persistent ChromaDB vector database files
│   └── documents/           # University & Academy business documents (TXT)
│       ├── 01_programs_and_modalities.txt
│       ├── 02_tuition_fees_and_financial_aid.txt
│       └── 03_admissions_and_certifications.txt
├── .env.example             # Environment variables configuration template
├── n8n_workflow.json        # Exported n8n automation workflow (Telegram + Webhook)
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

### 2. Real-time Streaming Query Endpoint (`POST /api/chat/stream`)

**Request:**
```bash
curl -N -X POST "http://localhost:8000/api/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How much is undergraduate tuition per semester?"
     }'
```

**SSE Event Stream Output:**
```
data: {"type": "token", "content": "Full-Time"}
data: {"type": "token", "content": " Undergraduate Tuition is $4,200 USD per semester."}
data: {"type": "done", "answer": "Full-Time Undergraduate Tuition is $4,200 USD per semester.", "escalated": false, "suggested_chips": ["Scholarships & Financial Aid", "Monthly Installment Plans", "Registration Deadlines"], ...}
```

### 3. Webhook Endpoint for Telegram / n8n (`POST /api/webhook`)

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

### 4. Telemetry & Metrics Endpoint (`GET /api/metrics`)

**Request:**
```bash
curl http://localhost:8000/api/metrics
```

**Response:**
```json
{
  "total_queries_processed": 14,
  "total_tokens_used": 4790,
  "prompt_tokens": 4210,
  "completion_tokens": 580,
  "token_limit_monthly": 1000000,
  "token_usage_percentage": 0.479,
  "escalation_metrics": {
    "total_escalations": 2,
    "escalation_rate_pct": 14.29,
    "pre_llm_escalations_saved_cost": 1,
    "post_llm_escalations": 1
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

## n8n Automation Workflow Setup

The project includes an exportable n8n workflow file (`n8n_workflow.json`):

1. Open your **n8n instance** (Self-hosted or Cloud).
2. Go to **Workflows** > **Import from File** and select `n8n_workflow.json`.
3. The workflow automatically connects:
   - **Telegram Trigger / Webhook Trigger** $\rightarrow$ Captures student message.
   - **HTTP Request** $\rightarrow$ Dispatches query to `POST http://localhost:8000/api/chat`.
   - **IF Node (`Is Escalated?`)** $\rightarrow$ Routes normal answers to the user and triggers an automated alert with `Ticket ID` to the Human Admissions team when escalation is needed.

---

## Cloud Deployment on Render

The application is deployed and publicly accessible in production on **Render**:

🔗 **Production Web Application:** [https://miku-ai-admissions-assistant.onrender.com](https://miku-ai-admissions-assistant.onrender.com)  
📖 **Interactive Swagger Docs:** [https://miku-ai-admissions-assistant.onrender.com/docs](https://miku-ai-admissions-assistant.onrender.com/docs)  
⚡ **Real-Time Streaming Endpoint:** `POST https://miku-ai-admissions-assistant.onrender.com/api/chat/stream`  
📊 **Live Telemetry & Metrics:** `GET https://miku-ai-admissions-assistant.onrender.com/api/metrics`

### Deploying Your Own Instance on Render

The repository includes a ready-to-use **Render Blueprint** (`render.yaml`) and runtime definition (`runtime.txt`):

#### Method 1: Zero-Config Render Blueprint (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** > **Blueprint**.
2. Connect your GitHub account and select repository: `Diego-capu/pruebaDesempe-o_AI`.
3. Enter your `OPENAI_API_KEY` (Google AI Studio or OpenAI key).
4. Click **Apply**. Render will automatically build the environment, launch Uvicorn, and initialize ChromaDB vector storage on startup.

#### Method 2: Manual Web Service Setup
- **Environment:** Python 3 (`3.11.9`)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables:**
  - `OPENAI_API_KEY`: `<YOUR_API_KEY>`
  - `OPENAI_BASE_URL`: `https://generativelanguage.googleapis.com/v1beta/openai/`
  - `LLM_MODEL`: `gemini-3.5-flash-lite`
  - `EMBEDDING_MODEL`: `gemini-embedding-001`
  - `SIMILARITY_THRESHOLD`: `0.55`
  - `MAX_RETRIEVED_CHUNKS`: `3`
  - `TOKEN_LIMIT_MONTHLY`: `1000000`

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

## Author

Diego Andres Ospino Barrios
