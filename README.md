# Technological University Intelligent Admissions AI Assistant (Miku AI — RAG Pipeline)

[![Render Deployment](https://img.shields.io/badge/Render-Live%20Production-46E3B7?logo=render&logoColor=white)](https://miku-ai-admissions-assistant.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Vector Store](https://img.shields.io/badge/ChromaDB-Persistent-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-17%20Passed%20(100%25)-success.svg)](https://github.com/Diego-capu/pruebaDesempe-o_AI)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An intelligent, production-grade AI Admissions Assistant for Technological University (TechUni) built with **FastAPI**, **ChromaDB**, and **Google Gemini / OpenAI orchestration**. Designed to handle repetitive applicant inquiries regarding degree programs, study modalities, tuition costs, financial aid scholarships, application deadlines, and industry certifications grounded strictly in verified business documents.

🌐 **Live Production URL:** [https://miku-ai-admissions-assistant.onrender.com](https://miku-ai-admissions-assistant.onrender.com)  
📖 **Interactive API Documentation (Swagger UI):** [https://miku-ai-admissions-assistant.onrender.com/docs](https://miku-ai-admissions-assistant.onrender.com/docs)  
🐙 **GitHub Repository:** [https://github.com/Diego-capu/pruebaDesempe-o_AI](https://github.com/Diego-capu/pruebaDesempe-o_AI)

---

## Architecture & System Overview

```mermaid
flowchart TD
    User([Student / Applicant]) -->|Query| Frontend[Web UI / SSE / Telegram]
    Frontend -->|POST /api/chat or /api/chat/stream| API[FastAPI Controller]
    
    API --> Cache{Exact / Semantic Cache?}
    Cache -->|Hit| InstantResponse[Return Cached Response\n0ms / $0.00]
    
    Cache -->|Miss| PreFilter{Greeting Fast-Path?}
    PreFilter -->|Greeting| DirectGreeting[Cordial Greeting\n0 Context Chunks / No Ticket]
    
    PreFilter -->|Inquiry| QueryCleaner[Regex Symbol Cleaner\nStrip () / , + &]
    QueryCleaner --> Expander[Domain Query Expander\n<= 8 Words Expansion]
    Expander --> Chroma[(ChromaDB Vector Store\n28 Chunks / Cosine Distance)]
    
    Chroma --> ContextCleaner[Context Chunk Sanitizer\nStrip === Dividers & Raw Headers]
    ContextCleaner --> PromptBuilder[Grounded Prompt Builder\n10 Few-Shot Examples]
    
    PromptBuilder --> LLM[LLM Engine\nGemini / GPT-4o]
    LLM --> PostFilter{Post-LLM Evaluator\nTag Detection}
    
    PostFilter -->|ESCALATE_TO_HUMAN| TicketGen[Escalation Ticket Generated\nTICK-XXXXXX]
    PostFilter -->|Grounded Answer| AnswerGen[Synthesized Response\nNo Robotic Headers]
    
    TicketGen --> Telemetry[Metrics & Telemetry Service\nToken Usage & USD Cost]
    AnswerGen --> Telemetry
    Telemetry --> ResponseStream[Server-Sent Events / JSON Payload]
    ResponseStream --> Frontend
```

---

## Key Features

### 1. Grounded Business RAG Architecture
- **Persistent Vector Store:** ChromaDB persistent storage initialized with 28 indexed document chunks from official university documents (`01_programs_and_modalities.txt`, `02_tuition_fees_and_financial_aid.txt`, `03_admissions_and_certifications.txt`).
- **Domain-Aware Query Expansion:** Automatically expands short inquiries ($\le 8$ words) across 6 critical domains:
  - **Certifications:** Expands with AWS Academy, Cisco CCNA, CyberOps, Google Cloud, NVIDIA DLI.
  - **Study Modalities:** Expands with On-Campus, Hybrid, 100% Online, asynchronous lectures.
  - **Academic Programs:** Expands with B.Sc. AI & Data, B.Sc. CyberCloud, M.Sc. Software Architecture, Bootcamps.
  - **Tuition & Aid:** Expands with semester fees, STEM scholarships, payment installments.
  - **Admissions:** Expands with application requirements, $75 fee, transcripts, deadlines.
  - **Payment Methods:** Expands queries regarding unlisted payment methods (cryptocurrency, PayPal) to retrieve official financing options.
- **Punctuation & Noise Sanitization:** Regex cleaning (`re.sub(r"[()/,+&]", " ", query)`) removes characters that degrade dense vector distance metrics.
- **Context Leaking Prevention:** Eliminates divider lines (`===`) and raw headers (`DOCUMENT 01:`, `1. ADMISSIONS REQUIREMENTS`) before LLM prompt injection.
- **Direct, Natural Synthesis:** Zero repetitive or robotic boilerplate prefixes (e.g. removed *"Based on official University business documents:"*).

### 2. Double-Filter Human Escalation Engine
- **Pre-LLM Filtering:**
  - **Greeting Fast-Path:** Bilingual greetings (e.g. *"Hello!"*, *"Hola, buenos días"*) bypass vector search completely, returning warm, cordial welcomes with 0 chunks and no ticket generation.
  - **Spam / Trivia Boundary Enforcement:** Irrelevant non-educational queries (e.g. *"pizza"*, recipes, jokes) are politely delimited to admissions assistance without generating escalation tickets.
- **Post-LLM System Tag Detection (`[ESCALATE_TO_HUMAN]`):**
  - Legitimate academic queries outside standard documents (e.g. *"Civil Engineering"*, *"Aerospace Engineering"*, *"corporate discounts for 20+ engineers"*) trigger the `[ESCALATE_TO_HUMAN]` tag and create a structured tracking ticket (`TICK-XXXXXX`).
  - **Unlisted Payment Methods Protection:** Explicitly prevents escalation for unlisted payment methods (such as cryptocurrency or PayPal). The bot politely clarifies that the requested method is not accepted and outlines official available payment options (Early Payment Discount 8%, 3-Pay / 5-Pay Installments, Corporate Sponsorship).

### 3. Usage & Cost Telemetry Service (`response.usage`)
- Tracks exact token consumption (`prompt_tokens`, `completion_tokens`, `total_tokens`) directly from the LLM API `response.usage` object.
- Computes real-time estimated financial costs in USD ($) and monthly token quota consumption against a configurable limit (default: 1,000,000 tokens).
- Exposes complete metrics via `GET /api/metrics`.

### 4. High-Performance Caching (`CacheService`)
- In-memory semantic and normalized exact-match cache.
- Instant, zero-cost responses ($0.00 USD, $<10$ ms latency) for repetitive applicant questions.

### 5. Modern Glassmorphic Web Interface & Parallax Background
- **Ultracompact Floating Pill Navbar (`rounded-full`, `w-auto max-w-fit`, `z-30`):**
  - Brand identity: Circular Miku avatar frame (`rounded-full`, `w-6 h-6`), title `Miku AI`, and green `v1.0` badge.
  - Status indicator: Live `Online` pill with pulsating emerald light.
  - Action button: Dark pill `Show Telemetry` toggle that dynamically illuminates when the inspector is active.
- **Dynamic Glassmorphic Transition (`.translucent-glass`):**
  - **Default State:** Translucent glass (`rgba(9, 9, 11, 0.20)`, `blur(4px)`, opacity 0.85) allowing the multi-layered 3D Parallax mountain and fog scene to shine through cleanly.
  - **Active State:** Seamlessly transitions (`0.45s` cubic-bezier) to solid frosted glass (`rgba(9, 9, 11, 0.85)`, `blur(28px)`, opacity 1) upon cursor hover (`:hover`), card focus (`:focus-within`), or while typing (`input` listener).
  - **Zero Overlap Layout:** Generous top padding (`pt-20`) on the main container ensuring clean vertical spacing without overlapping elements.
- **Interactive Controls:**
  - Real-time Server-Sent Events (SSE) streaming with typewriter token delivery.
  - Quick-reply topic suggestion chips.
  - Collapsible RAG Context Inspector displaying retrieved ChromaDB chunks and similarity scores.

---

## Project Structure

```
.
├── app/
│   ├── api/                     # API route handlers
│   ├── main.py                  # FastAPI application, startup events, SSE & webhooks
│   ├── prompts/
│   │   └── system_prompts.py    # Grounded system prompt & 10 Few-Shot examples
│   ├── rag/
│   │   ├── engine.py            # End-to-end RAG orchestrator, query expansion & cleaner
│   │   ├── ingest.py            # Business document chunking and ChromaDB ingestion
│   │   ├── text_splitter.py     # Recursive text splitter with overlap
│   │   └── vector_store.py      # ChromaDB vector store manager & embedding generator
│   ├── services/
│   │   ├── cache_service.py     # In-memory query response cache
│   │   ├── escalation_service.py# Double-filter escalation evaluator & ticket manager
│   │   └── metrics_service.py   # Token telemetry, cost calculation & quota tracking
│   ├── static/
│   │   ├── index.html           # Dark-mode glassmorphism web interface
│   │   ├── miku.gif             # Official avatar animation
│   │   └── parallax/            # 17 local multi-layered parallax assets (PNG)
│   └── tests/
│       └── test_rag.py          # Unit tests (Cache, Vector store, Chunking, Telemetry)
├── data/
│   ├── chroma_db/               # Persistent ChromaDB vector database files
│   └── documents/               # Verified University & Academy business documents
│       ├── 01_programs_and_modalities.txt
│       ├── 02_tuition_fees_and_financial_aid.txt
│       └── 03_admissions_and_certifications.txt
├── src/
│   ├── components/
│   │   └── ui/
│   │       └── mini-navbar.tsx  # React/TypeScript Floating Pill Navbar component
│   └── lib/
│       └── utils.ts             # Tailwind class merging utility (cn)
├── tests/
│   └── test_rag_flow.py         # End-to-end RAG intent & escalation verification tests
├── .env.example                 # Environment variable template
├── n8n_workflow.json            # Automation workflow template (Telegram + Webhook)
├── package_project.py           # Submission zip packaging utility
├── requirements.txt             # Python production dependencies
├── render.yaml                  # Render Infrastructure-as-Code Blueprint
├── runtime.txt                  # Python runtime version for cloud deployment
└── README.md                    # Project documentation
```

---

## Quick Start & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Installation & Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Diego-capu/pruebaDesempe-o_AI.git
cd pruebaDesempe-o_AI

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure the environment variables in `.env`:
```ini
OPENAI_API_KEY=your_openai_or_gemini_api_key
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
> **Note:** The engine includes a built-in offline simulation fallback so the system remains fully functional and 100% testable even without an active external API key.

### 4. Vector Database Ingestion
Populate ChromaDB with the official documents:
```bash
python -m app.rag.ingest
```
*(FastAPI also performs automatic cold-boot ingestion during startup if ChromaDB is empty).*

### 5. Run the Application
Launch the server with Uvicorn:
```bash
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Navigate to:
- **Interactive Web UI:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Explorer:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Documentation & Examples

### 1. Synchronous Chat (`POST /api/chat`)
**Request:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the tuition for the M.Sc.?",
       "session_id": "applicant_01"
     }'
```

**Response:**
```json
{
  "answer": "Full-time tuition for the Master of Science in Software Architecture (M.Sc. Software Arch) at TechUni is $3,600 USD per semester (12 credits), totaling $14,400 USD across the 4-semester program.",
  "escalated": false,
  "escalation_details": null,
  "sources": ["02_tuition_fees_and_financial_aid.txt"],
  "retrieved_chunks": [
    {
      "text": "Master of Science Programs (M.Sc. Software Arch): Tuition per semester (12 credits) is $3,600 USD. Total Program Cost across 4 semesters is $14,400 USD.",
      "source": "02_tuition_fees_and_financial_aid.txt",
      "similarity": 0.892
    }
  ],
  "suggested_chips": [
    "Are there any scholarships for women in STEM?",
    "What are the payment plan options?",
    "What are the application requirements?"
  ],
  "cached": false,
  "token_usage": {
    "prompt_tokens": 420,
    "completion_tokens": 38,
    "total_tokens": 458
  },
  "estimated_cost_usd": 0.000084,
  "latency_ms": 284.5
}
```

### 2. Real-Time Streaming (`POST /api/chat/stream`)
Supports Server-Sent Events (SSE) with progressive token delivery:
```bash
curl -N -X POST "http://localhost:8000/api/chat/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What industry certifications are included in the curriculum?"
     }'
```

### 3. Human Escalation Example (`POST /api/chat`)
**Request:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Do you offer Civil Engineering?",
       "session_id": "applicant_02"
     }'
```

**Response:**
```json
{
  "answer": "We currently do not offer a Civil Engineering program in our curriculum. I have forwarded your inquiry to an admissions advisor to assist you with available engineering tracks or transfer options.",
  "escalated": true,
  "escalation_details": {
    "stage": "Post-LLM (System Prompt Tag)",
    "reason": "Legitimate request requiring human advisor follow-up",
    "ticket_id": "TICK-A8F2B1",
    "max_similarity_score": 0.28
  },
  "sources": [],
  "retrieved_chunks": [],
  "suggested_chips": [
    "Academic Programs Available",
    "What are the admission requirements?",
    "How much is tuition per semester?"
  ]
}
```

### 4. Telemetry Metrics (`GET /api/metrics`)
**Request:**
```bash
curl -s http://localhost:8000/api/metrics
```

**Response:**
```json
{
  "total_queries_processed": 24,
  "total_tokens_used": 6840,
  "prompt_tokens": 5820,
  "completion_tokens": 1020,
  "token_limit_monthly": 1000000,
  "token_usage_percentage": 0.684,
  "average_latency_ms": 245.8,
  "escalation_metrics": {
    "total_escalations": 3,
    "escalation_rate_pct": 12.5,
    "pre_llm_escalations_saved_cost": 2,
    "post_llm_escalations": 1
  },
  "financial_metrics": {
    "total_estimated_cost_usd": 0.00142,
    "currency": "USD"
  },
  "cache_metrics": {
    "hits": 6,
    "misses": 18,
    "total_requests": 24,
    "hit_rate_pct": 25.0,
    "cached_entries_count": 8
  },
  "indexed_vector_chunks": 28
}
```

---

## Comprehensive Automated Test Matrix

The project features a **17-test automated verification matrix** covering retrieval accuracy, intent handling, escalation controls, and system telemetry:

| # | Test Name | Query / Focus | Expected Behavior | Status |
| :-: | :--- | :--- | :--- | :-: |
| 1 | `test_greeting_bypass_pre_llm` | `"Hello!"` | Pre-LLM bypass, 0 vector chunks, zero tickets | ✅ PASS |
| 2 | `test_general_intent_sign_up` | `"I want to sign up"` | Broad signup orientation, lists degree tracks, no ticket | ✅ PASS |
| 3 | `test_specific_data_msc_tuition` | `"What is the tuition for the M.Sc.?"` | Accurate pricing ($3,600/sem, $14,400 total), no ticket | ✅ PASS |
| 4 | `test_legitimate_out_of_scope` | `"Do you offer Civil Engineering?"` | Legitimate escalation, generates ticket `TICK-XXXXXX` | ✅ PASS |
| 5 | `test_spam_irrelevant_pizza` | `"pizza"` | Polite boundary rejection, no tickets | ✅ PASS |
| 6 | `test_academic_programs_available` | `"Academic Programs Available"` | In-scope programs breakdown (B.Sc., M.Sc., Bootcamps), no ticket | ✅ PASS |
| 7 | `test_study_modalities` | `"Study Modalities (Online/Hybrid)"` | In-scope breakdown (On-Campus, Hybrid, 100% Online), no ticket | ✅ PASS |
| 8 | `test_cisco_and_aws_certifications`| `"Cisco & AWS Certifications"` | In-scope certification pathways & vouchers, no ticket | ✅ PASS |
| 9 | `test_cryptocurrency_payment` | `"Do you accept cryptocurrency...?"` | Clarifies crypto refusal, lists official payment plans, no ticket | ✅ PASS |
| 10 | `test_vector_store_ingestion` | Ingestion verification | ChromaDB persists and indexes 28 document chunks | ✅ PASS |
| 11 | `test_text_splitter_overlap` | Recursive chunking | 500 characters with 100 characters sliding overlap | ✅ PASS |
| 12 | `test_semantic_cache_hit` | Cache service | Second identical query returns instant cached response | ✅ PASS |
| 13 | `test_telemetry_cost_recording` | Cost calculation | Verifies exact token extraction and USD tracking | ✅ PASS |
| 14 | `test_quota_consumption_limit` | Quota calculation | Correctly tracks % usage against monthly limit | ✅ PASS |
| 15 | `test_multilingual_spanish_support`| Spanish inquiries | Correct Spanish synthesis without false escalations | ✅ PASS |
| 16 | `test_clean_context_chunks` | Header sanitizer | Strips raw dividers (`===`) and section headers | ✅ PASS |
| 17 | `test_pydantic_schema_validation` | API payload models | Validates request and response contracts | ✅ PASS |

To execute the complete test suite:
```bash
PYTHONPATH=. pytest -v
```

---

## Cloud Deployment (Render)

The application includes native **Render Blueprint** configuration (`render.yaml`) and runtime specifications (`runtime.txt`):

### Deployment Instructions:
1. Push your changes to GitHub: `https://github.com/Diego-capu/pruebaDesempe-o_AI`.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** > **Blueprint**.
3. Select your repository. Render automatically reads `render.yaml` and deploys the web service.
4. Set your `OPENAI_API_KEY` in the Render Environment Variables tab.
5. Render automatically executes cold-boot ingestion during startup and begins serving the application.

---

## Submission Packaging Utility

To generate the final `.zip` submission package:
```bash
python package_project.py
```
This produces `university_admissions_rag_assistant.zip` in the root directory containing all source code, business documents, vector database assets, tests, configurations, and documentation.

---

## Author
**Diego Andres Ospino Barrios**  
TechUni Admissions AI Assistant Project
