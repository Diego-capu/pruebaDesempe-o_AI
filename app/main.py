import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.rag.engine import RAGEngine
from app.rag.ingest import ingest_documents

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TechUniAdmissionsAPI")

# Initialize FastAPI App
app = FastAPI(
    title="TechUni Intelligent Admissions RAG Assistant API",
    description="Production RAG Backend for Technological University Admissions with Double-Filter Escalation & Cost Metrics",
    version="1.0.0"
)

# Initialize RAG Engine Instance
rag_engine = RAGEngine()

# Static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Automatic Startup Event to ingest docs into ChromaDB if collection is empty
@app.on_event("startup")
def startup_event():
    try:
        count = rag_engine.vector_store.get_document_count()
        if count == 0:
            logger.info("ChromaDB vector store is empty. Auto-ingesting business documents...")
            ingest_documents()
        else:
            logger.info(f"ChromaDB initialized with {count} indexed document chunks.")
    except Exception as e:
        logger.error(f"Startup vector database initialization error: {e}")

# Request Models
class ChatRequest(BaseModel):
    query: str = Field(..., description="Student query or question regarding admissions, fees, or courses.", example="How much is undergraduate tuition?")
    session_id: Optional[str] = Field("default_session", description="Session or user identifier.")

class WebhookRequest(BaseModel):
    query: Optional[str] = None
    chat_id: Optional[str] = None
    message: Optional[Dict[str, Any]] = None

# Root Endpoint: Serves HTML Dashboard / UI
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>TechUni Admissions RAG Assistant API is running.</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>"

# Endpoint 1: Primary Chat Query API
@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        response = rag_engine.process_query(query=request.query, session_id=request.session_id)
        return response
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=f"Internal RAG pipeline error: {str(e)}")

# Endpoint 1b: Real-time Streaming Chat API (Server-Sent Events)
@app.post("/api/chat/stream")
def chat_stream_endpoint(request: ChatRequest):
    """
    Real-time streaming SSE chat endpoint. Returns tokens progressively
    along with contextual quick reply chips and full RAG telemetry.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        return StreamingResponse(
            rag_engine.process_query_stream(query=request.query, session_id=request.session_id),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.exception("Error processing chat stream request")
        raise HTTPException(status_code=500, detail=f"Internal streaming RAG pipeline error: {str(e)}")

# Endpoint 2: Telegram / n8n HTTP Webhook Entrypoint
@app.post("/api/webhook")
async def webhook_endpoint(req: Request):
    """
    HTTP Webhook endpoint for Telegram bot updates or external n8n automation pipelines.
    Accepts direct JSON payload or Telegram update format.
    """
    try:
        payload = await req.json()
        
        # Extract query text based on payload structure
        query_text = ""
        chat_id = "webhook_user"

        if "message" in payload and isinstance(payload["message"], dict):
            query_text = payload["message"].get("text", "")
            chat_id = str(payload["message"].get("chat", {}).get("id", "telegram_user"))
        elif "query" in payload:
            query_text = payload.get("query", "")
            chat_id = str(payload.get("chat_id", "webhook_user"))
        elif "text" in payload:
            query_text = payload.get("text", "")

        if not query_text.strip():
            return JSONResponse({"status": "ignored", "reason": "No text field found in webhook payload."})

        # Process query through RAG pipeline
        rag_response = rag_engine.process_query(query=query_text, session_id=chat_id)

        # Standard Webhook Response Format
        return {
            "status": "success",
            "chat_id": chat_id,
            "response": rag_response["answer"],
            "escalated": rag_response["escalated"],
            "sources": rag_response["sources"],
            "metrics": {
                "token_usage": rag_response["token_usage"],
                "cost_usd": rag_response["estimated_cost_usd"],
                "latency_ms": rag_response["latency_ms"]
            }
        }
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

# Endpoint 3: Manual Re-ingestion Trigger
@app.post("/api/ingest")
def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Triggers re-indexing of documents in data/documents into ChromaDB.
    """
    try:
        num_chunks = ingest_documents()
        return {
            "status": "success",
            "message": f"Successfully ingested business documents into ChromaDB.",
            "total_chunks": num_chunks,
            "document_count": rag_engine.vector_store.get_document_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# Endpoint 4: Metrics Telemetry API
@app.get("/api/metrics")
def get_metrics():
    cache_stats = rag_engine.cache_service.get_stats()
    metrics_data = rag_engine.metrics_service.get_metrics(cache_stats=cache_stats)
    metrics_data["recent_escalation_tickets"] = rag_engine.escalation_service.get_tickets(limit=10)
    metrics_data["indexed_vector_chunks"] = rag_engine.vector_store.get_document_count()
    return metrics_data

# Endpoint 5: System Health Check
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TechUni Admissions RAG Assistant API",
        "vector_store_chunks": rag_engine.vector_store.get_document_count(),
        "openai_key_configured": bool(rag_engine.api_key and not rag_engine.api_key.startswith("your_openai"))
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
