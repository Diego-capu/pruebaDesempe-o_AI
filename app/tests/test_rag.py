import os
import pytest
from dotenv import load_dotenv
from app.rag.ingest import ingest_documents
from app.rag.engine import RAGEngine

load_dotenv()

@pytest.fixture(scope="module")
def setup_rag():
    # Ingest documents before tests
    ingest_documents()
    engine = RAGEngine()
    return engine

def test_vector_store_ingestion(setup_rag):
    engine = setup_rag
    count = engine.vector_store.get_document_count()
    assert count > 0, "Vector store should contain indexed chunks after ingestion."

def test_in_scope_tuition_query(setup_rag):
    engine = setup_rag
    query = "How much is full-time undergraduate tuition per semester?"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Valid tuition query should not be escalated."
    assert len(res["retrieved_chunks"]) > 0, "Should retrieve relevant context chunks."
    assert any("4,200" in chunk["text"] or "tuition" in chunk["text"].lower() for chunk in res["retrieved_chunks"]), "Retrieved context should contain tuition info."
    assert "answer" in res and len(res["answer"]) > 10, "Response should contain an answer."

def test_in_scope_program_query(setup_rag):
    engine = setup_rag
    query = "What Master of Science programs are offered?"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Valid program query should not be escalated."
    assert len(res["retrieved_chunks"]) > 0, "Should retrieve degree program chunks."

def test_irrelevant_query_no_escalation(setup_rag):
    engine = setup_rag
    query = "What is the recipe for baking chocolate fudge cake?"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Irrelevant query (recipes/spam) must NOT be escalated."
    assert res["escalation_details"] is None, "Escalation details should be None for irrelevant queries."

def test_legitimate_out_of_scope_escalation(setup_rag):
    engine = setup_rag
    query = "Do you offer corporate discounts for groups of 20+ engineers?"
    res = engine.process_query(query)
    
    assert res["escalated"] is True, "Legitimate out-of-scope query should trigger escalation."
    assert res["escalation_details"] is not None, "Escalation details should be populated."
    assert res["escalation_details"]["ticket_id"] is not None, "A human escalation ticket must be generated."

def test_semantic_cache_hit(setup_rag):
    engine = setup_rag
    query = "Are there scholarships for women in STEM?"
    
    # First query (Cache miss)
    res1 = engine.process_query(query)
    assert res1["cached"] is False
    
    # Second query (Cache hit)
    res2 = engine.process_query(query)
    assert res2["cached"] is True
    assert res2["answer"] == res1["answer"]

def test_metrics_service_accumulation(setup_rag):
    engine = setup_rag
    metrics = engine.metrics_service.get_metrics()
    
    assert metrics["total_queries_processed"] > 0, "Total queries counter should be greater than zero."
    assert "token_usage" in metrics, "Token usage metrics should be present."
    assert "financial_metrics" in metrics, "Financial metrics should be present."
    assert "total_tokens_used" in metrics, "total_tokens_used must be exposed."
    assert "prompt_tokens" in metrics, "prompt_tokens must be exposed."
    assert "completion_tokens" in metrics, "completion_tokens must be exposed."
    assert "token_limit_monthly" in metrics, "token_limit_monthly must be exposed."
    assert "token_usage_percentage" in metrics, "token_usage_percentage must be exposed."
    assert metrics["total_tokens_used"] >= 0
    assert metrics["token_limit_monthly"] == 1000000

def test_streaming_and_suggested_chips(setup_rag):
    import json
    engine = setup_rag
    query = "How much is undergraduate tuition per semester?"
    
    events = list(engine.process_query_stream(query))
    assert len(events) >= 2, "Stream should yield at least one token and a done event."
    
    # Verify done event structure
    done_event = None
    for ev in events:
        if ev.startswith("data: "):
            payload = json.loads(ev[6:].strip())
            if payload.get("type") == "done":
                done_event = payload
                break
                
    assert done_event is not None, "Stream must emit a 'done' event."
    assert "suggested_chips" in done_event, "Done event must include suggested_chips."
    assert len(done_event["suggested_chips"]) == 3, "Should provide 3 suggested quick reply chips."
    assert "answer" in done_event and len(done_event["answer"]) > 0, "Done event should have non-empty answer."
