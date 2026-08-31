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

def test_out_of_scope_pre_llm_escalation(setup_rag):
    engine = setup_rag
    query = "What is the recipe for baking chocolate fudge cake?"
    res = engine.process_query(query)
    
    assert res["escalated"] is True, "Out-of-scope query should trigger escalation."
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
