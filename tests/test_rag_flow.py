import pytest
from app.rag.engine import RAGEngine

@pytest.fixture(scope="module")
def engine():
    return RAGEngine()

def test_general_intent_sign_up(engine):
    """
    Smoke Test 1: General Intent
    'I want to sign up' -> Orientative response + program options (No ticket)
    """
    query = "I want to sign up"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "General intent to sign up must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated."
    answer = res["answer"].lower()
    assert "undergraduate" in answer or "master" in answer or "bootcamp" in answer, "Response should outline available academic tracks."

def test_specific_data_msc_tuition(engine):
    """
    Smoke Test 2: Specific Data
    'How much is the M.Sc. per semester?' -> Responds with ,600 USD (No ticket)
    """
    query = "How much is the M.Sc. per semester?"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Specific tuition inquiry must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated."
    assert "3,600" in res["answer"], "Response must state ,600 USD tuition per semester."

def test_greeting_no_vector_store(engine):
    """
    Smoke Test 3: Greeting
    'Hello good morning' -> Cordial greeting + 0 vector store chunks (No ticket)
    """
    query = "Hello good morning"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Greeting must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated."
    assert len(res["retrieved_chunks"]) == 0, "Greetings should bypass vector store search (0 chunks)."
    assert len(res["sources"]) == 0, "Greetings should have 0 sources."
    assert any(g in res["answer"].lower() for g in ["hello", "welcome", "assist"]), "Must return a cordial greeting."

def test_legitimate_out_of_scope_aerospace(engine):
    """
    Smoke Test 4: Legitimate Out-of-Scope
    'Do you offer Aerospace Engineering?' -> [ESCALATE_TO_HUMAN] + Ticket generated
    """
    query = "Do you offer Aerospace Engineering?"
    res = engine.process_query(query)
    
    assert res["escalated"] is True, "Legitimate missing program inquiry should be escalated."
    assert res["escalation_details"] is not None, "Escalation details must be present."
    assert res["escalation_details"]["ticket_id"] is not None, "A human escalation ticket ID must be generated."

def test_spam_irrelevant_pizza(engine):
    """
    Smoke Test 5: Spam / Irrelevant
    'pizza' -> Cordial rejection + role boundary delimiter (No ticket)
    """
    query = "pizza"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Spam/food queries must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated for spam."
    assert any(w in res["answer"].lower() for w in ["assist", "admissions", "programs", "university"]), "Must politely delimit role boundaries."
