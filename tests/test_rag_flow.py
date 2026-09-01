import pytest
from app.rag.engine import RAGEngine

@pytest.fixture(scope="module")
def engine():
    return RAGEngine()

def test_greeting_bypass_pre_llm(engine):
    """
    Test 1: Hello! -> Bypass Pre-LLM, Saludo cordial (0 tokens de contexto, sin ticket).
    """
    query = "Hello!"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Greeting must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated for greetings."
    assert len(res["retrieved_chunks"]) == 0, "Greetings should bypass vector store retrieval (0 chunks)."
    assert len(res["sources"]) == 0, "Greetings should have 0 sources."
    assert any(w in res["answer"].lower() for w in ["hello", "welcome", "assist", "techuni"]), "Response must be a friendly greeting."

def test_general_intent_sign_up(engine):
    """
    Test 2: 'I want to sign up' -> RAG + Orientación: Explica pasos generales, lista carreras/posgrados y pide elegir opción (Sin ticket).
    """
    query = "I want to sign up"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "General intent to sign up must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated."
    answer = res["answer"].lower()
    assert "undergraduate" in answer or "master" in answer or "bootcamp" in answer, "Response should outline available academic tracks."

def test_specific_data_msc_tuition(engine):
    """
    Test 3: 'What is the tuition for the M.Sc.?' -> RAG Preciso: $3,600 USD per semester ($14,400 USD total) sintetizado fluidamente.
    """
    query = "What is the tuition for the M.Sc.?"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Specific tuition inquiry must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated."
    assert "3,600" in res["answer"], "Response must state $3,600 USD tuition per semester."

def test_legitimate_out_of_scope_civil_engineering(engine):
    """
    Test 4: 'Do you offer Civil Engineering?' -> Escalamiento Legítimo: [ESCALATE_TO_HUMAN] + Notificación de asesor.
    """
    query = "Do you offer Civil Engineering?"
    res = engine.process_query(query)
    
    assert res["escalated"] is True, "Legitimate unlisted degree inquiry should be escalated."
    assert res["escalation_details"] is not None, "Escalation details must be present."
    assert res["escalation_details"]["ticket_id"] is not None, "A human escalation ticket ID must be generated."

def test_spam_irrelevant_pizza(engine):
    """
    Test 5: 'pizza' -> Rechazo Educado: Delimita función a admisiones (Sin ticket).
    """
    query = "pizza"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "Spam/food queries must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated for spam."
    assert any(w in res["answer"].lower() for w in ["assist", "admissions", "programs", "university"]), "Must politely delimit role boundaries."

def test_academic_programs_available_no_escalation(engine):
    """
    Test 6: 'Academic Programs Available' -> In-Scope Overview, No Escalation, lists degree programs.
    """
    query = "Academic Programs Available"
    res = engine.process_query(query)
    
    assert res["escalated"] is False, "'Academic Programs Available' must NOT be escalated."
    assert res["escalation_details"] is None, "No escalation ticket should be generated."
    answer = res["answer"].lower()
    assert "artificial intelligence" in answer or "cybersecurity" in answer or "software architecture" in answer, "Response should list degree programs."


