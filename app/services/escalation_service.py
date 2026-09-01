import time
import uuid
from typing import List, Dict, Any, Tuple

class EscalationService:
    """
    Handles the double-filter human escalation pipeline and tickets management.
    Filter 1 (Context Evaluation): Filters ChromaDB retrieved chunks based on similarity confidence threshold.
    Filter 2 (Post-LLM): Scans LLM response for [ESCALATE_TO_HUMAN] prefix tag.
    """

    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold
        self.tickets: List[Dict[str, Any]] = []

    def filter_relevant_chunks(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Filters retrieved chunks based on similarity threshold without directly escalating.
        Returns: (relevant_chunks: List, max_similarity: float)
        """
        if not retrieved_chunks:
            return [], 0.0
        max_similarity = max((chunk.get("similarity", 0.0) for chunk in retrieved_chunks), default=0.0)
        relevant = [c for c in retrieved_chunks if c.get("similarity", 0.0) >= self.similarity_threshold]
        return relevant, max_similarity

    def evaluate_post_llm(self, llm_output: str) -> Tuple[bool, str]:
        """
        Filter 2: Scans LLM output for explicit escalation prefix tag.
        Returns: (is_escalated: bool, cleaned_text: str)
        """
        tag = "[ESCALATE_TO_HUMAN]"
        if tag in llm_output:
            cleaned = llm_output.replace(tag, "").strip()
            return True, cleaned
        
        return False, llm_output.strip()

    def create_ticket(self, query: str, reason: str, filter_type: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Generates and registers an escalation ticket for human counselor follow-up.
        """
        ticket = {
            "ticket_id": f"TICK-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "user_query": query,
            "escalation_reason": reason,
            "filter_stage": filter_type,  # "Pre-LLM (Vector Search)" or "Post-LLM (System Prompt)"
            "status": "PENDING_HUMAN_COUNSELOR"
        }
        self.tickets.append(ticket)
        return ticket

    def get_tickets(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.tickets[-limit:]
