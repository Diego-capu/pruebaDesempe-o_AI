import time
import uuid
from typing import List, Dict, Any, Tuple, Optional

class EscalationService:
    """
    Handles the double-filter human escalation pipeline and tickets management.
    Filter 1 (Pre-LLM): Checks ChromaDB similarity distance score. If no relevant chunk matches, escalates immediately without API call.
    Filter 2 (Post-LLM): Scans LLM response for [ESCALATE_TO_HUMAN] prefix tag.
    """

    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold
        self.tickets: List[Dict[str, Any]] = []

    def evaluate_pre_llm(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
        """
        Filter 1: Evaluates retrieved chunks before calling the LLM.
        Returns: (should_escalate: bool, reason: str, max_similarity: float)
        """
        if not retrieved_chunks:
            return True, "No document chunks retrieved from vector store.", 0.0

        max_similarity = max(chunk.get("similarity", 0.0) for chunk in retrieved_chunks)

        if max_similarity < self.similarity_threshold:
            return (
                True,
                f"Retrieved context similarity ({max_similarity:.4f}) is below confidence threshold ({self.similarity_threshold:.4f}).",
                max_similarity
            )

        return False, "", max_similarity

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
