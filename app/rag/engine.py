import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

from app.rag.vector_store import VectorStoreManager
from app.prompts.system_prompts import format_prompt
from app.services.cache_service import CacheService
from app.services.metrics_service import MetricsService
from app.services.escalation_service import EscalationService

load_dotenv()
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Core RAG Pipeline Orchestrator connecting query input, semantic cache,
    ChromaDB retrieval, double-filter human escalation, LLM generation, and metrics.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", 0.1))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))
        self.base_url = os.getenv("OPENAI_BASE_URL", None)

        self.vector_store = VectorStoreManager()
        self.cache_service = CacheService()
        self.metrics_service = MetricsService()
        self.escalation_service = EscalationService(similarity_threshold=self.similarity_threshold)

    def process_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline for a student query.
        """
        start_time = time.time()
        query = query.strip()

        # Step 1: Check Semantic / Exact Cache
        cached_result = self.cache_service.get(query)
        if cached_result:
            cached_result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            self.metrics_service.record_query(escalated=cached_result.get("escalated", False))
            return cached_result

        # Step 2: Semantic Retrieval from ChromaDB
        retrieved_chunks = self.vector_store.search_similar(query, top_k=3)

        # Step 3: FILTER 1 (Pre-LLM Similarity Score Check)
        should_escalate_pre, pre_reason, max_similarity = self.escalation_service.evaluate_pre_llm(retrieved_chunks)
        
        if should_escalate_pre:
            ticket = self.escalation_service.create_ticket(
                query=query,
                reason=pre_reason,
                filter_type="Pre-LLM (Vector Search Similarity)",
                session_id=session_id
            )
            
            response_payload = {
                "answer": (
                    "I am the TechUni Admissions Assistant. I don't have enough specific information "
                    "in my university documents to answer your question accurately. I have routed your query "
                    "to a human admissions counselor for direct assistance. You may also contact admissions@techuni.edu."
                ),
                "escalated": True,
                "escalation_details": {
                    "stage": "Pre-LLM (Vector Search)",
                    "reason": pre_reason,
                    "max_similarity_score": max_similarity,
                    "ticket_id": ticket["ticket_id"]
                },
                "sources": [c["source"] for c in retrieved_chunks],
                "retrieved_chunks": retrieved_chunks,
                "cached": False,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "estimated_cost_usd": 0.0,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            self.metrics_service.record_query(escalated=True, pre_llm=True)
            self.cache_service.set(query, response_payload)
            return response_payload

        # Step 4: Build Context String & System Prompt
        context_str = "\n\n".join([
            f"[Source: {chunk['source']} | Similarity: {chunk['similarity']:.2f}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        prompt_text = format_prompt(query=query, context=context_str)

        # Step 5: Call LLM API (with fallback simulation if API key is unconfigured)
        raw_llm_response, usage_data = self._call_llm_api(prompt_text)

        # Step 6: FILTER 2 (Post-LLM Prefix Tag Detection)
        should_escalate_post, final_answer = self.escalation_service.evaluate_post_llm(raw_llm_response)

        ticket_id = None
        if should_escalate_post:
            ticket = self.escalation_service.create_ticket(
                query=query,
                reason="System prompt designated query as out-of-scope / unanswerable from context.",
                filter_type="Post-LLM (System Prompt Tag)",
                session_id=session_id
            )
            ticket_id = ticket["ticket_id"]

        # Step 7: Record Token Usage & Cost Calculation
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        query_cost = self.metrics_service.record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=self.model_name
        )

        self.metrics_service.record_query(escalated=should_escalate_post, pre_llm=False)

        response_payload = {
            "answer": final_answer,
            "escalated": should_escalate_post,
            "escalation_details": {
                "stage": "Post-LLM (System Prompt Tag)" if should_escalate_post else "None",
                "reason": "Out of scope request" if should_escalate_post else "N/A",
                "max_similarity_score": max_similarity,
                "ticket_id": ticket_id
            } if should_escalate_post else None,
            "sources": list(set(c["source"] for c in retrieved_chunks)),
            "retrieved_chunks": retrieved_chunks,
            "cached": False,
            "token_usage": usage_data,
            "estimated_cost_usd": query_cost,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

        # Save to cache
        self.cache_service.set(query, response_payload)
        return response_payload

    def _call_llm_api(self, prompt_text: str) -> Tuple[str, Dict[str, int]]:
        """
        Executes standard OpenAI API call and extracts text and response.usage object.
        Includes a local intelligent mock fallback if API key is not set.
        """
        if self.api_key and not self.api_key.startswith("your_openai"):
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url

                client = OpenAI(**kwargs)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": prompt_text}],
                    temperature=self.temperature
                )
                
                content = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
                return content, usage

            except Exception as e:
                logger.warning(f"LLM API call error ({e}). Returning fallback response.")

        # Fallback generator for offline execution / keyless evaluation
        return self._generate_fallback_llm_response(prompt_text)

    def _generate_fallback_llm_response(self, prompt_text: str) -> Tuple[str, Dict[str, int]]:
        """
        Provides intelligent mock generation when running without external API keys.
        Extracts relevant information from context or triggers [ESCALATE_TO_HUMAN] for out-of-scope topics.
        """
        prompt_tokens = len(prompt_text.split()) * 2
        
        query_part = prompt_text.split("STUDENT QUERY:")[1].strip() if "STUDENT QUERY:" in prompt_text else ""
        context_part = prompt_text.split("CONTEXT FROM TECHUNI BUSINESS DOCUMENTS:")[1] if "CONTEXT FROM TECHUNI BUSINESS DOCUMENTS:" in prompt_text else ""
        
        admissions_keywords = ["tuition", "fee", "cost", "scholarship", "grant", "program", "degree", "master", "bachelor", "schedule", "modalities", "online", "campus", "certif", "admiss", "apply", "requirement", "transfer", "credit", "bootcamp", "class", "course"]
        is_admissions_query = any(kw in query_part.lower() for kw in admissions_keywords)

        lines = [line.strip() for line in context_part.split("\n") if line.strip() and not line.startswith("[Source:") and not line.startswith("-")]
        relevant_lines = [l for l in lines if any(w.lower() in l.lower() for w in query_part.split() if len(w) > 3)]
        
        if is_admissions_query and relevant_lines:
            answer = "Based on TechUni official business documents:\n\n" + "\n".join(relevant_lines[:4])
        else:
            answer = (
                "[ESCALATE_TO_HUMAN] I am the TechUni Admissions Assistant. Your request appears to be "
                "outside the scope of our university business documents. I have escalated this to a human admissions counselor."
            )

        completion_tokens = len(answer.split()) * 2
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        return answer, usage
