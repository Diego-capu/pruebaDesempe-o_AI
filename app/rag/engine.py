import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Generator
from dotenv import load_dotenv

from app.rag.vector_store import VectorStoreManager
from app.prompts.system_prompts import SYSTEM_PROMPT, build_rag_prompt, format_prompt
from app.services.cache_service import CacheService
from app.services.metrics_service import MetricsService
from app.services.escalation_service import EscalationService

load_dotenv()
logger = logging.getLogger(__name__)

def generate_suggested_chips(query: str, answer: str = "") -> List[str]:
    """
    Generates 3 contextual suggested quick-reply chips based on user query and response.
    """
    q = query.lower()
    a = answer.lower()
    
    is_spanish = any(w in q for w in ["cuanto", "precio", "costo", "beca", "pago", "horario", "carrera", "programa", "requisito", "hola", "inscripc", "que", "como", "donde"]) or any(w in a for w in ["matrícula", "programa", "arancel", "horario", "academia"])

    if any(w in q for w in ["tuition", "fee", "cost", "price", "matricula", "precio", "costo", "arancel", "cuota", "pagar", "pay"]):
        if is_spanish:
            return ["Becas y ayudas económicas", "Planes de pago en cuotas", "Fechas límite de matrícula"]
        return ["Scholarships & Financial Aid", "Monthly Installment Plans", "Registration Deadlines"]

    if any(w in q for w in ["scholarship", "grant", "aid", "beca", "descuento", "discount"]):
        if is_spanish:
            return ["Requisitos para becas STEM", "Costos de matrícula por semestre", "Proceso de postulación"]
        return ["Women in STEM Grant Requirements", "Tuition Fees per Semester", "Application Process"]

    if any(w in q for w in ["program", "degree", "master", "bachelor", "course", "carrera", "licenciatura", "maestria", "curso", "bootcamp"]):
        if is_spanish:
            return ["Modalidades de estudio (Online/Presencial)", "Horarios y turnos de clases", "Requisitos de admisión"]
        return ["Study Modalities (Online/Hybrid)", "Class Schedules & Shifts", "Admissions Requirements"]

    if any(w in q for w in ["schedule", "shift", "time", "hour", "horario", "turno", "clase", "noche", "weekend"]):
        if is_spanish:
            return ["Modalidad 100% Online", "Costos de programas", "Fechas de inicio de clases"]
        return ["100% Online Modality", "Tuition & Program Costs", "Academic Calendar Dates"]

    if any(w in q for w in ["admiss", "apply", "requirement", "transfer", "requisito", "ingreso", "inscripc", "postular", "document"]):
        if is_spanish:
            return ["Costos de matrícula y aranceles", "Fechas de examen de admisión", "Transferencia de créditos"]
        return ["Tuition Fees & Deposit", "Admissions Exam Details", "Transfer Credit Policies"]

    if any(w in q for w in ["certif", "cisco", "aws", "google", "nvidia", "certificacion", "toefl", "ielts"]):
        if is_spanish:
            return ["Programas de grado ofrecidos", "Costos de certificación", "Modalidades de estudio"]
        return ["Degree Programs Included", "Tuition Fees & Bootcamps", "Study Modalities"]

    if any(w in q for w in ["hello", "hi", "hey", "hola", "buenos", "good morning", "saludos"]):
        if is_spanish:
            return ["Ver programas académicos", "Costos de matrícula", "Proceso de inscripción"]
        return ["Undergraduate Programs", "Tuition & Payment Plans", "Admissions Process"]

    if is_spanish:
        return ["Programas académicos disponibles", "Costos y opciones de pago", "Requisitos de admisión"]
    return ["Academic Programs Available", "Tuition & Payment Plans", "Admissions Requirements"]

class RAGEngine:
    """
    Core RAG Pipeline Orchestrator connecting query input, semantic cache,
    ChromaDB retrieval, double-filter human escalation, LLM generation, streaming, and metrics.
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
        Executes end-to-end synchronous RAG pipeline for a student query.
        """
        start_time = time.time()
        query = query.strip()

        # Step 1: Check Semantic / Exact Cache
        cached_result = self.cache_service.get(query)
        if cached_result:
            cached_result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            if "suggested_chips" not in cached_result:
                cached_result["suggested_chips"] = generate_suggested_chips(query, cached_result.get("answer", ""))
            self.metrics_service.record_query(escalated=cached_result.get("escalated", False))
            return cached_result

        # Step 2: Semantic Retrieval from ChromaDB
        retrieved_chunks = self.vector_store.search_similar(query, top_k=3)

        # Step 3: Filter relevant chunks by confidence threshold (allow empty context for LLM evaluation)
        relevant_chunks, max_similarity = self.escalation_service.filter_relevant_chunks(retrieved_chunks)

        # Step 4: Build Context Chunks & User Prompt
        context_chunks = [
            f"[Source: {chunk['source']} | Similarity: {chunk['similarity']:.2f}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ]
        prompt_text = build_rag_prompt(user_query=query, context_chunks=context_chunks)

        # Step 5: Call LLM API (with fallback simulation if API key is unconfigured)
        raw_llm_response, usage_data = self._call_llm_api(prompt_text)

        # Step 6: FILTER (Post-LLM Prefix Tag Detection)
        should_escalate_post, final_answer = self.escalation_service.evaluate_post_llm(raw_llm_response)

        ticket_id = None
        if should_escalate_post:
            ticket = self.escalation_service.create_ticket(
                query=query,
                reason="Legitimate academic/admissions query not found in official documents.",
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

        suggested_chips = generate_suggested_chips(query, final_answer)

        response_payload = {
            "answer": final_answer,
            "escalated": should_escalate_post,
            "escalation_details": {
                "stage": "Post-LLM (System Prompt Tag)" if should_escalate_post else "None",
                "reason": "Legitimate request requiring human advisor follow-up" if should_escalate_post else "N/A",
                "max_similarity_score": max_similarity,
                "ticket_id": ticket_id
            } if should_escalate_post else None,
            "sources": list(set(c["source"] for c in relevant_chunks)) if relevant_chunks else [],
            "retrieved_chunks": relevant_chunks,
            "suggested_chips": suggested_chips,
            "cached": False,
            "token_usage": usage_data,
            "estimated_cost_usd": query_cost,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

        # Save to cache
        self.cache_service.set(query, response_payload)
        return response_payload

    def process_query_stream(self, query: str, session_id: str = "default") -> Generator[str, None, None]:
        """
        Executes end-to-end streaming RAG pipeline with Server-Sent Events (SSE).
        Yields progressive tokens and finishes with a structured 'done' payload.
        """
        start_time = time.time()
        query = query.strip()

        # Step 1: Check Semantic / Exact Cache
        cached_result = self.cache_service.get(query)
        if cached_result:
            cached_result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            if "suggested_chips" not in cached_result:
                cached_result["suggested_chips"] = generate_suggested_chips(query, cached_result.get("answer", ""))
            cached_result["cached"] = True
            self.metrics_service.record_query(escalated=cached_result.get("escalated", False))
            
            # Emit token stream for cached answer
            yield f"data: {json.dumps({'type': 'token', 'content': cached_result['answer']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', **cached_result})}\n\n"
            return

        # Step 2: Semantic Retrieval from ChromaDB
        retrieved_chunks = self.vector_store.search_similar(query, top_k=3)

        # Step 3: Filter relevant chunks
        relevant_chunks, max_similarity = self.escalation_service.filter_relevant_chunks(retrieved_chunks)

        # Step 4: Build Context Chunks & User Prompt
        context_chunks = [
            f"[Source: {chunk['source']} | Similarity: {chunk['similarity']:.2f}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ]
        prompt_text = build_rag_prompt(user_query=query, context_chunks=context_chunks)

        # Step 5: Streaming LLM API Call
        accumulated_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        if self.api_key and not self.api_key.startswith("your_openai"):
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url

                client = OpenAI(**kwargs)
                stream_response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=self.temperature,
                    stream=True,
                    stream_options={"include_usage": True}
                )

                for chunk in stream_response:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        accumulated_text += token
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens or completion_tokens

            except Exception as e:
                logger.warning(f"Streaming LLM error ({e}). Using fallback generator.")
                accumulated_text = ""

        # Fallback if streaming failed or key is unconfigured
        if not accumulated_text:
            fallback_answer, fallback_usage = self._generate_fallback_llm_response(prompt_text)
            accumulated_text = fallback_answer
            prompt_tokens = fallback_usage.get("prompt_tokens", 0)
            completion_tokens = fallback_usage.get("completion_tokens", 0)
            yield f"data: {json.dumps({'type': 'token', 'content': fallback_answer})}\n\n"

        # Calculate tokens if not provided by stream
        if prompt_tokens == 0:
            prompt_tokens = len(prompt_text.split()) * 2
        if completion_tokens == 0:
            completion_tokens = len(accumulated_text.split()) * 2
        total_tokens = prompt_tokens + completion_tokens

        # Step 6: Evaluate Post-LLM Escalation Tag
        should_escalate_post, final_answer = self.escalation_service.evaluate_post_llm(accumulated_text)

        ticket_id = None
        if should_escalate_post:
            ticket = self.escalation_service.create_ticket(
                query=query,
                reason="Legitimate academic/admissions query not found in official documents.",
                filter_type="Post-LLM (System Prompt Tag)",
                session_id=session_id
            )
            ticket_id = ticket["ticket_id"]

        # Step 7: Record Metrics
        query_cost = self.metrics_service.record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=self.model_name
        )
        self.metrics_service.record_query(escalated=should_escalate_post, pre_llm=False)

        suggested_chips = generate_suggested_chips(query, final_answer)

        response_payload = {
            "answer": final_answer,
            "escalated": should_escalate_post,
            "escalation_details": {
                "stage": "Post-LLM (System Prompt Tag)" if should_escalate_post else "None",
                "reason": "Legitimate request requiring human advisor follow-up" if should_escalate_post else "N/A",
                "max_similarity_score": max_similarity,
                "ticket_id": ticket_id
            } if should_escalate_post else None,
            "sources": list(set(c["source"] for c in relevant_chunks)) if relevant_chunks else [],
            "retrieved_chunks": relevant_chunks,
            "suggested_chips": suggested_chips,
            "cached": False,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "estimated_cost_usd": query_cost,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }

        # Cache response
        self.cache_service.set(query, response_payload)

        # Final done event
        yield f"data: {json.dumps({'type': 'done', **response_payload})}\n\n"

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
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text}
                    ],
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
        Extracts relevant information from context, rejects non-academic/spam politely,
        or triggers [ESCALATE_TO_HUMAN] for legitimate missing academic topics.
        """
        prompt_tokens = len(prompt_text.split()) * 2
        
        query_part = ""
        if "User Inquiry:" in prompt_text:
            query_part = prompt_text.split("User Inquiry:")[1].split("Assistant Answer:")[0].strip()
        elif "STUDENT QUERY:" in prompt_text:
            query_part = prompt_text.split("STUDENT QUERY:")[1].strip()

        context_part = ""
        if "[CONTEXT]" in prompt_text and "[/CONTEXT]" in prompt_text:
            context_part = prompt_text.split("[CONTEXT]")[1].split("[/CONTEXT]")[0].strip()

        query_lower = query_part.lower()

        # Check greetings
        greetings = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey", "hola", "buenos dias", "thanks", "thank you", "gracias"]
        if any(query_lower.startswith(g) or query_lower == g for g in greetings):
            answer = "Hello! Welcome to our Language Academy admissions support. I can assist you with information about our language courses, class schedules, tuition fees, placement tests, and official certifications. How can I help you today?"
        
        # Check irrelevant / non-academic / spam / food topics
        elif any(w in query_lower for w in ["pizza", "recipe", "receta", "joke", "chiste", "cake", "cook", "food", "weather", "clima", "pasta", "football", "futbol", "burger"]):
            answer = "I can only assist with academic and admissions inquiries for our Language Academy (courses, schedules, fees, levels, and enrollment). How can I assist you with your studies today?"
        
        else:
            admissions_keywords = ["tuition", "fee", "cost", "scholarship", "grant", "program", "degree", "master", "bachelor", "schedule", "modalities", "online", "campus", "certif", "admiss", "apply", "requirement", "transfer", "credit", "bootcamp", "class", "course", "level", "language", "english", "french", "german", "toefl", "ielts"]
            is_admissions_query = any(kw in query_lower for kw in admissions_keywords)

            lines = [line.strip() for line in context_part.split("\n") if line.strip() and not line.startswith("[Source:") and not line.startswith("-") and "No relevant documents" not in line]
            relevant_lines = [l for l in lines if any(w in l.lower() for w in query_lower.split() if len(w) > 3)]
            
            if is_admissions_query and relevant_lines:
                answer = "Based on official Language Academy business documents:\n\n" + "\n".join(relevant_lines[:4])
            elif is_admissions_query:
                answer = (
                    "[ESCALATE_TO_HUMAN] We currently do not have specific details regarding this request "
                    "in our standard documents. I have forwarded your request to an academic advisor to assist you directly."
                )
            else:
                answer = "I can only assist with academic and admissions inquiries for our Language Academy (courses, schedules, fees, levels, and enrollment). How can I assist you with your studies today?"

        completion_tokens = len(answer.split()) * 2
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
        return answer, usage
