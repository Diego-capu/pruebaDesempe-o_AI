import os
import math
import logging
from typing import List, Dict, Any
import chromadb
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages vector database storage, indexing, and semantic similarity retrieval using ChromaDB.
    Supports OpenAI API embeddings with fallback to lightweight normalized TF-IDF vectorizer if needed.
    """

    def __init__(self, chroma_dir: str = "./data/chroma_db", collection_name: str = "university_admissions"):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.base_url = os.getenv("OPENAI_BASE_URL", None)

        os.makedirs(self.chroma_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generates embedding vector for a given text.
        Tries OpenAI Embeddings API first, falls back to lightweight deterministic vectorizer.
        """
        if self.api_key and not self.api_key.startswith("your_openai"):
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                client = OpenAI(**kwargs)
                response = client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return response.data[0].embedding
            except Exception as e:
                logger.warning(f"OpenAI Embedding call failed ({e}). Falling back to local vectorizer.")

        # Local deterministic fallback embedding (384-dim hash vector normalized)
        return self._local_fallback_embedding(text)

    def _local_fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Lightweight fallback embedding generator when API key is offline or unavailable.
        Uses deterministic zlib.crc32 hashing with bilingual (Spanish-English) concept expansion.
        """
        import zlib
        
        SPANISH_TO_ENGLISH_MAP = {
            "matricula": ["tuition", "registration", "fee", "enrollment"],
            "matrícula": ["tuition", "registration", "fee", "enrollment"],
            "costo": ["tuition", "cost", "fee", "price"],
            "costos": ["tuition", "cost", "fee", "price"],
            "precio": ["tuition", "cost", "fee", "price"],
            "precios": ["tuition", "cost", "fee", "price"],
            "arancel": ["tuition", "fee"],
            "aranceles": ["tuition", "fee"],
            "semestre": ["semester", "term", "academic"],
            "semestres": ["semester", "term", "academic"],
            "beca": ["scholarship", "grant", "financial", "aid"],
            "becas": ["scholarship", "grant", "financial", "aid"],
            "ayuda": ["financial", "aid", "scholarship"],
            "ayudas": ["financial", "aid", "scholarship"],
            "economica": ["financial", "economic"],
            "económica": ["financial", "economic"],
            "economicas": ["financial", "economic"],
            "económicas": ["financial", "economic"],
            "financiera": ["financial", "aid"],
            "financieras": ["financial", "aid"],
            "financiamiento": ["installment", "payment", "financial"],
            "pago": ["payment", "installment", "pay"],
            "pagos": ["payment", "installment", "pay"],
            "cuota": ["installment", "payment", "monthly"],
            "cuotas": ["installment", "payment", "monthly"],
            "descuento": ["discount", "reduction"],
            "descuentos": ["discount", "reduction"],
            "carrera": ["degree", "program", "bachelor", "master"],
            "carreras": ["degree", "program", "bachelor", "master"],
            "programa": ["program", "degree", "course"],
            "programas": ["program", "degree", "course"],
            "pregrado": ["undergraduate", "bachelor", "bsc"],
            "licenciatura": ["undergraduate", "bachelor", "bsc"],
            "licenciaturas": ["undergraduate", "bachelor", "bsc"],
            "maestria": ["master", "msc", "graduate"],
            "maestría": ["master", "msc", "graduate"],
            "maestrias": ["master", "msc", "graduate"],
            "maestrías": ["master", "msc", "graduate"],
            "posgrado": ["postgraduate", "master", "msc"],
            "postgrado": ["postgraduate", "master", "msc"],
            "bootcamp": ["bootcamp", "certificate"],
            "bootcamps": ["bootcamp", "certificate"],
            "modalidad": ["modality", "modalities", "online", "campus", "hybrid"],
            "modalidades": ["modality", "modalities", "online", "campus", "hybrid"],
            "estudio": ["study", "academic"],
            "estudios": ["study", "academic"],
            "presencial": ["campus", "presential"],
            "remoto": ["online", "remote"],
            "virtual": ["online", "virtual"],
            "horario": ["schedule", "shift", "hours"],
            "horarios": ["schedule", "shift", "hours"],
            "turno": ["shift", "schedule"],
            "turnos": ["shift", "schedule"],
            "requisito": ["requirement", "requirements", "admission", "transcripts"],
            "requisitos": ["requirement", "requirements", "admission", "transcripts"],
            "admision": ["admission", "admissions", "apply", "application"],
            "admisión": ["admission", "admissions", "apply", "application"],
            "admisiones": ["admission", "admissions", "apply", "application"],
            "inscripcion": ["registration", "enrollment", "application"],
            "inscripción": ["registration", "enrollment", "application"],
            "inscripciones": ["registration", "enrollment", "application"],
            "postulacion": ["application", "apply"],
            "postulación": ["application", "apply"],
            "postulaciones": ["application", "apply"],
            "postular": ["apply", "application"],
            "fecha": ["deadline", "date", "calendar"],
            "fechas": ["deadline", "date", "calendar"],
            "limite": ["deadline"],
            "límite": ["deadline"],
            "limites": ["deadline"],
            "límites": ["deadline"],
            "transferencia": ["transfer", "credit"],
            "credito": ["credit", "hours"],
            "crédito": ["credit", "hours"],
            "creditos": ["credit", "hours"],
            "créditos": ["credit", "hours"],
            "certificacion": ["certification", "certifications", "cisco", "aws", "google", "nvidia"],
            "certificación": ["certification", "certifications", "cisco", "aws", "google", "nvidia"],
            "certificaciones": ["certification", "certifications", "cisco", "aws", "google", "nvidia"],
            "inteligencia": ["artificial", "intelligence", "ai", "data"],
            "artificial": ["artificial", "intelligence", "ai"],
            "ciberseguridad": ["cybersecurity", "cybercloud", "security", "defense"],
            "nube": ["cloud", "systems", "aws", "azure"],
            "software": ["software", "architecture"],
            "datos": ["data", "engineering", "analytics"],
            "stem": ["stem", "women", "leadership", "innovators"],
            "mujeres": ["women", "stem", "female"]
        }

        vec = [0.0] * dim
        raw_words = text.lower().replace("?", " ").replace("¿", " ").replace("(", " ").replace(")", " ").replace("/", " ").replace(",", " ").split()
        
        # Expand tokens
        all_tokens = []
        for word in raw_words:
            all_tokens.append(word)
            if word in SPANISH_TO_ENGLISH_MAP:
                all_tokens.extend(SPANISH_TO_ENGLISH_MAP[word])

        for i, token in enumerate(all_tokens):
            h = zlib.crc32(token.encode("utf-8")) % dim
            vec[h] += 1.0
            if i < len(all_tokens) - 1:
                bigram = f"{all_tokens[i]}_{all_tokens[i+1]}"
                bigram_h = zlib.crc32(bigram.encode("utf-8")) % dim
                vec[bigram_h] += 0.5

        # L2 normalize
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude > 0:
            vec = [v / magnitude for v in vec]
        return vec

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Indexes chunks into ChromaDB vector store.
        """
        if not chunks:
            return

        documents = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"source": c["source"]} for c in chunks]
        embeddings = [self._get_embedding(c["text"]) for c in chunks]

        # Upsert into ChromaDB
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
                logger.warning(f"Embedding dimension mismatch ({e}). Recreating collection '{self.collection_name}'...")
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                raise e
        logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB.")

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search against ChromaDB.
        Returns retrieved chunks with cosine similarity score (0.0 to 1.0, where 1.0 is exact match).
        """
        count = self.collection.count()
        if count == 0:
            return []

        query_vec = self._get_embedding(query)
        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
                logger.warning(f"Embedding dimension mismatch on search ({e}). Returning empty results.")
                return []
            raise e

        retrieved_chunks = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                # Cosine distance in ChromaDB is d = 1 - similarity.
                # Convert distance to similarity score: similarity = 1 - distance
                # Ensure bounded between 0.0 and 1.0
                similarity = max(0.0, min(1.0, 1.0 - dist))
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "distance": round(float(dist), 4),
                    "similarity": round(float(similarity), 4)
                })

        return retrieved_chunks

    def get_document_count(self) -> int:
        return self.collection.count()
