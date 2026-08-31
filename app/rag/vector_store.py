import os
import math
import logging
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
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
                client = OpenAI(api_key=self.api_key)
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
        Uses deterministic zlib.crc32 hashing to produce cosine-comparable dense vectors.
        """
        import zlib
        vec = [0.0] * dim
        words = text.lower().split()
        for i, word in enumerate(words):
            h = zlib.crc32(word.encode("utf-8")) % dim
            vec[h] += 1.0
            if i < len(words) - 1:
                bigram = f"{words[i]}_{words[i+1]}"
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
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
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
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )

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
