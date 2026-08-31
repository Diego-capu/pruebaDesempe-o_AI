import os
import logging
from app.rag.text_splitter import RecursiveCharacterTextSplitter
from app.rag.vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_documents(docs_dir: str = "./data/documents", chunk_size: int = 500, chunk_overlap: int = 100) -> int:
    """
    Ingests all text documents from docs_dir into ChromaDB vector database.
    Returns total number of ingested chunks.
    """
    if not os.path.exists(docs_dir):
        logger.error(f"Documents directory '{docs_dir}' does not exist.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store = VectorStoreManager()

    all_chunks = []
    doc_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt") or f.endswith(".md")]

    logger.info(f"Found {len(doc_files)} business documents in '{docs_dir}'...")

    for filename in doc_files:
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content, source_doc=filename)
        logger.info(f"Document '{filename}': Created {len(chunks)} chunks with chunk_size={chunk_size}, overlap={chunk_overlap}.")
        all_chunks.extend(chunks)

    if all_chunks:
        vector_store.add_documents(all_chunks)
        logger.info(f"Ingestion Complete! Total {len(all_chunks)} chunks stored in ChromaDB.")
    else:
        logger.warning("No document content found to ingest.")

    return len(all_chunks)

if __name__ == "__main__":
    chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    ingest_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
