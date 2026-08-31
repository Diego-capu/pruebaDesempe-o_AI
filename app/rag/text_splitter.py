import re
from typing import List, Dict, Any

class RecursiveCharacterTextSplitter:
    """
    Custom text splitter that splits text into chunks with specified chunk_size and chunk_overlap.
    Maintains document structure by attempting to split on section headers and paragraph breaks first.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source_doc: str = "") -> List[Dict[str, Any]]:
        """
        Splits string content into overlapping chunks with metadata.
        """
        # Clean text
        text = text.strip()
        if not text:
            return []

        # Split into initial paragraphs/sections
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks: List[Dict[str, Any]] = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            # If paragraph itself exceeds chunk size, split by sentences or fixed character window
            if len(para) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                        current_chunk = f"{current_chunk} {sentence}".strip()
                    else:
                        if current_chunk:
                            chunks.append({
                                "chunk_id": f"{source_doc}_chunk_{chunk_index}",
                                "text": current_chunk,
                                "source": source_doc
                            })
                            chunk_index += 1
                            # Maintain overlap
                            overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                            current_chunk = current_chunk[overlap_start:] + " " + sentence
                        else:
                            # Hard split if sentence is longer than chunk_size
                            for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                                sub = sentence[i:i + self.chunk_size]
                                chunks.append({
                                    "chunk_id": f"{source_doc}_chunk_{chunk_index}",
                                    "text": sub,
                                    "source": source_doc
                                })
                                chunk_index += 1
                            current_chunk = ""
            else:
                if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                    current_chunk = f"{current_chunk}\n\n{para}".strip()
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_id": f"{source_doc}_chunk_{chunk_index}",
                            "text": current_chunk,
                            "source": source_doc
                        })
                        chunk_index += 1
                        # Build overlap context from end of current chunk
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                    else:
                        current_chunk = para

        if current_chunk:
            chunks.append({
                "chunk_id": f"{source_doc}_chunk_{chunk_index}",
                "text": current_chunk,
                "source": source_doc
            })

        return chunks
