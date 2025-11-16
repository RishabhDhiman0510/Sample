"""Text chunking and normalization."""
import re
from typing import List, Tuple
import numpy as np

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 150, min_chunk_size: int = 50) -> List[Tuple[str, int]]:
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [(text, 0)]
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        if len(chunk_words) >= min_chunk_size:
            char_offset = len(" ".join(words[:start]))
            chunks.append((chunk_text, char_offset))
        if end >= len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\/\%]', '', text)
    return text.strip()
