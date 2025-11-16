"""FeedbackStore with embedding-based retrieval and PII redaction."""
import json, re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ..utils.logging import get_logger

logger = get_logger(__name__)

class PIIRedactor:
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(\+\d{1,3}[-.]\s?)?(\(?\d{3}\)?[-.]\s?\d{3}[-.]\s?\d{4})\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }

    @classmethod
    def redact(cls, text: str) -> str:
        redacted = text
        for pii_type, pattern in cls.PATTERNS.items():
            redacted = re.sub(pattern, f"<REDACTED_{pii_type.upper()}>", redacted)
        return redacted

class FeedbackStore:
    def __init__(self, feedback_file: str, embedding_model: SentenceTransformer, encryption_key: Optional[str] = None, enable_pii_detection: bool = True):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.enable_pii = enable_pii_detection
        self.cipher = None
        self.feedback_data = self._load_feedback()
        self.index, self.correction_ids = self._build_correction_index()
        logger.info("feedback_store_initialized", num_corrections=len(self.feedback_data.get("corrections", [])))

    def _load_feedback(self) -> Dict[str, Any]:
        if not self.feedback_file.exists():
            return {"corrections": [], "preferences": {}}
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error("feedback_load_error", error=str(e))
            return {"corrections": [], "preferences": {}}

    def _save_feedback(self) -> None:
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            logger.info("feedback_saved", num_corrections=len(self.feedback_data["corrections"]))
        except Exception as e:
            logger.error("feedback_save_error", error=str(e))

    def _build_correction_index(self) -> tuple:
        corrections = self.feedback_data.get("corrections", [])
        if not corrections:
            logger.info("no_corrections_to_index")
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatIP(dimension)
            return index, []
        logger.info("building_correction_index", num_corrections=len(corrections))
        questions = [c["question"] for c in corrections]
        embeddings = self.embedding_model.encode(questions, normalize_embeddings=True, show_progress_bar=False).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        correction_ids = list(range(len(corrections)))
        logger.info("correction_index_built", index_size=index.ntotal)
        return index, correction_ids

    def add_correction(self, question: str, wrong_answer: str, correct_answer: str, tags: Optional[List[str]] = None) -> None:
        logger.info("adding_correction", question=question[:50])
        if self.enable_pii:
            question = PIIRedactor.redact(question)
            wrong_answer = PIIRedactor.redact(wrong_answer)
            correct_answer = PIIRedactor.redact(correct_answer)
            logger.info("pii_redacted")
        correction = {"timestamp": datetime.now().isoformat(), "question": question, "wrong_answer": wrong_answer, "correct_answer": correct_answer, "tags": tags or []}
        self.feedback_data["corrections"].append(correction)
        self._save_feedback()
        self.index, self.correction_ids = self._build_correction_index()
        logger.info("correction_added", total_corrections=len(self.feedback_data["corrections"]))

    def get_relevant_corrections(self, query: str, top_k: int = 3, min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        logger.info("retrieving_corrections", query=query[:50], top_k=top_k)
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype('float32')
        k = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(query_emb, k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= min_similarity and idx < len(self.feedback_data["corrections"]):
                correction = self.feedback_data["corrections"][idx].copy()
                correction["similarity"] = float(sim)
                results.append(correction)
        logger.info("corrections_retrieved", num_results=len(results))
        return results
