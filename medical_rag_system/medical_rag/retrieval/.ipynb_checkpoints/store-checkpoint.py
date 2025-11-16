"""AdvancedVectorStore: Retrieval with reranking and calibration."""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from sklearn.isotonic import IsotonicRegression
from ..utils.logging import get_logger

logger = get_logger(__name__)

class AdvancedVectorStore:
    def __init__(self, local_rag, reranker_model: Optional[str] = None, min_sim: float = 0.2):
        self.embedding_model = local_rag.embedding_model
        self.index = local_rag.index
        self.documents = local_rag.documents
        self.min_sim = min_sim
        self.reranker = None
        if reranker_model:
            logger.info("loading_reranker", model=reranker_model)
            self.reranker = CrossEncoder(reranker_model)
            logger.info("reranker_loaded")
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self._init_placeholder_calibrator()

    def _init_placeholder_calibrator(self):
        dummy_raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        dummy_calibrated = np.array([0.05, 0.15, 0.25, 0.40, 0.55, 0.70, 0.80, 0.88, 0.95])
        self.calibrator.fit(dummy_raw, dummy_calibrated)
        logger.info("placeholder_calibrator_initialized")

    def retrieve_and_rerank(self, query: str, top_k: int = 10, rerank_top_k: int = 5, use_reranking: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        logger.info("retrieval_start", query=query[:100], top_k=top_k)
        if self.index.ntotal == 0:
            logger.warning("empty_index")
            return []
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True).astype('float32')
        search_k = min(top_k, self.index.ntotal)
        sims, indices = self.index.search(query_emb, search_k)
        candidates = []
        for idx, sim in zip(indices[0], sims[0]):
            if idx != -1 and idx < len(self.documents) and float(sim) >= self.min_sim:
                doc = self.documents[idx]
                candidates.append((doc, float(sim)))
        if not candidates:
            logger.info("no_candidates_above_threshold", min_sim=self.min_sim)
            return []
        logger.info("initial_candidates_retrieved", count=len(candidates))
        if self.reranker and use_reranking and len(candidates) > 1:
            logger.info("reranking_start", num_candidates=len(candidates))
            texts = [doc["text"] for doc, _ in candidates]
            pairs = [(query, text) for text in texts]
            rerank_scores = self.reranker.predict(pairs, batch_size=16)
            combined = []
            for i, (doc, retrieval_score) in enumerate(candidates):
                combined_score = 0.4 * retrieval_score + 0.6 * float(rerank_scores[i])
                combined.append((doc, combined_score))
            combined.sort(key=lambda x: x[1], reverse=True)
            candidates = combined[:rerank_top_k]
            logger.info("reranking_complete", final_count=len(candidates))
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:rerank_top_k]
        calibrated_candidates = []
        for doc, score in candidates:
            calibrated_score = float(self.calibrator.predict([score])[0])
            calibrated_candidates.append((doc, calibrated_score))
        return calibrated_candidates

    def calibrate_score(self, raw_scores: List[float], num_sources: int, has_web_sources: bool) -> float:
        if not raw_scores:
            return 0.0
        mean_score = float(np.mean(raw_scores))
        max_score = float(np.max(raw_scores))
        base_confidence = 0.5 * mean_score + 0.5 * max_score
        source_boost = min(0.1 * (num_sources - 1), 0.2)
        web_boost = 0.15 if has_web_sources else 0.0
        combined = base_confidence + source_boost + web_boost
        combined = np.clip(combined, 0.0, 1.0)
        calibrated = float(self.calibrator.predict([combined])[0])
        return calibrated
