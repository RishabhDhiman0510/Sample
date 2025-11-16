"""FastAPI server with endpoints for query, feedback, and metrics."""
import asyncio, time, re
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from ..config import settings
from ..schema.response_schema import QueryRequest, FeedbackRequest, LLMResponse, EvidenceItem
from ..utils.logging import setup_logging, get_logger, get_metrics, REQUEST_COUNT, ERROR_COUNT, RETRIEVAL_LATENCY, GENERATION_LATENCY, FEEDBACK_COUNT
from ..model_loader import ModelLoader
from ..retrieval.loader import LocalRAGLoader
from ..retrieval.store import AdvancedVectorStore
from ..web_search.pubmed_ddg import PubMedSearch, DuckDuckGoSearch, WebSearchCache
from ..feedback.feedback_store import FeedbackStore
from ..prompt.templating import PromptTemplate, ResponseParser

logger = get_logger(__name__)
model_loader, vector_store, feedback_store = None, None, None
web_cache, pubmed_search, ddg_search = None, None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loader, vector_store, feedback_store, web_cache, pubmed_search, ddg_search
    logger.info("application_startup_begin")
    setup_logging(settings.LOG_LEVEL)
    logger.info("loading_model")
    model_loader = ModelLoader(adapter_path=settings.MODEL_ADAPTER_PATH, base_model_path=settings.BASE_MODEL_PATH, load_in_8bit=settings.LOAD_IN_8BIT, device_map=settings.DEVICE_MAP)
    model_loader.load()
    logger.info("loading_rag_library")
    rag_loader = LocalRAGLoader(library_dir=settings.RAG_LIBRARY_DIR, library_name=settings.RAG_LIBRARY_NAME, embedding_model_name=settings.EMBEDDING_MODEL, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    vector_store = AdvancedVectorStore(local_rag=rag_loader, reranker_model=settings.RERANKER_MODEL, min_sim=settings.MIN_SIMILARITY)
    feedback_store = FeedbackStore(feedback_file=settings.FEEDBACK_FILE, embedding_model=rag_loader.embedding_model, encryption_key=settings.FEEDBACK_ENCRYPTION_KEY, enable_pii_detection=settings.ENABLE_PII_DETECTION)
    if settings.ENABLE_WEB_SEARCH:
        web_cache = WebSearchCache(ttl=settings.CACHE_TTL)
        pubmed_search = PubMedSearch(email=settings.PUBMED_EMAIL, timeout=settings.WEB_SEARCH_TIMEOUT, cache=web_cache)
        ddg_search = DuckDuckGoSearch(timeout=settings.WEB_SEARCH_TIMEOUT, cache=web_cache)
    logger.info("application_startup_complete")
    yield
    logger.info("application_shutdown")

app = FastAPI(title="Medical RAG API", description="Production-quality medical Q&A system", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "model_loaded": model_loader is not None}

@app.get("/metrics")
async def metrics():
    if not settings.ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain")

@app.post("/query", response_model=LLMResponse)
async def query(request: QueryRequest):
    REQUEST_COUNT.labels(endpoint="query", method="POST").inc()
    logger.info("query_received", question=request.question[:100])
    try:
        start_time = time.time()
        corrections = feedback_store.get_relevant_corrections(request.question, top_k=3, min_similarity=0.5)
        logger.info("corrections_retrieved", num_corrections=len(corrections))
        with RETRIEVAL_LATENCY.time():
            retrieval_results = vector_store.retrieve_and_rerank(request.question, top_k=settings.TOP_K_RETRIEVAL, rerank_top_k=settings.TOP_K_RERANK)
        logger.info("local_retrieval_complete", num_results=len(retrieval_results))
        web_results = []
        if settings.ENABLE_WEB_SEARCH and (not retrieval_results or sum(score for _, score in retrieval_results) / len(retrieval_results) < 0.35):
            logger.info("triggering_web_search")
            pubmed_task = pubmed_search.search(request.question, max_results=3)
            ddg_task = ddg_search.search(request.question)
            pubmed_results, ddg_results = await asyncio.gather(pubmed_task, ddg_task, return_exceptions=True)
            if not isinstance(pubmed_results, Exception):
                web_results.extend(pubmed_results)
            if not isinstance(ddg_results, Exception):
                web_results.extend(ddg_results)
            logger.info("web_search_complete", num_web_results=len(web_results))
        context_parts, evidence_items = [], []
        for i, web_text in enumerate(web_results[:3]):
            context_parts.append(f"[WEB SOURCE {i+1}] {web_text}")
            url_match = re.search(r'https?://[^\s\]]+', web_text)
            url = url_match.group(0) if url_match else None
            evidence_items.append(EvidenceItem(doc_id=f"web_{i}", source="web", excerpt=web_text[:200], score=0.8, url=url))
        for i, (doc, score) in enumerate(retrieval_results[:3]):
            context_parts.append(f"[LOCAL {i+1}] {doc['text'][:300]}")
            evidence_items.append(EvidenceItem(doc_id=doc["doc_id"], source=doc.get("source", "local"), excerpt=doc["text"][:200], score=score, url=doc.get("source_url")))
        for corr in corrections[:2]:
            context_parts.append(f"[CORRECTION] Q: {corr['question']} A: {corr['correct_answer']}")
        if not context_parts and not corrections:
            raise HTTPException(status_code=404, detail="No relevant information found")
        context = "\n\n".join(context_parts)
        with GENERATION_LATENCY.time():
            prompt = PromptTemplate.create_reasoning_prompt(question=request.question, context=context, corrections=corrections, conversation_history="")
            generated_text = model_loader.generate(prompt=prompt, max_new_tokens=settings.MAX_NEW_TOKENS, temperature=settings.TEMPERATURE, top_p=settings.TOP_P, repetition_penalty=settings.REPETITION_PENALTY)
        scores = [item.score for item in evidence_items]
        confidence = vector_store.calibrate_score(scores, len(evidence_items), len(web_results) > 0)
        response = ResponseParser.parse_response(generated_text, prompt, evidence_items, confidence)
        response.method = "hybrid" if web_results else "local_rag"
        latency = time.time() - start_time
        logger.info("query_complete", latency=latency, confidence=confidence, method=response.method)
        return response
    except ValidationError as e:
        ERROR_COUNT.labels(error_type="validation").inc()
        logger.error("validation_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        ERROR_COUNT.labels(error_type="internal").inc()
        logger.error("query_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    FEEDBACK_COUNT.inc()
    REQUEST_COUNT.labels(endpoint="feedback", method="POST").inc()
    logger.info("feedback_received", question=request.question[:100])
    try:
        feedback_store.add_correction(question=request.question, wrong_answer=request.wrong_answer, correct_answer=request.correct_answer, tags=request.tags)
        return {"status": "success", "message": "Feedback recorded and learned"}
    except Exception as e:
        ERROR_COUNT.labels(error_type="feedback").inc()
        logger.error("feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to save feedback")
