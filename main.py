"""
main.py  —  Mashreq NEO RAG API (Gemini free tier) - Updated 5
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import time
import requests
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from google.api_core import exceptions as google_exceptions

from prompt_builder import build_rag_prompt, build_decomposition_prompt, DECOMPOSITION_SYSTEM_PROMPT
from retriever import MashreqRetriever, RetrievedChunk
from query_analyzer import analyze_query, detect_product_name_hint
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
DEFAULT_TOP_K   = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_CHUNKS      = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "1024"))
CHROMA_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading retriever ...")
    app_state["retriever"] = MashreqRetriever(chroma_dir=CHROMA_DIR)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    app_state["gemini_model"] = genai.GenerativeModel(GEMINI_MODEL)
    logger.info("Gemini model ready: %s", GEMINI_MODEL)
    yield
    app_state.clear()


app = FastAPI(
    title="Mashreq NEO Banking RAG API",
    description="RAG chatbot for Mashreq Bank NEO products (Gemini free tier)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/source/{filename}")
async def serve_source(filename: str):
    if filename != "productreport.md":
        raise HTTPException(status_code=404, detail="Source not found")

    source_path = Path("productreport.md")
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file missing")
    return FileResponse(source_path)


# ── Schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "question": "What are the fees for the Mashreq Solitaire Credit Card?",
                    "top_k": 5
                }
            ]
        }
    )

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    product_type_filter: Optional[str] = None
    product_name_filter: Optional[str] = None
    section_filter: Optional[str] = None
    conversation_context: Optional[str] = None


class SourceChunk(BaseModel):
    chunk_id: str
    product_type: str
    product_name: str
    section: str
    subsection: str
    score: float
    text_snippet: str


class RetrievedContextChunk(BaseModel):
    text: str
    similarity_score: float
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str
    sources: list[SourceChunk]
    model_used: str
    latency_ms: float
    chunks_retrieved: int
    retrieved_context: list[RetrievedContextChunk] = []
    subqueries: list[str] = []


# ── Context-aware retrieval helpers ──────────────────────────────────────

# Pronoun/vague words that signal a follow-up referring to a prior product
_FOLLOWUP_SIGNALS = (
    " that", " it", " this", " those", " them", " the account", " the card",
    "do i get", "does it", "what about", "tell me more", "more about",
    "what are its", "what are the",
)


def _extract_product_from_history(conversation_context: str) -> Optional[str]:
    """Scan conversation history lines (newest first) for a recognisable product name."""
    if not conversation_context:
        return None
    for line in reversed(conversation_context.strip().split("\n")):
        product = detect_product_name_hint(line)
        if product:
            return product
    return None


def _enrich_query_with_context(
    question: str,
    conversation_context: Optional[str],
) -> str:
    """
    If the question looks like a vague pronoun follow-up AND the conversation
    history mentions a specific product, append that product name to the search
    query so ChromaDB retrieves the right chunks.
    """
    lower = question.lower()
    is_followup = any(sig in lower for sig in _FOLLOWUP_SIGNALS)
    current_intent = analyze_query(question)
    has_product_already = bool(current_intent.product_name_hint or current_intent.product_type_hint)

    if is_followup and not has_product_already:
        context_product = _extract_product_from_history(conversation_context)
        if context_product:
            enriched = f"{question} {context_product}"
            logger.info("Follow-up detected — enriching query with context product: '%s'", context_product)
            return enriched
    return question


# ── LLM call ───────────────────────────────────────────────────────────────

def call_gemini(system_prompt: str, user_prompt: str) -> str:
    model = app_state["gemini_model"]
    # Gemini takes system instruction separately
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    response = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=MAX_TOKENS),
    )
    return response.text


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call local Ollama API."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {"num_predict": MAX_TOKENS},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "No response from Ollama.")
    except requests.exceptions.RequestException as e:
        logger.error("Ollama HTTP error: %s", e)
        raise RuntimeError(f"Ollama connection error: {e}")
    except Exception as e:
        logger.error("Unexpected Ollama error: %s", e)
        raise RuntimeError(f"Ollama unexpected error: {e}")


def chunks_to_sources(chunks: list[RetrievedChunk]) -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id=c.chunk_id,
            product_type=c.product_type,
            product_name=c.product_name,
            section=c.section,
            subsection=c.subsection,
            score=round(c.score, 6),
            text_snippet=c.text[:300] + ("..." if len(c.text) > 300 else ""),
        )
        for c in chunks
    ]


def append_product_pdfs(question: str, answer: str, chunks: list[RetrievedChunk]) -> str:
    # If the answer indicates no info was found, we do not append links
    if "could not find" in answer.lower() or "sorry" in answer.lower():
        return answer

    PRODUCT_MAPPING = {
        "NEO Current Account": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/accounts-deposits/Neo-Current-Account-KFS.ashx",
            "keywords": ["neo current account", "neo current", "current account"],
            "chunk_names": ["NEO Current Account", "NEO Current Account KFS"]
        },
        "NEO Simple Account": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/accounts-deposits/Neo-Simple-Account-EN-KFS.ashx",
            "keywords": ["neo simple account", "neo simple", "simple account"],
            "chunk_names": ["NEO Simple Account", "NEO Simple Account KFS"]
        },
        "NEO PLUS Saver Account": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/accounts-deposits/KFS/2025/neo-plus-saver-account-kfs-en-ar.ashx",
            "keywords": ["neo plus saver", "plus saver", "neo plus"],
            "chunk_names": ["NEO PLUS Saver Account"]
        },
        "NEO Savings Account": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/accounts-deposits/KFS/2026/mashreq-unified-casa-products-kfs.ashx",
            "keywords": ["neo savings account", "neo savings"],
            "chunk_names": ["NEO Savings Account"]
        },
        "NEO NXT Account": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/accounts-deposits/KFS/KFS-Neo-NXT-Digital-Account-en-ar.ashx",
            "keywords": ["neo nxt", "nxt account", "nxt generation"],
            "chunk_names": ["NEO NXT Account", "NEO NXT Digital Account KFS"]
        },
        "NEO Debit Card": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/accounts-deposits/Neo-Current-Account-KFS.ashx",
            "keywords": ["neo debit card", "neo debit"],
            "chunk_names": ["NEO Debit Card"]
        },
        "Mashreq noon Savings Account": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/accounts-deposits/Mashreq_noon_VIP_Savings_Interest_Rates-EN-AR.ashx",
            "keywords": ["noon savings", "noon savings account", "noon saving"],
            "chunk_names": ["Mashreq noon Savings Account"]
        },
        "Mashreq Solitaire Credit Card": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/cards/KFS-TnC/Mashreq-Cards-KFS-new-en-ar.ashx",
            "keywords": ["solitaire credit card", "solitaire card", "solitaire"],
            "chunk_names": ["Mashreq Solitaire Credit Card"]
        },
        "Mashreq Platinum Plus Credit Card": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/cards/KFS-TnC/Mashreq-Cards-KFS-new-en-ar.ashx",
            "keywords": ["platinum plus credit card", "platinum plus card", "platinum plus"],
            "chunk_names": ["Mashreq Platinum Plus Credit Card"]
        },
        "Mashreq Cashback Credit Card": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/cards/KFS-TnC/Mashreq-Cards-KFS-new-en-ar.ashx",
            "keywords": ["cashback credit card", "cashback card", "cashback credit", "cashback"],
            "chunk_names": ["Mashreq Cashback Credit Card"]
        },
        "Mashreq noon Credit Card": {
            "url": "https://www.mashreq.com/-/jssmedia/pdfs/neo/cards/noon/Mahreq_noon_Credit_Card_Terms_and_Conditions_en.ashx",
            "keywords": ["noon credit card", "noon card", "noon credit"],
            "chunk_names": ["Mashreq noon Credit Card"]
        }
    }

    question_lower = question.lower()
    answer_lower = answer.lower()
    
    top_chunks = chunks[:MAX_CHUNKS]
    retrieved_product_names = {c.product_name for c in top_chunks}
    
    matched_products = []
    
    for product_name, info in PRODUCT_MAPPING.items():
        chunk_match = any(cn in retrieved_product_names for cn in info["chunk_names"])
        
        kw_match_q = any(kw in question_lower for kw in info["keywords"])
        kw_match_a = any(kw in answer_lower for kw in info["keywords"])
        
        if (chunk_match and (kw_match_q or kw_match_a)) or (kw_match_q and kw_match_a):
            matched_products.append((product_name, info["url"]))
            
    # Fallback for single product context/follow-up when no keywords matched:
    if not matched_products and len(retrieved_product_names) == 1:
        single_name = list(retrieved_product_names)[0]
        for product_name, info in PRODUCT_MAPPING.items():
            if single_name in info["chunk_names"]:
                matched_products.append((product_name, info["url"]))
                break

    if not matched_products:
        return answer

    seen_urls = set()
    unique_links = []
    for prod_name, url in matched_products:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_links.append(url)
            
    if not unique_links:
        return answer

    if len(unique_links) == 1:
        link_block = "\n\nRelevant PDF:\n" + unique_links[0]
    else:
        link_block = "\n\nRelevant PDFs:\n" + "\n".join(unique_links)
    return answer + link_block


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    retriever = app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Not ready")
    return {
        "status": "ok",
        "documents": retriever.collection.count(),
        "model": GEMINI_MODEL,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    t0 = time.perf_counter()
    retriever: MashreqRetriever = app_state["retriever"]

    # 1. Decompose the question into sub-queries
    sub_queries = [request.question]
    try:
        logger.info("Decomposing query: %s", request.question)
        decomp_prompt = build_decomposition_prompt(request.question)
        # We use the system prompt to enforce JSON output
        decomp_response = call_gemini(DECOMPOSITION_SYSTEM_PROMPT, decomp_prompt)
        
        # Clean up potential markdown code blocks from LLM response
        clean_response = decomp_response.strip().replace("```json", "").replace("```", "").strip()
        sub_queries = json.loads(clean_response)
        if not isinstance(sub_queries, list):
            sub_queries = [request.question]
        logger.info("Decomposed into: %s", sub_queries)
    except Exception as e:
        logger.warning("Query decomposition failed: %s. Falling back to original question.", e)
        sub_queries = [request.question]

    # 2. Perform multi-query retrieval
    all_retrieved_chunks = []
    seen_chunk_ids = set()

    for sq in sub_queries:
        # Enrich each sub-query with context if it's a follow-up
        enriched_sq = _enrich_query_with_context(sq, request.conversation_context)
        
        chunks = retriever.retrieve(
            query=enriched_sq,
            top_k=request.top_k,
            product_type_filter=request.product_type_filter,
            product_name_filter=request.product_name_filter,
            section_filter=request.section_filter,
        )

        for c in chunks:
            if c.chunk_id not in seen_chunk_ids:
                all_retrieved_chunks.append(c)
                seen_chunk_ids.add(c.chunk_id)

    # Sort all chunks by score descending
    all_retrieved_chunks.sort(key=lambda x: x.score, reverse=True)

    if not all_retrieved_chunks:
        return QueryResponse(
            question=request.question,
            answer="I could not find relevant information. Please visit mashreq.com or contact Mashreq support.",
            sources=[],
            model_used=GEMINI_MODEL,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            chunks_retrieved=0,
            retrieved_context=[],
            subqueries=sub_queries,
        )

    # 3. Build RAG prompt with the consolidated chunks
    rag_prompt = build_rag_prompt(
        question=request.question,
        chunks=all_retrieved_chunks,
        max_chunks=MAX_CHUNKS,
        conversation_context=request.conversation_context,
    )

    # 4. Generate final answer
    try:
        answer = call_gemini(rag_prompt.system_prompt, rag_prompt.user_prompt)
    except google_exceptions.ResourceExhausted:
        logger.error("Gemini API quota exceeded (ResourceExhausted). Attempting Ollama fallback...")
        try:
            answer = call_ollama(rag_prompt.system_prompt, rag_prompt.user_prompt)
        except Exception as ollama_err:
            logger.error("Ollama fallback also failed: %s", ollama_err)
            return QueryResponse(
                question=request.question,
                answer="I'm sorry, I've reached my API limits and my local backup is unavailable. Please try again later.",
                sources=chunks_to_sources(all_retrieved_chunks),
                model_used="none",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                chunks_retrieved=len(all_retrieved_chunks),
                retrieved_context=[
                    RetrievedContextChunk(
                        text=c.text,
                        similarity_score=round(c.score, 6),
                        vector_score=round(c.vector_score, 6),
                        rerank_score=round(c.rerank_score, 6) if c.rerank_score != 0.0 else None,
                        chunk_id=c.chunk_id
                    )
                    for c in all_retrieved_chunks[:MAX_CHUNKS]
                ],
                subqueries=sub_queries,
            )
    except Exception as e:
        logger.exception("Unexpected error during LLM call")
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Answered in %.0f ms", latency)

    final_answer = append_product_pdfs(request.question, answer, all_retrieved_chunks)

    final_chunks = all_retrieved_chunks[:MAX_CHUNKS]
    retrieved_context_data = [
        RetrievedContextChunk(
            text=c.text,
            similarity_score=round(c.score, 6),
            vector_score=round(c.vector_score, 6),
            rerank_score=round(c.rerank_score, 6) if c.rerank_score != 0.0 else None,
            chunk_id=c.chunk_id
        )
        for c in final_chunks
    ]

    return QueryResponse(
        question=request.question,
        answer=final_answer,
        sources=chunks_to_sources(all_retrieved_chunks),
        model_used=GEMINI_MODEL,
        latency_ms=latency,
        chunks_retrieved=len(all_retrieved_chunks),
        retrieved_context=retrieved_context_data,
        subqueries=sub_queries,
    )



@app.get("/products")
async def products():
    meta = Path("data/metadata.json")
    if not meta.exists():
        raise HTTPException(404, "metadata.json not found")
    return json.loads(meta.read_text())

