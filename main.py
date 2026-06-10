"""
main.py  —  Mashreq NEO RAG API (Gemini free tier) - Updated 5
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from prompt_builder import build_rag_prompt
from retriever import MashreqRetriever, RetrievedChunk
from query_analyzer import analyze_query, detect_product_name_hint
from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
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


# ── Schemas ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What are the fees for the Mashreq Solitaire Credit Card?",
                    "top_k": 5
                }
            ]
        }
    )

    question: str = Field(..., min_length=3, max_length=2000)
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


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    model_used: str
    latency_ms: float
    chunks_retrieved: int


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

    # Enrich vague follow-up questions with product context from history
    retrieval_query = _enrich_query_with_context(
        request.question, request.conversation_context
    )

    chunks = retriever.retrieve(
        query=retrieval_query,
        top_k=request.top_k,
        product_type_filter=request.product_type_filter,
        product_name_filter=request.product_name_filter,
        section_filter=request.section_filter,
    )

    if not chunks:
        return QueryResponse(
            question=request.question,
            answer="I could not find relevant information. Please visit mashreq.com or contact Mashreq support.",
            sources=[],
            model_used=GEMINI_MODEL,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            chunks_retrieved=0,
        )

    rag_prompt = build_rag_prompt(
        question=request.question,
        chunks=chunks,
        max_chunks=MAX_CHUNKS,
        conversation_context=request.conversation_context,
    )

    answer = call_gemini(rag_prompt.system_prompt, rag_prompt.user_prompt)
    latency = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Answered in %.0f ms", latency)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=chunks_to_sources(chunks),
        model_used=GEMINI_MODEL,
        latency_ms=latency,
        chunks_retrieved=len(chunks),
    )


@app.get("/products")
async def products():
    meta = Path("data/metadata.json")
    if not meta.exists():
        raise HTTPException(404, "metadata.json not found")
    return json.loads(meta.read_text())
