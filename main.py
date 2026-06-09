"""
main.py  —  Mashreq NEO RAG API (Gemini free tier)
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
from pydantic import BaseModel, ConfigDict, Field

from prompt_builder import build_rag_prompt
from retriever import MashreqRetriever, RetrievedChunk
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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

    chunks = retriever.retrieve(
        query=request.question,
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
