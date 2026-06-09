# Mashreq NEO Banking RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) pipeline for Mashreq Bank NEO products — accounts, debit cards, credit cards, and KFS documents.

---

## Project Structure

```
project/
├── data/
│   ├── chunks.json            # 54 hierarchical chunks from the knowledge base
│   ├── metadata.json          # Product/section inventory and counts
│   └── embedded_chunks.json   # chunks.json + SentenceTransformer embeddings (generated)
├── embed.py                   # Step 1: generate embeddings
├── vector_store.py            # Step 2: populate ChromaDB
├── retriever.py               # Step 3: semantic search
├── prompt_builder.py          # Step 4: construct RAG prompts
├── main.py                    # Step 5: FastAPI backend
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # if using Claude
export OPENAI_API_KEY=sk-...          # if using OpenAI GPT-4o
export LLM_PROVIDER=anthropic         # or "openai"
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
```

### 3. Generate Embeddings

```bash
python embed.py
```

This reads `data/chunks.json`, generates embeddings using `all-MiniLM-L6-v2`, and writes `data/embedded_chunks.json`.

### 4. Populate Vector Store

```bash
python vector_store.py
```

This creates a persistent ChromaDB collection at `./chroma_db/`.

### 5. Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the fees for the Mashreq Solitaire Credit Card?",
    "top_k": 5
  }'
```

With metadata filters:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What documents do I need?",
    "top_k": 3,
    "product_type_filter": "Account",
    "product_name_filter": "NEO Current Account",
    "section_filter": "Documents Required"
  }'
```

---

## API Endpoints

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | /health     | Health check and vector store count  |
| POST   | /query      | Main RAG query endpoint              |
| GET    | /products   | List all products in knowledge base  |
| GET    | /docs       | Swagger UI (auto-generated)          |

### POST /query — Request Body

| Field                  | Type   | Required | Description                               |
|------------------------|--------|----------|-------------------------------------------|
| question               | string | Yes      | Customer question (3–2000 chars)          |
| top_k                  | int    | No       | Chunks to retrieve (default 5, max 20)    |
| product_type_filter    | string | No       | "Account", "Credit Card", "Debit Card", "KFS" |
| product_name_filter    | string | No       | Exact product name                        |
| section_filter         | string | No       | "Fees", "Eligibility", "Benefits", etc.  |
| conversation_context   | string | No       | Prior conversation text for multi-turn    |

### POST /query — Response Body

```json
{
  "question": "...",
  "answer": "... [Source: CC001-FEE, Mashreq Solitaire Credit Card — Fees]",
  "sources": [
    {
      "chunk_id": "CC001-FEE",
      "product_type": "Credit Card",
      "product_name": "Mashreq Solitaire Credit Card",
      "section": "Fees",
      "subsection": "Fee Schedule",
      "score": 0.912,
      "text_snippet": "..."
    }
  ],
  "model_used": "claude-sonnet-4-20250514",
  "latency_ms": 812.4,
  "chunks_retrieved": 5
}
```

---

## Document Hierarchy (PART 1)

```
Level 1 — Product Type
├── Account
│   ├── NEO Current Account
│   ├── NEO Simple Account
│   ├── NEO PLUS Saver Account
│   ├── NEO Savings Account
│   └── NEO NXT Account
├── Debit Card
│   ├── NEO Debit Card
│   └── Mashreq noon Debit Card
├── Credit Card
│   ├── Mashreq Solitaire Credit Card
│   ├── Mashreq Platinum Plus Credit Card
│   ├── Mashreq Cashback Credit Card
│   └── Mashreq noon Credit Card
└── KFS
    ├── NEO Current Account KFS (January 2026)
    ├── NEO Simple Account KFS (April 2026)
    ├── NEO NXT Digital Account KFS
    ├── Unified CASA KFS (April 2026)
    └── Mashreq Cards KFS (effective 13 June 2026)

Level 2 — Product Name (see above)

Level 3 — Section
├── Product Overview
├── Benefits
├── Eligibility
├── Documents Required
├── Fees
├── Account Limits
├── Features
├── Key Facts Statement
└── Terms and Conditions

Level 4 — Subsection (where available)
    e.g. "Salary Transfer Cashback", "Tier Criteria", "Fee Schedule", "Rewards Structure"
```

---

## How This Works — Step-by-Step (PART 11)

### 1. Markdown Ingestion
The source document `Mashreq_NEO_UAE_Product_Discovery_Report.md` (1,262 lines) was read in full. Its structure — product categories, product names, sections, subsections, tables, and fee schedules — was manually analysed to identify every distinct knowledge unit.

### 2. Hierarchical Chunking
Each knowledge unit was extracted into a standalone chunk preserving its full banking context. Chunks are never split mid-concept. Fee tables, eligibility criteria, document requirements, KFS extracts, and interest rate tables are each kept as complete, self-contained units. Every chunk carries structured metadata: `chunk_id`, `product_type`, `product_name`, `section`, `subsection`, `content`, and `source_document`.

### 3. JSON Creation
`data/chunks.json` holds 54 chunks. `data/metadata.json` holds a complete inventory: all products, sections, product types, total chunk count, and per-product/per-section counts.

### 4. Embedding Generation (`embed.py`)
Each chunk's content is prefixed with its metadata labels (`Product Type:`, `Product Name:`, `Section:`) to anchor the embedding in product context, then encoded with `all-MiniLM-L6-v2` (384-dimensional, normalised for cosine similarity). Output: `data/embedded_chunks.json`.

### 5. Vector Storage (`vector_store.py`)
ChromaDB is initialised with `hnsw:space=cosine`. All 54 chunks are inserted with their embeddings, metadata, and document text in batches of 100. The collection is named `mashreq_neo_knowledge_base`.

### 6. Retrieval (`retriever.py`)
The `MashreqRetriever` class embeds the user query, queries ChromaDB for the top-k nearest neighbours, and returns `RetrievedChunk` objects sorted by cosine similarity score. Optional metadata filters (`product_type`, `product_name`, `section`) can narrow results before the vector search.

### 7. Prompt Construction (`prompt_builder.py`)
`build_rag_prompt()` assembles a system prompt (role, rules, citation requirements) and a user turn containing the numbered context block (chunk ID, product, section, relevance score, text) followed by the customer's question and an instruction to cite sources. The result is a `RAGPrompt` object with `system_prompt`, `user_prompt`, and `messages` fields.

### 8. LLM Answering (`main.py`)
FastAPI receives the query, calls the retriever, builds the RAG prompt, and dispatches to the configured LLM (Claude or GPT-4o). The LLM reads only the provided context and is instructed to cite chunk IDs. It cannot hallucinate information not present in the chunks.

### 9. Citation Generation
The LLM is instructed via the system prompt to include citations in the format `[Source: CHUNK_ID, Product Name — Section]` for every factual claim. The API response also returns the raw `sources` array with chunk IDs, scores, product names, and text snippets so the caller can render citations independently.

---

## Retrieval Optimisation (PART 12)

### Chunk Size Strategy
- **Banking documents require complete semantic units**: a fee table split across two chunks would cause incomplete answers. Every chunk in this pipeline preserves full tables, complete eligibility lists, and all KFS content.
- **Target**: 150–500 words per chunk. Fee and eligibility chunks tend to be 100–200 words; benefits chunks run 300–500 words. This fits comfortably in `all-MiniLM-L6-v2`'s 256-token context limit after metadata prefixing.
- **Do not chunk mid-table**: banking queries about fees must always retrieve the complete fee table.

### Overlap Strategy
- **Hierarchical chunking eliminates the need for sliding-window overlap**: each chunk is a complete, non-overlapping knowledge unit anchored to a specific product-section pair.
- For very long sections (e.g., a 20-item benefits list), a 10–15% overlap on the boundary sentences prevents context loss, but this is not needed in the current corpus.

### Metadata Filters
Use ChromaDB `where` clauses to narrow search before vector comparison:
- `product_type_filter="Credit Card"` — restrict to credit card chunks only
- `product_name_filter="NEO Current Account"` — pin to a single product
- `section_filter="Fees"` — target fee questions to fee chunks only
- Combine filters with `$and` for precision (e.g., Solitaire + Fees)

### Hybrid Retrieval
For production, combine dense (vector) and sparse (BM25/keyword) retrieval:
1. Run ChromaDB vector search for top-20 candidates.
2. Re-rank with BM25 over the candidate set using the original query keywords.
3. Take the top-5 after re-ranking.
This handles queries with specific terms (e.g., exact fee amounts, product codes) that may not rank well on embedding similarity alone.

### Re-ranking
After vector retrieval, apply a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top-20 candidates. Cross-encoders evaluate query+document jointly and significantly improve precision for banking FAQ queries. Add as a second pass between retrieval and prompt building.

### Top-K Values
- **Default top_k = 5**: sufficient for most single-product queries.
- **Increase to 8–10** for: cross-product comparison questions, eligibility questions spanning multiple products, general fee questions.
- **Use metadata filters to reduce top_k**: filtering to `product_name + section` allows top_k = 2–3 without losing recall.
- **Never exceed 10** without compressing chunks — the LLM context window and coherence degrade with too many chunks.
