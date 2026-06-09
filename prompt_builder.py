"""
prompt_builder.py
Constructs production-quality RAG prompts for the Mashreq banking assistant.
"""

from dataclasses import dataclass
from typing import Optional

from retriever import RetrievedChunk


@dataclass
class RAGPrompt:
    system_prompt: str
    user_prompt: str
    messages: list[dict]   # ready-to-send format for Anthropic / OpenAI APIs


SYSTEM_PROMPT = """You are a knowledgeable and professional Mashreq Bank assistant specialising in Mashreq NEO banking products, including accounts, debit cards, credit cards, and Key Facts Statements (KFS).

Your responsibilities:
1. Answer customer questions accurately using ONLY the context provided below.
2. If the answer is not available in the provided context, respond with:
   "I could not find that information in the knowledge base. Please visit mashreq.com or contact Mashreq customer support for the most up-to-date information."
3. Always cite the source chunk ID and product name when providing information (e.g., [Source: ACC001-FEE, NEO Current Account — Fees]).
4. When fee or rate information involves discrepancies between documents, flag this clearly and recommend the customer verify via the official Schedule of Charges at mashreq.com/soc.
5. Never fabricate fees, rates, eligibility requirements, or product features not present in the context.
6. For regulatory or legal questions, direct customers to Mashreqbank PSC directly.
7. All products are regulated by the Central Bank of the United Arab Emirates."""


def build_context_block(chunks: list[RetrievedChunk], max_chunks: int = 5) -> str:
    """
    Formats retrieved chunks into a numbered context block for the prompt.
    Includes chunk metadata as structured labels so the LLM can cite sources.
    """
    top_chunks = chunks[:max_chunks]
    lines = ["### RETRIEVED CONTEXT\n"]

    for i, chunk in enumerate(top_chunks, 1):
        lines.append(f"--- Context [{i}] ---")
        lines.append(f"Chunk ID      : {chunk.chunk_id}")
        lines.append(f"Product Type  : {chunk.product_type}")
        lines.append(f"Product Name  : {chunk.product_name}")
        lines.append(f"Section       : {chunk.section}")
        if chunk.subsection:
            lines.append(f"Subsection    : {chunk.subsection}")
        lines.append(f"Relevance Score: {chunk.score:.4f}")
        lines.append(f"Source Document: {chunk.source_document}")
        lines.append("")
        lines.append(chunk.text)
        lines.append("")

    return "\n".join(lines)


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    max_chunks: int = 5,
    conversation_context: Optional[str] = None,
) -> str:
    """
    Constructs the full user turn including context and the question.
    """
    parts = []

    if conversation_context:
        parts.append("### CONVERSATION HISTORY")
        parts.append(conversation_context)
        parts.append("")

    parts.append(build_context_block(chunks, max_chunks=max_chunks))
    parts.append("### CUSTOMER QUESTION")
    parts.append(question)
    parts.append("")
    parts.append(
        "Using ONLY the context provided above, answer the customer's question accurately "
        "and cite the relevant chunk IDs as sources. If the context does not contain the answer, "
        "say so and suggest where the customer can find the information."
    )

    return "\n".join(parts)


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    max_chunks: int = 5,
    conversation_context: Optional[str] = None,
) -> RAGPrompt:
    """
    Main entry point. Returns a RAGPrompt with system prompt, user prompt,
    and pre-formatted messages array for the LLM API.

    Args:
        question: The user's natural language question.
        chunks: Retrieved chunks from the vector store.
        max_chunks: Maximum number of context chunks to include.
        conversation_context: Optional prior conversation turns as a string.

    Returns:
        RAGPrompt dataclass with all prompt components.
    """
    user_prompt = build_user_prompt(
        question=question,
        chunks=chunks,
        max_chunks=max_chunks,
        conversation_context=conversation_context,
    )

    messages = [
        {"role": "user", "content": user_prompt},
    ]

    return RAGPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from retriever import MashreqRetriever

    retriever = MashreqRetriever()
    question = "What are the fees and interest rates for the Solitaire Credit Card?"
    chunks = retriever.retrieve(question, top_k=5)
    rag_prompt = build_rag_prompt(question, chunks)

    print("SYSTEM PROMPT:")
    print("-" * 60)
    print(rag_prompt.system_prompt)
    print("\nUSER PROMPT:")
    print("-" * 60)
    print(rag_prompt.user_prompt[:2000], "...")
