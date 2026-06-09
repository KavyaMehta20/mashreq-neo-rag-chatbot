"""
embed.py
Generates sentence embeddings for all chunks using SentenceTransformers (all-MiniLM-L6-v2)
and saves the result to embedded_chunks.json.
"""

import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/chunks.json")
OUTPUT_PATH = Path("data/embedded_chunks.json")
MODEL_NAME = "all-MiniLM-L6-v2"


def build_embedding_text(chunk: dict) -> str:
    """
    Constructs a rich text representation for embedding.
    Prepends structured metadata context so the embedding captures
    product identity in addition to content semantics.
    """
    parts = [
        f"Product Type: {chunk.get('product_type', '')}",
        f"Product Name: {chunk.get('product_name', '')}",
        f"Section: {chunk.get('section', '')}",
    ]
    if chunk.get("subsection"):
        parts.append(f"Subsection: {chunk.get('subsection', '')}")
    parts.append(chunk.get("content", ""))
    return "\n".join(parts)


def main():
    logger.info("Loading chunks from %s", CHUNKS_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))

    logger.info("Loading SentenceTransformer model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = [build_embedding_text(chunk) for chunk in chunks]
    logger.info("Generating embeddings for %d texts ...", len(texts))

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity ready
        convert_to_numpy=True,
    )
    logger.info("Embeddings generated. Shape: %s", embeddings.shape)

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = embedding.tolist()
        embedded_chunk["embedding_model"] = MODEL_NAME
        embedded_chunk["embedding_dim"] = len(embedding)
        embedded_chunks.append(embedded_chunk)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d embedded chunks to %s", len(embedded_chunks), OUTPUT_PATH)


if __name__ == "__main__":
    main()
