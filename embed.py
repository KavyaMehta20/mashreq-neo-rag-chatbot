"""
embed.py
Generates sentence embeddings using BAAI/bge-base-en-v1.5 with enriched metadata.
"""

import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from metadata_extractor import (
    enrich_chunk_with_metadata,
    format_metadata_for_embedding,
    merge_product_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/chunks.json")
OUTPUT_PATH = Path("data/embedded_chunks.json")
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def build_embedding_text(chunk: dict, metadata: dict) -> str:
    """
    Constructs a rich text representation for embedding including
    eligibility, age, and income metadata alongside content.
    """
    parts = [
        f"Product Type: {chunk.get('product_type', '')}",
        f"Product Name: {chunk.get('product_name', '')}",
        f"Section: {chunk.get('section', '')}",
    ]
    if chunk.get("subsection"):
        parts.append(f"Subsection: {chunk.get('subsection', '')}")

    meta_block = format_metadata_for_embedding(metadata, chunk.get("section", ""))
    if meta_block:
        parts.append(meta_block)

    parts.append("Content:")
    parts.append(chunk.get("content", ""))
    return "\n".join(parts)


def main():
    logger.info("Loading chunks from %s", CHUNKS_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))

    logger.info("Extracting product metadata ...")
    product_metadata = merge_product_metadata(chunks)

    enriched = []
    for chunk in chunks:
        meta = enrich_chunk_with_metadata(chunk, product_metadata)
        record = dict(chunk)
        record["recommendation_metadata"] = meta
        record["embedding_text"] = build_embedding_text(chunk, meta)
        enriched.append(record)

    logger.info("Loading SentenceTransformer model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = [r["embedding_text"] for r in enriched]
    logger.info("Generating embeddings for %d texts ...", len(texts))

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    logger.info("Embeddings generated. Shape: %s", embeddings.shape)

    embedded_chunks = []
    for record, embedding in zip(enriched, embeddings):
        embedded_chunk = dict(record)
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
