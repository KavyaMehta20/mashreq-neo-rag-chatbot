"""
vector_store.py
Loads embedded_chunks.json and inserts all chunks into a ChromaDB persistent collection.
"""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDED_CHUNKS_PATH = Path("data/embedded_chunks.json")
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "mashreq_neo_knowledge_base"

# Fields to store as filterable metadata (exclude raw embedding and long content)
METADATA_FIELDS = [
    "product_type",
    "product_name",
    "section",
    "subsection",
    "source_document",
    "embedding_model",
    "embedding_dim",
]


def build_metadata(chunk: dict) -> dict:
    """Extract a flat metadata dict from a chunk (ChromaDB only accepts str/int/float/bool)."""
    meta = {}
    for field in METADATA_FIELDS:
        value = chunk.get(field, "")
        meta[field] = value if value is not None else ""
    return meta


def main():
    logger.info("Loading embedded chunks from %s", EMBEDDED_CHUNKS_PATH)
    with open(EMBEDDED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        embedded_chunks = json.load(f)
    logger.info("Loaded %d embedded chunks", len(embedded_chunks))

    logger.info("Initialising ChromaDB at %s", CHROMA_PERSIST_DIR)
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if it exists (idempotent re-run)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        logger.warning("Collection '%s' already exists — deleting and recreating.", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine distance for normalised embeddings
    )
    logger.info("Created collection: %s", COLLECTION_NAME)

    # Batch insert in groups of 100 to respect ChromaDB limits
    BATCH_SIZE = 100
    ids, embeddings, metadatas, documents = [], [], [], []

    for chunk in embedded_chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        metadatas.append(build_metadata(chunk))
        documents.append(chunk["content"])

    total = len(ids)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            documents=documents[start:end],
        )
        logger.info("Inserted batch %d–%d / %d", start + 1, end, total)

    logger.info("Vector store populated. Collection '%s' contains %d documents.", COLLECTION_NAME, collection.count())


if __name__ == "__main__":
    main()
