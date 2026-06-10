"""
vector_store.py
Loads embedded_chunks.json and inserts all chunks into a ChromaDB persistent collection
with full recommendation metadata for filterable hybrid retrieval.
"""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from metadata_extractor import metadata_for_chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDED_CHUNKS_PATH = Path("data/embedded_chunks.json")
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "mashreq_neo_knowledge_base"

BASE_METADATA_FIELDS = [
    "product_type",
    "product_name",
    "section",
    "subsection",
    "source_document",
    "embedding_model",
    "embedding_dim",
]

RECOMMENDATION_METADATA_FIELDS = [
    "minimum_age",
    "maximum_age",
    "minimum_income",
    "benefit_score",
    "reward_score",
    "cashback_score",
    "travel_score",
    "teen_product",
    "adult_product",
    "investment_product",
    "resident_required",
    "guardian_required",
    "maximum_income",
]


def build_metadata(chunk: dict) -> dict:
    """Extract a flat metadata dict from a chunk (ChromaDB only accepts str/int/float/bool)."""
    meta = {}
    for field in BASE_METADATA_FIELDS:
        value = chunk.get(field, "")
        meta[field] = value if value is not None else ""

    rec_meta = chunk.get("recommendation_metadata", {})
    if rec_meta:
        meta.update(metadata_for_chroma(rec_meta))
    else:
        meta.update(metadata_for_chroma({}))

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

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        logger.warning("Collection '%s' already exists — deleting and recreating.", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created collection: %s", COLLECTION_NAME)

    BATCH_SIZE = 100
    ids, embeddings, metadatas, documents = [], [], [], []

    for chunk in embedded_chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        metadatas.append(build_metadata(chunk))
        documents.append(chunk.get("content", ""))

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

    logger.info(
        "Vector store populated. Collection '%s' contains %d documents.",
        COLLECTION_NAME,
        collection.count(),
    )


if __name__ == "__main__":
    main()
