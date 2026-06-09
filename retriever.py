"""
retriever.py
Accepts a user query, embeds it, searches ChromaDB, and returns ranked results.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "mashreq_neo_knowledge_base"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float          # cosine similarity (higher = more relevant)
    product_type: str
    product_name: str
    section: str
    subsection: str
    source_document: str
    text: str
    metadata: dict = field(default_factory=dict)


class MashreqRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chroma_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

        logger.info("Connecting to ChromaDB at %s", chroma_dir)
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(collection_name)
        logger.info(
            "Connected to collection '%s' (%d documents)", collection_name, self.collection.count()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        product_type_filter: Optional[str] = None,
        product_name_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Embed the query and perform a nearest-neighbour search in ChromaDB.
        Optional metadata filters narrow the search to a specific product type,
        product name, or section (useful for targeted banking queries).

        Args:
            query: Natural language question from the user.
            top_k: Number of results to return.
            product_type_filter: e.g. "Credit Card", "Account", "Debit Card", "KFS"
            product_name_filter: e.g. "Mashreq Solitaire Credit Card"
            section_filter: e.g. "Fees", "Eligibility", "Benefits"

        Returns:
            List of RetrievedChunk ordered by descending relevance score.
        """
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Build optional where clause for metadata filtering
        where_clause = self._build_where_clause(
            product_type_filter, product_name_filter, section_filter
        )

        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.collection.query(**query_kwargs)

        retrieved = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            # ChromaDB returns L2 distance when space=cosine internally;
            # convert distance to similarity: similarity = 1 - distance
            distance = results["distances"][0][i]
            score = round(1.0 - distance, 6)
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=score,
                    product_type=meta.get("product_type", ""),
                    product_name=meta.get("product_name", ""),
                    section=meta.get("section", ""),
                    subsection=meta.get("subsection", ""),
                    source_document=meta.get("source_document", ""),
                    text=doc,
                    metadata=meta,
                )
            )

        retrieved.sort(key=lambda x: x.score, reverse=True)
        return retrieved

    @staticmethod
    def _build_where_clause(
        product_type: Optional[str],
        product_name: Optional[str],
        section: Optional[str],
    ) -> Optional[dict]:
        conditions = []
        if product_type:
            conditions.append({"product_type": {"$eq": product_type}})
        if product_name:
            conditions.append({"product_name": {"$eq": product_name}})
        if section:
            conditions.append({"section": {"$eq": section}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# Standalone usage / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    retriever = MashreqRetriever()

    test_queries = [
        "What are the fees for the Solitaire Credit Card?",
        "What documents do I need to open a NEO Current Account?",
        "What is the eligibility for NEO NXT kids account?",
        "What cashback does the noon credit card offer?",
        "What is the interest rate on NEO PLUS Saver Account?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print("="*70)
        chunks = retriever.retrieve(q, top_k=3)
        for rank, chunk in enumerate(chunks, 1):
            print(f"\n  [{rank}] chunk_id={chunk.chunk_id}  score={chunk.score:.4f}")
            print(f"       product={chunk.product_name}  section={chunk.section}")
            print(f"       text snippet: {chunk.text[:200]}...")
