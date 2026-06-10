"""
retriever.py
Hybrid retrieval: metadata filtering, vector search, score boosting, benefit ranking,
and cross-encoder reranking.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer

from metadata_extractor import UNSET_NUM
from query_analyzer import QueryIntent, analyze_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "mashreq_neo_knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_TOP_K = 5
VECTOR_CANDIDATES = 20


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    vector_score: float = 0.0
    rerank_score: float = 0.0
    product_type: str = ""
    product_name: str = ""
    section: str = ""
    subsection: str = ""
    source_document: str = ""
    text: str = ""
    metadata: dict = field(default_factory=dict)


class MashreqRetriever:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        reranker_name: str = RERANKER_MODEL,
        chroma_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        enable_reranker: bool = True,
    ):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.embedding_model_name = model_name

        self.reranker = None
        if enable_reranker:
            logger.info("Loading cross-encoder reranker: %s", reranker_name)
            self.reranker = CrossEncoder(reranker_name)

        logger.info("Connecting to ChromaDB at %s", chroma_dir)
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(collection_name)
        logger.info(
            "Connected to collection '%s' (%d documents)",
            collection_name,
            self.collection.count(),
        )

    def _expand_query(self, query: str, intent: QueryIntent) -> str:
        """Enrich query with detected product/section hints for better embedding match."""
        parts = [query]
        if intent.product_name_hint:
            parts.append(intent.product_name_hint)
        if intent.section_hint:
            parts.append(intent.section_hint)
        return " ".join(parts)

    def _encode_query(self, query: str, intent: Optional[QueryIntent] = None) -> list[float]:
        text = self._expand_query(query, intent) if intent else query
        if "bge" in self.embedding_model_name.lower():
            text = BGE_QUERY_PREFIX + text
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    @staticmethod
    def _meta_int(meta: dict, key: str) -> Optional[int]:
        val = meta.get(key, UNSET_NUM)
        if val is None or val == UNSET_NUM:
            return None
        return int(val)

    @staticmethod
    def _passes_salary_filter(meta: dict, salary: int) -> bool:
        min_income = MashreqRetriever._meta_int(meta, "minimum_income")
        max_income = MashreqRetriever._meta_int(meta, "maximum_income")

        if max_income is not None and salary > max_income:
            return False
        if min_income is not None and min_income > 0 and salary < min_income:
            return False
        return True

    @staticmethod
    def _passes_age_filter(meta: dict, age: int) -> bool:
        min_age = MashreqRetriever._meta_int(meta, "minimum_age")
        max_age = MashreqRetriever._meta_int(meta, "maximum_age")

        if min_age is not None and age < min_age:
            return False
        if max_age is not None and age > max_age:
            return False
        return True

    @staticmethod
    def _passes_investment_filter(meta: dict) -> bool:
        return bool(meta.get("investment_product", False))

    @staticmethod
    def _build_where_clause(
        intent: QueryIntent,
        product_type_filter: Optional[str],
        product_name_filter: Optional[str],
        section_filter: Optional[str],
    ) -> Optional[dict]:
        conditions = []

        name_filter = product_name_filter or intent.product_name_hint
        
        type_filter = product_type_filter
        if not type_filter and not name_filter and not intent.recommendation_intent:
            type_filter = intent.product_type_hint

        if type_filter:
            conditions.append({"product_type": {"$eq": type_filter}})
        if name_filter:
            conditions.append({"product_name": {"$eq": name_filter}})
        if section_filter:
            conditions.append({"section": {"$eq": section_filter}})

        apply_salary_filter = intent.salary is not None and not name_filter
        apply_age_filter = intent.age is not None and not name_filter
        apply_investment_filter = intent.investment_intent

        if apply_investment_filter:
            conditions.append({"investment_product": {"$eq": True}})

        if apply_salary_filter:
            conditions.append({"minimum_income": {"$lte": intent.salary}})

        if apply_age_filter:
            conditions.append({"minimum_age": {"$lte": intent.age}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _compute_boost(meta: dict, intent: QueryIntent) -> float:
        boost = 0.0

        if intent.cashback_intent:
            boost += meta.get("cashback_score", 0) * 0.01
        if intent.travel_intent:
            boost += meta.get("travel_score", 0) * 0.01
        if intent.recommendation_intent and not intent.salary:
            boost += meta.get("benefit_score", 0) * 0.008

        section = meta.get("section", "")
        if intent.section_hint and section == intent.section_hint:
            boost += 0.2
        if intent.product_name_hint and meta.get("product_name") == intent.product_name_hint:
            boost += 0.25
        if intent.product_type_hint and meta.get("product_type") == intent.product_type_hint:
            boost += 0.1

        if intent.age is not None and meta.get("teen_product"):
            boost += 0.12
        if intent.salary is not None and meta.get("minimum_income", UNSET_NUM) not in (UNSET_NUM, -1):
            min_inc = int(meta["minimum_income"])
            if min_inc <= intent.salary:
                boost += 0.05

        return boost

    @staticmethod
    def _benefit_rank_key(meta: dict, intent: QueryIntent) -> tuple:
        min_income = MashreqRetriever._meta_int(meta, "minimum_income") or 0
        min_age = MashreqRetriever._meta_int(meta, "minimum_age") or 0

        salary_fit = 0
        if intent.salary is not None and min_income > 0:
            salary_fit = 1 if min_income <= intent.salary else 0
        elif intent.salary is None:
            salary_fit = 1

        age_fit = 1
        if intent.age is not None:
            age_fit = 1 if MashreqRetriever._passes_age_filter(meta, intent.age) else 0

        return (
            -salary_fit,
            -age_fit,
            -int(meta.get("benefit_score", 0)),
            -int(meta.get("reward_score", 0)),
            -int(meta.get("cashback_score", 0)),
            -int(meta.get("travel_score", 0)),
        )

    def _post_filter(self, results: list[RetrievedChunk], intent: QueryIntent) -> list[RetrievedChunk]:
        if intent.product_name_hint:
            return results

        filtered = []
        for chunk in results:
            meta = chunk.metadata
            if intent.salary is not None and not self._passes_salary_filter(meta, intent.salary):
                continue
            if intent.age is not None and not self._passes_age_filter(meta, intent.age):
                continue
            if intent.investment_intent and not self._passes_investment_filter(meta):
                continue
            filtered.append(chunk)
        return filtered

    def _rerank(self, query: str, candidates: list[RetrievedChunk], intent: QueryIntent) -> list[RetrievedChunk]:
        if not self.reranker or not candidates:
            return candidates

        pairs = [[self._expand_query(query, intent), c.text] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for chunk, rs in zip(candidates, rerank_scores):
            chunk.rerank_score = float(rs)

        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        for chunk in candidates:
            chunk.score = chunk.vector_score
        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        product_type_filter: Optional[str] = None,
        product_name_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        intent: Optional[QueryIntent] = None,
        skip_rerank: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval pipeline:
        1. Query intent detection
        2. Metadata filtering + vector search (top 20)
        3. Post-filter + score boosting
        4. Benefit ranking for recommendations
        5. Cross-encoder rerank → top_k
        """
        if intent is None:
            intent = analyze_query(query)

        query_embedding = self._encode_query(query, intent)
        where_clause = self._build_where_clause(
            intent, product_type_filter, product_name_filter, section_filter
        )

        n_candidates = VECTOR_CANDIDATES
        query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        try:
            results = self.collection.query(**query_kwargs)
        except Exception as exc:
            logger.warning("Filtered query failed (%s), falling back to unfiltered search.", exc)
            query_kwargs.pop("where", None)
            query_kwargs["n_results"] = n_candidates
            results = self.collection.query(**query_kwargs)

        candidates: list[RetrievedChunk] = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            vector_score = round(1.0 - distance, 6)
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]

            boost = self._compute_boost(meta, intent)
            combined_score = round(vector_score + boost, 6)

            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=combined_score,
                    vector_score=vector_score,
                    product_type=meta.get("product_type", ""),
                    product_name=meta.get("product_name", ""),
                    section=meta.get("section", ""),
                    subsection=meta.get("subsection", ""),
                    source_document=meta.get("source_document", ""),
                    text=doc,
                    metadata=meta,
                )
            )

        candidates = self._post_filter(candidates, intent)

        if not candidates:
            fallback_kwargs = dict(
                query_embeddings=[query_embedding],
                n_results=n_candidates,
                include=["documents", "metadatas", "distances"],
            )
            results = self.collection.query(**fallback_kwargs)
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                vector_score = round(1.0 - distance, 6)
                boost = self._compute_boost(meta, intent)
                candidates.append(
                    RetrievedChunk(
                        chunk_id=results["ids"][0][i],
                        score=round(vector_score + boost, 6),
                        vector_score=vector_score,
                        product_type=meta.get("product_type", ""),
                        product_name=meta.get("product_name", ""),
                        section=meta.get("section", ""),
                        subsection=meta.get("subsection", ""),
                        source_document=meta.get("source_document", ""),
                        text=results["documents"][0][i],
                        metadata=meta,
                    )
                )
            candidates = self._post_filter(candidates, intent)

        if intent.recommendation_intent or intent.salary or intent.age:
            candidates.sort(key=lambda c: self._benefit_rank_key(c.metadata, intent))
        else:
            candidates.sort(key=lambda x: x.score, reverse=True)

        pre_rerank = candidates[:VECTOR_CANDIDATES]

        if not skip_rerank and self.reranker:
            pre_rerank = self._rerank(query, pre_rerank, intent)
        else:
            pre_rerank.sort(key=lambda x: x.score, reverse=True)

        return pre_rerank[:top_k]


if __name__ == "__main__":
    retriever = MashreqRetriever()

    test_queries = [
        "What salary do I need for Solitaire Card?",
        "What card can I get with AED 7000 salary?",
        "I am 16 years old. What account can I open?",
        "What investment products are available?",
        "What documents are needed for NEO Current Account?",
        "Which card offers the best cashback?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        intent = analyze_query(q)
        print(f"INTENT: salary={intent.salary}, age={intent.age}, "
              f"investment={intent.investment_intent}, cashback={intent.cashback_intent}")
        print("=" * 70)
        chunks = retriever.retrieve(q, top_k=3)
        for rank, chunk in enumerate(chunks, 1):
            print(f"\n  [{rank}] chunk_id={chunk.chunk_id}  score={chunk.score:.4f}  "
                  f"vector={chunk.vector_score:.4f}")
            print(f"       product={chunk.product_name}  section={chunk.section}")
            print(f"       text snippet: {chunk.text[:180]}...")
