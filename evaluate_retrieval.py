"""
evaluate_retrieval.py
Evaluates hybrid retrieval quality with Recall@5, MRR, and Top-1 Accuracy.
"""

import logging
from dataclasses import dataclass

from retriever import MashreqRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ground-truth: at least one acceptable chunk_id per query
EVAL_QUERIES: list[dict] = [
    {"query": "What salary do I need for Solitaire Card?", "relevant_ids": ["solitaire-cc-002", "solitaire-cc-003"]},
    {"query": "What is the minimum income for Mashreq Solitaire Credit Card?", "relevant_ids": ["solitaire-cc-003", "solitaire-cc-002"]},
    {"query": "What card can I get with AED 7000 salary?", "relevant_ids": ["cross-product-002", "cashback-cc-002", "platinum-plus-cc-002", "platinum-plus-cc-003", "noon-cc-002"]},
    {"query": "I earn AED 7000, which credit card am I eligible for?", "relevant_ids": ["cross-product-002", "cashback-cc-002", "platinum-plus-cc-002"]},
    {"query": "I am 16 years old. What account can I open?", "relevant_ids": ["neo-nxt-003", "neo-nxt-002", "cross-product-001"]},
    {"query": "I am 16 years old, what banking options do I have?", "relevant_ids": ["neo-nxt-003", "cross-product-001"]},
    {"query": "What investment products are available?", "relevant_ids": ["neo-plus-saver-001", "neo-plus-saver-003", "neo-current-001", "neo-plus-saver-002"]},
    {"query": "What investment options do I have with Mashreq NEO?", "relevant_ids": ["neo-plus-saver-001", "neo-current-001"]},
    {"query": "What documents are needed for NEO Current Account?", "relevant_ids": ["neo-current-006"]},
    {"query": "Which card offers the best cashback?", "relevant_ids": ["cross-product-003", "cashback-cc-003", "cashback-cc-001"]},
    {"query": "What are the fees for the Solitaire Credit Card?", "relevant_ids": ["solitaire-cc-008"]},
    {"query": "What is the eligibility for NEO NXT kids account?", "relevant_ids": ["neo-nxt-002", "neo-nxt-003"]},
    {"query": "What cashback does the noon credit card offer?", "relevant_ids": ["noon-cc-003", "noon-cc-001"]},
    {"query": "What is the interest rate on NEO PLUS Saver Account?", "relevant_ids": ["neo-plus-saver-003"]},
    {"query": "What is the minimum balance for NEO Current Account?", "relevant_ids": ["neo-current-008", "neo-current-004"]},
    {"query": "What are NEO Simple Account eligibility requirements?", "relevant_ids": ["neo-simple-002", "neo-simple-003", "neo-simple-004"]},
    {"query": "How old do I need to be for NEO Current Account?", "relevant_ids": ["neo-current-003", "neo-current-002"]},
    {"query": "What is the annual fee for Platinum Plus credit card?", "relevant_ids": ["platinum-plus-cc-007", "platinum-plus-cc-001"]},
    {"query": "Does Mashreq Cashback card have annual fee?", "relevant_ids": ["cashback-cc-007", "cashback-cc-001"]},
    {"query": "What lounge access does Solitaire card provide?", "relevant_ids": ["solitaire-cc-006"]},
    {"query": "What are NEO NXT parental controls?", "relevant_ids": ["neo-nxt-006"]},
    {"query": "What is the welcome bonus on Cashback credit card?", "relevant_ids": ["cashback-cc-006"]},
    {"query": "What debit card comes with NEO Savings Account?", "relevant_ids": ["neo-savings-001", "cross-product-011"]},
    {"query": "Can non-residents open NEO account?", "relevant_ids": ["neo-current-005", "cross-product-009"]},
    {"query": "What is NEO PLUS Saver eligibility?", "relevant_ids": ["neo-plus-saver-002"]},
    {"query": "What are noon debit card cashback rates?", "relevant_ids": ["noon-debit-003"]},
    {"query": "What is the fall-below fee for NEO Current Account?", "relevant_ids": ["neo-current-007", "neo-current-008"]},
    {"query": "What salary transfer cashback does NEO Current offer?", "relevant_ids": ["neo-current-009"]},
    {"query": "What happens when NEO NXT child turns 18?", "relevant_ids": ["neo-nxt-009", "neo-nxt-001"]},
    {"query": "What are Mashreq noon credit card fees?", "relevant_ids": ["noon-cc-007"]},
    {"query": "Which products require AED 5000 minimum salary?", "relevant_ids": ["cross-product-002"]},
    {"query": "What are Solitaire card reward points?", "relevant_ids": ["solitaire-cc-005"]},
    {"query": "What is Platinum Plus rewards structure?", "relevant_ids": ["platinum-plus-cc-004"]},
    {"query": "What are NEO Savings account fees?", "relevant_ids": ["neo-savings-005"]},
    {"query": "What documents needed for NEO Simple Account?", "relevant_ids": ["neo-simple-005"]},
    {"query": "What is NEO Simple account daily limit?", "relevant_ids": ["neo-simple-007"]},
    {"query": "What interest rate does NEO Savings pay?", "relevant_ids": ["neo-savings-004"]},
    {"query": "What are cross-product interest rates?", "relevant_ids": ["cross-product-006", "neo-savings-004", "neo-plus-saver-003"]},
    {"query": "Which NEO accounts have no minimum balance?", "relevant_ids": ["cross-product-005", "neo-simple-001", "neo-nxt-001"]},
    {"query": "What travel benefits does Solitaire offer?", "relevant_ids": ["solitaire-cc-006", "solitaire-cc-001"]},
    {"query": "What movie discounts on Cashback card?", "relevant_ids": ["cashback-cc-005"]},
    {"query": "What is noon credit card welcome bonus?", "relevant_ids": ["noon-cc-006"]},
    {"query": "What are NEO debit card features?", "relevant_ids": ["neo-debit-003", "neo-debit-001"]},
    {"query": "What is Mashreq noon savings minimum income?", "relevant_ids": ["noon-savings-002", "noon-savings-001"]},
    {"query": "Can I get Solitaire card with 7000 salary?", "relevant_ids": ["solitaire-cc-003", "cross-product-002"]},
    {"query": "I earn 3000 AED, which account suits me?", "relevant_ids": ["neo-simple-002", "neo-simple-004", "neo-simple-003"]},
    {"query": "What is the highest income requirement credit card?", "relevant_ids": ["solitaire-cc-001", "solitaire-cc-003"]},
    {"query": "What products offer salary transfer bonus?", "relevant_ids": ["neo-current-009", "neo-savings-007", "cross-product-003"]},
    {"query": "What are NEO NXT account limits?", "relevant_ids": ["neo-nxt-007"]},
    {"query": "What is NEO NXT welcome bonus?", "relevant_ids": ["neo-nxt-008"]},
    {"query": "Which credit cards are free for life?", "relevant_ids": ["cashback-cc-001", "noon-cc-001", "cross-product-004"]},
    {"query": "What is the credit card interest rate?", "relevant_ids": ["cross-product-006", "cross-product-015", "solitaire-cc-008"]},
    {"query": "What Emirates ID documents for credit card?", "relevant_ids": ["solitaire-cc-004"]},
    {"query": "What is NEO Current account opening process?", "relevant_ids": ["neo-current-001"]},
    {"query": "What are all NEO credit cards?", "relevant_ids": ["cross-product-004", "cross-product-012", "cross-product-015", "cross-product-016"]},
    {"query": "Best card for dining cashback?", "relevant_ids": ["cashback-cc-003", "cashback-cc-001"]},
    {"query": "What noon ecosystem benefits on noon card?", "relevant_ids": ["noon-cc-005", "noon-cc-001"]},
]


@dataclass
class EvalMetrics:
    recall_at_5: float
    mrr: float
    top1_accuracy: float
    avg_top1_score: float
    direct_factual_top1_accuracy: float
    direct_factual_avg_score: float
    total_queries: int


DIRECT_FACTUAL_INDICES = {
    0, 1, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 52, 53,
}


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def run_evaluation(top_k: int = 5) -> EvalMetrics:
    retriever = MashreqRetriever()
    recalls, rrs, top1_hits = [], [], []
    top1_scores = []
    direct_top1_hits, direct_scores = [], []

    for idx, item in enumerate(EVAL_QUERIES):
        query = item["query"]
        relevant = set(item["relevant_ids"])

        results = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r.chunk_id for r in results]

        recalls.append(recall_at_k(retrieved_ids, relevant, k=top_k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant))
        top1_hits.append(1 if retrieved_ids and retrieved_ids[0] in relevant else 0)

        if results:
            top1_scores.append(results[0].vector_score)
            if idx in DIRECT_FACTUAL_INDICES:
                direct_scores.append(results[0].vector_score)
                direct_top1_hits.append(1 if retrieved_ids[0] in relevant else 0)

    n = len(EVAL_QUERIES)
    n_direct = len(DIRECT_FACTUAL_INDICES)

    return EvalMetrics(
        recall_at_5=sum(recalls) / n,
        mrr=sum(rrs) / n,
        top1_accuracy=sum(top1_hits) / n,
        avg_top1_score=sum(top1_scores) / len(top1_scores) if top1_scores else 0.0,
        direct_factual_top1_accuracy=sum(direct_top1_hits) / n_direct if n_direct else 0.0,
        direct_factual_avg_score=sum(direct_scores) / len(direct_scores) if direct_scores else 0.0,
        total_queries=n,
    )


def print_report(metrics: EvalMetrics) -> None:
    print("\n" + "=" * 70)
    print("MASHREQ NEO RAG — RETRIEVAL EVALUATION REPORT")
    print("=" * 70)
    print(f"Total queries evaluated : {metrics.total_queries}")
    print(f"Recall@5                : {metrics.recall_at_5:.2%}")
    print(f"MRR                     : {metrics.mrr:.4f}")
    print(f"Top-1 Accuracy          : {metrics.top1_accuracy:.2%}")
    print(f"Avg Top-1 Vector Score  : {metrics.avg_top1_score:.4f}")
    print("-" * 70)
    print("Direct Factual Questions:")
    print(f"  Top-1 Accuracy        : {metrics.direct_factual_top1_accuracy:.2%}")
    print(f"  Avg Vector Score      : {metrics.direct_factual_avg_score:.4f}")
    print(f"  Target (>0.75 score)  : {'PASS' if metrics.direct_factual_avg_score >= 0.75 else 'BELOW TARGET'}")
    print(f"  Target (>90% Top-1)   : {'PASS' if metrics.direct_factual_top1_accuracy >= 0.90 else 'BELOW TARGET'}")
    print("=" * 70)


if __name__ == "__main__":
    logger.info("Starting retrieval evaluation (%d queries) ...", len(EVAL_QUERIES))
    metrics = run_evaluation(top_k=5)
    print_report(metrics)
