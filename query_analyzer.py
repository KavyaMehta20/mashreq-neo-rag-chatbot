"""
query_analyzer.py
Detects salary, age, and investment intent from user queries.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

SALARY_PATTERNS = [
    re.compile(
        r"(?:earn|salary|income|make|paid)\s*(?:of\s+)?AED\s*([\d,]+)",
        re.IGNORECASE,
    ),
    re.compile(r"AED\s*([\d,]+)\s*(?:salary|income|per\s+month|/month)", re.IGNORECASE),
    re.compile(r"([\d,]+)\s*AED\s*(?:salary|income)", re.IGNORECASE),
]

AGE_PATTERNS = [
    re.compile(r"(?:i\s+am|i'm|age[d]?)\s*(\d{1,2})\s*years?\s*old", re.IGNORECASE),
    re.compile(r"(\d{1,2})\s*years?\s*old", re.IGNORECASE),
    re.compile(r"age\s*(?:of\s+)?(\d{1,2})", re.IGNORECASE),
]

INVESTMENT_KEYWORDS = (
    "investment",
    "invest",
    "term deposit",
    "stock",
    "savings rate",
    "interest rate",
    "high yield",
    "grow my money",
    "returns",
)

CASHBACK_KEYWORDS = (
    "cashback",
    "cash back",
    "best cashback",
    "most cashback",
)

TRAVEL_KEYWORDS = (
    "travel",
    "lounge",
    "airport",
    "flight",
    "hotel",
)

RECOMMENDATION_KEYWORDS = (
    "which card",
    "what card",
    "what account",
    "recommend",
    "best card",
    "can i get",
    "eligible for",
    "options do i have",
)


PRODUCT_NAME_PATTERNS: list[tuple[str, str]] = [
    ("solitaire", "Mashreq Solitaire Credit Card"),
    ("platinum plus", "Mashreq Platinum Plus Credit Card"),
    ("cashback credit", "Mashreq Cashback Credit Card"),
    ("cashback card", "Mashreq Cashback Credit Card"),
    ("noon credit", "Mashreq noon Credit Card"),
    ("noon debit", "Mashreq noon Debit Card"),
    ("neo debit", "NEO Debit Card"),
    ("nxt account", "NEO NXT Account"),
    ("neo current", "NEO Current Account"),
    ("neo simple", "NEO Simple Account"),
    ("neo savings", "NEO Savings Account"),
    ("neo plus saver", "NEO PLUS Saver Account"),
    ("noon savings", "Mashreq noon Savings Account"),
]


@dataclass
class QueryIntent:
    raw_query: str
    salary: Optional[int] = None
    age: Optional[int] = None
    investment_intent: bool = False
    cashback_intent: bool = False
    travel_intent: bool = False
    recommendation_intent: bool = False
    product_type_hint: Optional[str] = None
    product_name_hint: Optional[str] = None
    section_hint: Optional[str] = None
    keywords: list[str] = field(default_factory=list)


def _parse_amount(raw: str) -> int:
    return int(raw.replace(",", ""))


def detect_salary(query: str) -> Optional[int]:
    for pattern in SALARY_PATTERNS:
        match = pattern.search(query)
        if match:
            return _parse_amount(match.group(1))
    return None


def detect_age(query: str) -> Optional[int]:
    lower = query.lower()
    if any(w in lower for w in ("teenager", "teen", "youth", "child", "minor", "kid")):
        return 15
    for pattern in AGE_PATTERNS:
        match = pattern.search(query)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:
                return age
    return None


def detect_investment_intent(query: str) -> bool:
    lower = query.lower()
    return any(kw in lower for kw in INVESTMENT_KEYWORDS)


def detect_cashback_intent(query: str) -> bool:
    lower = query.lower()
    return any(kw in lower for kw in CASHBACK_KEYWORDS)


def detect_travel_intent(query: str) -> bool:
    lower = query.lower()
    return any(kw in lower for kw in TRAVEL_KEYWORDS)


OPEN_ENDED_PATTERNS = (
    "what products",
    "what options",
    "options do i have",
    "products are available",
    "which account suits",
    "which account can",
)


def detect_recommendation_intent(query: str) -> bool:
    lower = query.lower()
    if any(kw in lower for kw in RECOMMENDATION_KEYWORDS):
        return True
    if any(kw in lower for kw in OPEN_ENDED_PATTERNS):
        return True
    if detect_investment_intent(query) and any(k in lower for k in ("what", "which", "available", "options")):
        return True
    return False


def detect_product_name_hint(query: str) -> Optional[str]:
    lower = query.lower()
    for pattern, name in PRODUCT_NAME_PATTERNS:
        if pattern in lower:
            return name
    return None


def detect_product_type_hint(query: str) -> Optional[str]:
    lower = query.lower()
    if "credit card" in lower:
        return "Credit Card"
    if "debit card" in lower:
        return "Debit Card"
    if "savings" in lower:
        return "Savings Account"
    if "current account" in lower or "account" in lower:
        return "Current Account"
    if "card" in lower:
        return "Credit Card"
    return None


def detect_section_hint(query: str) -> Optional[str]:
    lower = query.lower()
    mapping = {
        "eligibility": "Eligibility",
        "fee": "Fees and Charges",
        "document": "Required Documents",
        "benefit": "Benefits",
        "cashback": "Cashback",
        "reward": "Rewards",
        "interest": "Interest Rates",
    }
    for key, section in mapping.items():
        if key in lower:
            return section
    return None


def analyze_query(query: str) -> QueryIntent:
    """Parse user query and return structured intent for hybrid retrieval."""
    product_name = detect_product_name_hint(query)
    recommendation = detect_recommendation_intent(query)

    # Specific product + factual question → not a open recommendation
    if product_name and not recommendation:
        recommendation = False
    elif product_name and any(k in query.lower() for k in ("what", "how", "which documents", "fee", "require")):
        recommendation = False

    return QueryIntent(
        raw_query=query,
        salary=detect_salary(query),
        age=detect_age(query),
        investment_intent=detect_investment_intent(query),
        cashback_intent=detect_cashback_intent(query),
        travel_intent=detect_travel_intent(query),
        recommendation_intent=recommendation,
        product_type_hint=detect_product_type_hint(query),
        product_name_hint=product_name,
        section_hint=detect_section_hint(query),
    )
