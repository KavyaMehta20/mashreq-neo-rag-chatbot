"""
metadata_extractor.py
Extracts recommendation and eligibility metadata from chunk content.
"""

import re
from copy import deepcopy
from typing import Any, Optional

# Sentinel stored in Chroma for unset numeric fields (Chroma does not support null)
UNSET_NUM = -1

DEFAULT_METADATA: dict[str, Any] = {
    "minimum_age": None,
    "maximum_age": None,
    "minimum_income": None,
    "resident_required": None,
    "guardian_required": None,
    "teen_product": False,
    "adult_product": False,
    "investment_product": False,
    "benefit_score": 0,
    "reward_score": 0,
    "cashback_score": 0,
    "travel_score": 0,
}

# Product-level hints when content alone is ambiguous
PRODUCT_HINTS: dict[str, dict[str, Any]] = {
    "NEO NXT Account": {
        "minimum_age": 8,
        "maximum_age": 18,
        "teen_product": True,
        "guardian_required": True,
    },
    "NEO Simple Account": {
        "minimum_age": 18,
        "adult_product": True,
        "minimum_income": 0,
    },
    "NEO PLUS Saver Account": {
        "investment_product": True,
        "minimum_age": 18,
        "adult_product": True,
    },
    "NEO Current Account": {
        "investment_product": True,
        "minimum_age": 18,
        "adult_product": True,
        "minimum_income": 5000,
    },
    "NEO Savings Account": {
        "minimum_age": 18,
        "adult_product": True,
        "minimum_income": 5000,
    },
}

AGE_RANGE_RE = re.compile(
    r"(?:aged?\s+)?(?:between\s+)?(\d{1,2})\s*(?:to|–|-|—|\band\b)\s*(\d{1,2})\s*years?",
    re.IGNORECASE,
)
AGE_MIN_RE = re.compile(
    r"(\d{1,2})\s*years?\s*(?:and\s+)?(?:or\s+)?above|"
    r"(\d{1,2})\s*years?\s*(?:of\s+age\s+)?or\s+(?:older|above)|"
    r"minimum\s+age[:\s]+(\d{1,2})|"
    r"aged?\s+(\d{1,2})\s*years?\s+or\s+above",
    re.IGNORECASE,
)
INCOME_RE = re.compile(
    r"(?:minimum\s+)?(?:monthly\s+)?(?:income|salary)\s*(?:of\s+|:)?\s*AED\s*([\d,]+)|"
    r"AED\s*([\d,]+)\s*/?\s*month\s+minimum|"
    r"minimum\s+(?:monthly\s+)?(?:income|salary)\s+AED\s*([\d,]+)",
    re.IGNORECASE,
)
INCOME_BELOW_RE = re.compile(
    r"(?:monthly\s+)?salary\s+below\s+AED\s*([\d,]+)|"
    r"earning\s+below\s+AED\s*([\d,]+)|"
    r"below\s+AED\s*([\d,]+)\s*(?:per\s+month|/month)",
    re.IGNORECASE,
)

BENEFIT_KEYWORDS = {
    "complimentary": 3,
    "lounge": 4,
    "insurance": 3,
    "discount": 2,
    "free": 1,
    "benefit": 2,
    "valet": 2,
    "golf": 2,
    "fitness": 2,
}
REWARD_KEYWORDS = {
    "vantage points": 5,
    "points per aed": 4,
    "rewards": 3,
    "redeem": 2,
    "airmiles": 3,
    "gift card": 2,
}
CASHBACK_KEYWORDS = {
    "cashback": 5,
    "% cashback": 8,
    "5%": 3,
    "2%": 2,
    "1%": 1,
}
TRAVEL_KEYWORDS = {
    "travel": 4,
    "airport": 4,
    "lounge": 3,
    "flight": 3,
    "hotel": 3,
    "airline": 3,
    "marhaba": 2,
}


def _parse_amount(raw: str) -> int:
    return int(raw.replace(",", ""))


def _score_keywords(text: str, keywords: dict[str, int]) -> int:
    lower = text.lower()
    score = 0
    for kw, weight in keywords.items():
        count = lower.count(kw)
        if count:
            score += count * weight
    return min(score, 100)


def extract_age_metadata(content: str) -> dict[str, Optional[int]]:
    result: dict[str, Optional[int]] = {"minimum_age": None, "maximum_age": None}

    range_match = AGE_RANGE_RE.search(content)
    if range_match:
        result["minimum_age"] = int(range_match.group(1))
        result["maximum_age"] = int(range_match.group(2))
        return result

    min_match = AGE_MIN_RE.search(content)
    if min_match:
        age = next(g for g in min_match.groups() if g)
        result["minimum_age"] = int(age)
        return result

    if re.search(r"minors?\s+(?:under|below)\s+(\d{1,2})", content, re.IGNORECASE):
        m = re.search(r"minors?\s+(?:under|below)\s+(\d{1,2})", content, re.IGNORECASE)
        if m:
            result["maximum_age"] = int(m.group(1)) - 1

    return result


def extract_income_metadata(content: str) -> dict[str, Optional[int]]:
    result: dict[str, Optional[int]] = {"minimum_income": None}

    below = INCOME_BELOW_RE.search(content)
    if below:
        amount = next(g for g in below.groups() if g)
        # Product caps income (e.g. NEO Simple); store as negative sentinel offset
        result["minimum_income"] = 0
        result["maximum_income"] = _parse_amount(amount) - 1
        return result

    match = INCOME_RE.search(content)
    if match:
        amount = next(g for g in match.groups() if g)
        result["minimum_income"] = _parse_amount(amount)

    return result


def extract_residency_metadata(content: str) -> dict[str, Optional[bool]]:
    lower = content.lower()
    resident = None
    if "uae resident" in lower or "emirates id" in lower:
        resident = True
    if "non-resident" in lower and ("not permitted" in lower or "not eligible" in lower):
        resident = True
    return {"resident_required": resident}


def extract_guardian_metadata(content: str) -> dict[str, Optional[bool]]:
    lower = content.lower()
    if "guardian" in lower or "parent/guardian" in lower or "parent must" in lower:
        return {"guardian_required": True}
    return {"guardian_required": None}


def extract_product_flags(content: str, product_name: str, age_meta: dict) -> dict[str, bool]:
    lower = content.lower()
    name_lower = product_name.lower()

    teen = bool(
        age_meta.get("maximum_age") is not None
        and age_meta.get("maximum_age", 99) <= 18
        and age_meta.get("minimum_age", 0) < 18
    ) or any(k in lower or k in name_lower for k in ("children", "kids", "youth", "nxt", "minor"))

    adult = bool(
        age_meta.get("minimum_age") is not None and age_meta.get("minimum_age", 0) >= 18
    ) or "18 years" in lower and "above" in lower

    if teen:
        adult = False

    investment = any(
        k in lower
        for k in (
            "term deposit",
            "stock exchange",
            "investment",
            "interest rate",
            "per annum",
            "high-yield",
            "6.25%",
            "5% per annum",
        )
    ) or "saver" in name_lower

    return {
        "teen_product": teen,
        "adult_product": adult and not teen,
        "investment_product": investment,
    }


def extract_benefit_scores(content: str, section: str) -> dict[str, int]:
    section_lower = (section or "").lower()
    text = f"{section_lower} {content}"

    scores = {
        "benefit_score": _score_keywords(text, BENEFIT_KEYWORDS),
        "reward_score": _score_keywords(text, REWARD_KEYWORDS),
        "cashback_score": _score_keywords(text, CASHBACK_KEYWORDS),
        "travel_score": _score_keywords(text, TRAVEL_KEYWORDS),
    }

    if "cashback" in section_lower:
        scores["cashback_score"] = max(scores["cashback_score"], 15)
    if section_lower in ("benefits", "rewards"):
        scores["benefit_score"] = max(scores["benefit_score"], 10)
    if "travel" in section_lower or "lifestyle" in section_lower:
        scores["travel_score"] = max(scores["travel_score"], 10)

    return scores


def extract_chunk_metadata(chunk: dict) -> dict[str, Any]:
    """Extract full recommendation metadata from a single chunk."""
    meta = deepcopy(DEFAULT_METADATA)
    content = chunk.get("content", "")
    product_name = chunk.get("product_name", "")
    section = chunk.get("section", "")

    age_meta = extract_age_metadata(content)
    income_meta = extract_income_metadata(content)
    residency = extract_residency_metadata(content)
    guardian = extract_guardian_metadata(content)
    flags = extract_product_flags(content, product_name, age_meta)
    scores = extract_benefit_scores(content, section)

    meta.update(age_meta)
    meta.update({k: v for k, v in income_meta.items() if k in meta or k == "maximum_income"})
    meta.update({k: v for k, v in residency.items() if v is not None})
    meta.update({k: v for k, v in guardian.items() if v is not None})
    meta.update(flags)
    meta.update(scores)

    # Apply product-level hints for missing age/income
    hints = PRODUCT_HINTS.get(product_name, {})
    for key, value in hints.items():
        if meta.get(key) is None or meta.get(key) is False:
            if key in ("teen_product", "adult_product", "investment_product", "guardian_required"):
                meta[key] = meta[key] or value
            elif meta.get(key) is None:
                meta[key] = value

    return meta


def merge_product_metadata(chunks: list[dict]) -> dict[str, dict[str, Any]]:
    """Aggregate metadata per product_name (max scores, tightest age/income bounds)."""
    product_map: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        name = chunk.get("product_name", "")
        if not name:
            continue
        extracted = extract_chunk_metadata(chunk)
        if name not in product_map:
            product_map[name] = deepcopy(extracted)
            continue

        agg = product_map[name]
        for score_key in ("benefit_score", "reward_score", "cashback_score", "travel_score"):
            agg[score_key] = max(agg[score_key], extracted[score_key])

        for age_key in ("minimum_age", "minimum_income"):
            if extracted[age_key] is not None:
                if agg[age_key] is None:
                    agg[age_key] = extracted[age_key]
                else:
                    agg[age_key] = min(agg[age_key], extracted[age_key])

        if extracted["maximum_age"] is not None:
            if agg["maximum_age"] is None:
                agg["maximum_age"] = extracted["maximum_age"]
            else:
                agg["maximum_age"] = max(agg["maximum_age"], extracted["maximum_age"])

        for flag in ("teen_product", "adult_product", "investment_product", "guardian_required"):
            agg[flag] = agg[flag] or extracted[flag]

        if extracted.get("resident_required"):
            agg["resident_required"] = True

    return product_map


def enrich_chunk_with_metadata(chunk: dict, product_metadata: dict[str, dict]) -> dict[str, Any]:
    """Merge chunk-level extraction with product-level aggregation."""
    chunk_meta = extract_chunk_metadata(chunk)
    product_name = chunk.get("product_name", "")
    product_meta = product_metadata.get(product_name, {})

    merged = deepcopy(chunk_meta)
    for key in ("minimum_age", "maximum_age", "minimum_income"):
        if merged[key] is None and product_meta.get(key) is not None:
            merged[key] = product_meta[key]

    for flag in ("teen_product", "adult_product", "investment_product", "guardian_required"):
        merged[flag] = merged[flag] or product_meta.get(flag, False)

    if merged["resident_required"] is None:
        merged["resident_required"] = product_meta.get("resident_required")

    for score_key in ("benefit_score", "reward_score", "cashback_score", "travel_score"):
        merged[score_key] = max(merged[score_key], product_meta.get(score_key, 0))

    return merged


def metadata_for_chroma(meta: dict[str, Any]) -> dict[str, Any]:
    """Convert metadata to Chroma-compatible types (no null floats)."""
    chroma: dict[str, Any] = {}
    for key in (
        "minimum_age",
        "maximum_age",
        "minimum_income",
        "benefit_score",
        "reward_score",
        "cashback_score",
        "travel_score",
    ):
        val = meta.get(key)
        chroma[key] = int(val) if val is not None else UNSET_NUM

    for key in ("teen_product", "adult_product", "investment_product"):
        chroma[key] = bool(meta.get(key, False))

    if meta.get("resident_required") is not None:
        chroma["resident_required"] = bool(meta["resident_required"])
    if meta.get("guardian_required") is not None:
        chroma["guardian_required"] = bool(meta["guardian_required"])

    if meta.get("maximum_income") is not None:
        chroma["maximum_income"] = int(meta["maximum_income"])

    return chroma


def format_metadata_for_embedding(meta: dict[str, Any], section: str) -> str:
    """Build eligibility/age/income lines for embedding text."""
    lines = []
    section_lower = (section or "").lower()

    if section_lower == "eligibility" or meta.get("minimum_income") is not None:
        lines.append("Eligibility Metadata")
    if meta.get("minimum_income") is not None and meta["minimum_income"] >= 0:
        lines.append(f"Minimum Income: AED {meta['minimum_income']}")
    if meta.get("minimum_age") is not None:
        lines.append(f"Minimum Age: {meta['minimum_age']}")
    if meta.get("maximum_age") is not None:
        lines.append(f"Maximum Age: {meta['maximum_age']}")
    if meta.get("teen_product"):
        lines.append("Teen Product: Yes")
    if meta.get("adult_product"):
        lines.append("Adult Product: Yes")
    if meta.get("investment_product"):
        lines.append("Investment Product: Yes")
    if meta.get("guardian_required"):
        lines.append("Guardian Required: Yes")

    return "\n".join(lines)
