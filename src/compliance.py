"""Compliance review engine combining rule-based heuristics and LLM verification.

Checks for:
1. Trademark misuse (50+ famous brands, comparisons, fake endorsements)
2. False claims (superlatives, medical claims, "100%", "risk free")
3. Automated tagline regeneration with up to 2 retries if non-compliant.
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

from .prompts import COMPLIANCE_SYSTEM_PROMPT, COMPLIANCE_USER_TEMPLATE
from .tagline import generate_taglines, validate_single_tagline

load_dotenv()
logger = logging.getLogger(__name__)

# Curated list of 50+ famous global trademarks across sportswear, drinkware, beverage, tech, and lifestyle
PROTECTED_TRADEMARKS = [
    # Drinkware & Hydration Brands
    "hydro flask", "stanley", "yeti", "camelbak", "nalgene", "s'well", "swell",
    "thermos", "contigo", "klean kanteen", "corkcicle", "owala", "blenderbottle",
    "brita", "hydropeak", "iron flask", "simple modern",
    # Sports, Fitness & Apparel Brands
    "nike", "adidas", "under armour", "puma", "reebok", "lululemon", "gymshark",
    "asics", "new balance", "brooks", "the north face", "patagonia", "columbia",
    "aloyoga", "oakley", "hoka",
    # Beverage & Nutrition Brands
    "gatorade", "powerade", "liquid i.v.", "liquid iv", "prime hydration",
    "red bull", "monster energy", "coca-cola", "coca cola", "coke", "pepsi",
    "dasani", "aquafina", "smartwater", "fiji water", "evian", "vitaminwater",
    # Tech & Wearables
    "apple", "iphone", "apple watch", "fitbit", "garmin", "whoop", "samsung",
    "galaxy", "sony", "google", "pixel", "bose", "oura", "strava", "peloton",
    # Luxury Brands
    "rolex", "gucci", "prada", "louis vuitton", "chanel", "hermes"
]

# Prohibited comparative marketing phrases
COMPARATIVE_PATTERNS = [
    r"\bbetter than\b",
    r"\bkills the competition\b",
    r"\bbeats\s+(every|all|other)\b",
    r"\bunlike\s+\w+\b",
    r"\breplaces\s+(your)?\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bdestroy(s)?\s+the\s+competition\b"
]

# Deceptive endorsements & pseudo-scientific claims
ENDORSEMENT_PATTERNS = [
    r"\bdoctor\s+(recommended|approved|endorsed)\b",
    r"\bphysician\s+(recommended|approved)\b",
    r"\bfda\s+(approved|cleared|certified)\b",
    r"\b(nasa|who|cdc)\s+certified\b",
    r"\bdentist\s+(recommended|approved)\b",
    r"\bclinically\s+proven\b",
    r"\bscientifically\s+proven\b",
    r"\bcelebrity\s+endorsed\b",
    r"\bofficially\s+certified\b"
]

# Absolute claims, superlatives, and false guarantees
SUPERLATIVE_PATTERNS = [
    r"\bthe\s+best\b",
    r"\bbest\b",
    r"#1\b",
    r"\bnumber\s+(one|1)\b",
    r"\bworld('?s)?\s+best\b",
    r"\bfinest\b",
    r"\bgreatest\b",
    r"\bmost\s+advanced\b",
    r"\bunbeatable\b",
    r"\bflawless\b",
    r"\bultimate\b",
    r"\bperfect\b"
]

FALSE_CLAIM_PATTERNS = [
    r"\bguaranteed\b",
    r"\bguarantee\b",
    r"\b100%\b",
    r"\b100\s+percent\b",
    r"\brisk[\s\-]free\b",
    r"\bzero\s+risk\b",
    r"\bmoney[\s\-]back\b",
    r"\bfailproof\b",
    r"\bmiracle\b"
]

# Medical & health cure claims
MEDICAL_CLAIM_PATTERNS = [
    r"\bcure(s)?\b",
    r"\btreat(s)?\b",
    r"\bheal(s)?\b",
    r"\bprevent(s)?\s+disease\b",
    r"\bprevent(s)?\s+illness\b",
    r"\bdetox(ify|ifies|ification)?\b",
    r"\beliminate(s)?\s+toxins\b",
    r"\bweight\s+loss\s+guaranteed\b"
]


def check_rule_compliance(tagline: str) -> Dict[str, Any]:
    """Execute rigorous rule-based compliance checks.
    
    Returns:
        dict: {
            "is_compliant": bool,
            "flags": list of violation descriptions,
            "category_violations": dict of category flags
        }
    """
    clean_text = tagline.strip().strip('"').strip("'")
    lower = clean_text.lower()
    flags = []
    category_violations = {
        "trademark": [],
        "comparative": [],
        "endorsement": [],
        "superlative": [],
        "false_guarantee": [],
        "medical": [],
        "word_count": []
    }

    # 1. Word count constraint
    words = clean_text.split()
    if len(words) > 8:
        msg = f"Length violation: {len(words)} words exceeds 8 words limit."
        flags.append(msg)
        category_violations["word_count"].append(msg)

    # 2. Trademark check
    for tm in PROTECTED_TRADEMARKS:
        pattern = r"\b" + re.escape(tm) + r"\b"
        if re.search(pattern, lower):
            msg = f"Trademark violation: unauthorized use of protected brand '{tm}'."
            flags.append(msg)
            category_violations["trademark"].append(msg)

    # 3. Comparative marketing check
    for cp in COMPARATIVE_PATTERNS:
        if re.search(cp, lower):
            msg = f"Comparative claim violation: unsubstantiated competitor comparison matched by '{cp}'."
            flags.append(msg)
            category_violations["comparative"].append(msg)

    # 4. Endorsement check
    for ep in ENDORSEMENT_PATTERNS:
        if re.search(ep, lower):
            msg = f"False endorsement violation: unauthorized authority claim matched by '{ep}'."
            flags.append(msg)
            category_violations["endorsement"].append(msg)

    # 5. Superlatives check
    for sp in SUPERLATIVE_PATTERNS:
        if re.search(sp, lower):
            msg = f"Superlative violation: unsubstantiated absolute superiority claim matched by '{sp}'."
            flags.append(msg)
            category_violations["superlative"].append(msg)

    # 6. False guarantees check
    for gp in FALSE_CLAIM_PATTERNS:
        if re.search(gp, lower):
            msg = f"False guarantee violation: misleading risk/guarantee assertion matched by '{gp}'."
            flags.append(msg)
            category_violations["false_guarantee"].append(msg)

    # 7. Medical claims check
    for mp in MEDICAL_CLAIM_PATTERNS:
        if re.search(mp, lower):
            msg = f"Medical claim violation: unlawful therapeutic or medical claim matched by '{mp}'."
            flags.append(msg)
            category_violations["medical"].append(msg)

    is_compliant = len(flags) == 0
    return {
        "is_compliant": is_compliant,
        "flags": flags,
        "category_violations": category_violations,
        "score": 100 if is_compliant else max(10, 100 - len(flags) * 25)
    }


def review_compliance_llm(
    tagline: str,
    product_name: str = "AquaPulse",
    audience: str = "fitness enthusiasts",
    api_key: Optional[str] = None,
    mock_mode: Optional[bool] = None
) -> Dict[str, Any]:
    """Execute LLM reviewer check for nuanced deceptive marketing practices."""
    env_mock = os.getenv("MOCK_LLM", "").lower() in ("true", "1", "yes")
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    if mock_mode is None:
        mock_mode = env_mock or not bool(resolved_api_key.strip())

    # In mock mode, emulate reviewer based on rule check
    if mock_mode:
        rule_res = check_rule_compliance(tagline)
        return {
            "is_compliant": rule_res["is_compliant"],
            "flags": rule_res["flags"],
            "score": rule_res["score"],
            "suggestion": "Keep claims objective and focused on hydration habit tracking." if not rule_res["is_compliant"] else "",
            "is_mock": True
        }

    # Live Gemini API reviewer
    from google import genai
    from google.genai import types

    user_prompt = COMPLIANCE_USER_TEMPLATE.format(
        product_name=product_name,
        tagline=tagline,
        audience=audience
    )

    try:
        client = genai.Client(api_key=resolved_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=COMPLIANCE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.2,
            )
        )
        data = json.loads(response.text.strip())
        return {
            "is_compliant": bool(data.get("is_compliant", True)),
            "flags": data.get("flags", []),
            "score": int(data.get("score", 100)),
            "suggestion": data.get("suggestion", ""),
            "is_mock": False
        }
    except Exception as e:
        logger.warning(f"LLM compliance reviewer failed: {e}. Falling back to rule reviewer.")
        rule_res = check_rule_compliance(tagline)
        return {
            "is_compliant": rule_res["is_compliant"],
            "flags": rule_res["flags"],
            "score": rule_res["score"],
            "suggestion": "",
            "is_mock": True,
            "error": str(e)
        }


def check_compliance(
    tagline: str,
    product_name: str = "AquaPulse",
    audience: str = "fitness enthusiasts",
    api_key: Optional[str] = None,
    mock_mode: Optional[bool] = None
) -> Dict[str, Any]:
    """Unified compliance evaluation combining rule-based heuristics and LLM review."""
    rule_res = check_rule_compliance(tagline)
    
    # If rule check caught definite violations, combine them directly
    llm_res = review_compliance_llm(
        tagline=tagline,
        product_name=product_name,
        audience=audience,
        api_key=api_key,
        mock_mode=mock_mode
    )

    all_flags = list(dict.fromkeys(rule_res["flags"] + llm_res.get("flags", [])))
    is_compliant = (len(all_flags) == 0) and rule_res["is_compliant"] and llm_res.get("is_compliant", True)
    
    final_score = min(rule_res["score"], llm_res.get("score", 100))

    return {
        "tagline": tagline,
        "is_compliant": is_compliant,
        "score": final_score,
        "flags": all_flags,
        "rule_details": rule_res,
        "llm_details": llm_res,
        "suggestion": llm_res.get("suggestion", "")
    }


def regenerate_compliant_tagline(
    bad_tagline: str,
    reasons: List[str],
    product_name: str,
    description: str,
    audience_key: str = "fitness enthusiasts",
    tone_key: str = "energetic",
    api_key: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    max_retries: int = 2
) -> Tuple[str, Dict[str, Any]]:
    """Regenerate a compliant tagline if a banner's tagline fails compliance (max 2 retries).
    
    Returns:
        (new_tagline, final_compliance_report)
    """
    logger.info(f"Regenerating non-compliant tagline '{bad_tagline}'. Reasons: {reasons}")
    
    # Try up to max_retries
    for attempt in range(1, max_retries + 1):
        gen_res = generate_taglines(
            product_name=product_name,
            description=description,
            audience_key=audience_key,
            tone_key=tone_key,
            api_key=api_key,
            mock_mode=mock_mode
        )
        
        # Test candidate taglines from the batch
        for candidate in gen_res.get("taglines", []):
            report = check_compliance(
                tagline=candidate,
                product_name=product_name,
                audience=audience_key,
                api_key=api_key,
                mock_mode=mock_mode
            )
            if report["is_compliant"]:
                logger.info(f"Regeneration attempt {attempt} succeeded with compliant tagline: '{candidate}'")
                return candidate, report

    # Safe fallback if retries exhaust
    safe_fallback = f"Smart Hydration for {product_name}."
    report = check_compliance(safe_fallback, product_name, audience_key, api_key, mock_mode=True)
    return safe_fallback, report

