"""Tagline generation and validation module using Gemini API with retry logic and mock mode."""

import os
import json
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv

from .prompts import (
    AUDIENCE_PROFILES,
    TONE_PRESETS,
    TAGLINE_SYSTEM_PROMPT,
    TAGLINE_USER_TEMPLATE,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Banned tokens and phrases (case-insensitive)
BANNED_WORDS = [
    "best",
    "#1",
    "number 1",
    "number one",
    "guaranteed",
    "guarantee",
    "100%",
    "risk-free",
    "risk free",
    "miracle",
    "cure",
    "cures",
    "treats",
    "heals",
    "fda approved",
    "doctor recommended",
    "doctor approved",
    "clinically proven",
]

# Competitor names to prohibit in taglines
COMPETITOR_NAMES = [
    "hydro flask",
    "stanley",
    "yeti",
    "camelbak",
    "thermos",
    "nalgene",
    "swell",
    "s'well",
    "contigo",
    "nike",
    "adidas",
    "under armour",
    "lululemon",
    "gatorade",
    "apple",
    "fitbit",
    "garmin",
    "samsung",
]

# Curated mock taglines for test/offline mode
MOCK_TAGLINE_PRESETS: Dict[str, Dict[str, List[str]]] = {
    "fitness enthusiasts": {
        "energetic": [
            "Hydrate Smarter. Train Harder. Crush Goals.",
            "Precision Hydration for Unstoppable Energy.",
            "Fuel Every Rep with Intelligent Flow."
        ],
        "playful": [
            "Sip Happens. Make Every Drop Count!",
            "Your Workout Buddy That Never Flakes.",
            "Track Sips, Not Steps. Stay Radiant!"
        ],
        "minimal": [
            "Smart Hydration. Peak Performance.",
            "Pure Flow. Measured Daily.",
            "Form Meets Function in Every Sip."
        ],
        "luxury": [
            "Master Your Hydration. Elevate Performance.",
            "Engineered Excellence for Discerning Athletes.",
            "The Refined Standard in Smart Hydration."
        ],
        "emotional": [
            "Every Drop Fuels Your Next Breakthrough.",
            "Because Your Body Deserves Better Hydration.",
            "Stay Strong. Stay Connected. Stay You."
        ],
        "witty": [
            "Thirst Quenched. Excuse Evaporated.",
            "Smarter Than Your Average Water Bottle.",
            "Because Dehydration Looks Good on Nobody."
        ]
    }
}

DEFAULT_MOCK_TAGLINES = [
    "Intelligent Hydration for Peak Daily Life.",
    "Track Every Sip. Feel the Difference.",
    "Pure Hydration Engineered for Your Flow."
]


def validate_single_tagline(tagline: str) -> Tuple[bool, Optional[str]]:
    """Validate a single tagline against length and compliance rules.
    
    Returns:
        (is_valid, error_reason)
    """
    clean_text = tagline.strip().strip('"').strip("'")
    if not clean_text:
        return False, "Tagline cannot be empty."

    # Word count check: maximum 8 words
    words = clean_text.split()
    if len(words) > 8:
        return False, f"Tagline exceeds 8 words limit ({len(words)} words): '{clean_text}'"

    lower = clean_text.lower()

    # Superlatives, false claims, and medical checks
    for banned in BANNED_WORDS:
        pre_b = r"\b" if banned[0].isalnum() else ""
        suf_b = r"\b" if banned[-1].isalnum() else ""
        pattern = pre_b + re.escape(banned) + suf_b
        if re.search(pattern, lower):
            return False, f"Contains prohibited claim/superlative '{banned}'"

    # Competitor checks
    for comp in COMPETITOR_NAMES:
        pattern = r"\b" + re.escape(comp) + r"\b"
        if re.search(pattern, lower):
            return False, f"Contains competitor reference '{comp}'"

    return True, None


def validate_taglines(taglines: List[str]) -> Tuple[bool, List[str]]:
    """Validate a list of taglines. Must have exactly 3 distinct valid taglines.
    
    Returns:
        (all_valid, list_of_error_messages)
    """
    errors = []
    if not isinstance(taglines, list):
        return False, ["Taglines response is not a list."]

    if len(taglines) != 3:
        errors.append(f"Expected exactly 3 taglines, received {len(taglines)}.")

    # Distinctness check
    unique_set = {t.strip().lower() for t in taglines if isinstance(t, str)}
    if len(unique_set) < len(taglines):
        errors.append("Taglines must all be unique and distinct.")

    for i, tag in enumerate(taglines, 1):
        if not isinstance(tag, str):
            errors.append(f"Tagline #{i} is not a string.")
            continue
        valid, reason = validate_single_tagline(tag)
        if not valid:
            errors.append(f"Tagline #{i} invalid: {reason}")

    return len(errors) == 0, errors


def get_mock_taglines(product_name: str, audience_key: str, tone_key: str) -> List[str]:
    """Return realistic mock taglines for offline/mock mode."""
    if audience_key in MOCK_TAGLINE_PRESETS and tone_key in MOCK_TAGLINE_PRESETS[audience_key]:
        return MOCK_TAGLINE_PRESETS[audience_key][tone_key]
    
    # Generic template fallback
    return [
        f"{product_name}: Smarter Everyday Flow.",
        f"Track Your Rhythm with {product_name}.",
        f"Elevate Every Moment with {product_name}."
    ]


def _call_gemini_api(api_key: str, user_prompt: str, system_prompt: str) -> str:
    """Call Gemini API using google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    
    # Attempt with gemini-2.5-flash, fallback if needed
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    last_err = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.7,
                )
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            logger.warning(f"Model {model_name} invocation failed: {e}. Trying fallback...")

    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


def _parse_tagline_json(raw_text: str) -> List[str]:
    """Parse JSON output from LLM, handling markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    
    data = json.loads(text.strip())
    if isinstance(data, dict) and "taglines" in data:
        return [str(t).strip() for t in data["taglines"]]
    elif isinstance(data, list):
        return [str(t).strip() for t in data]
    raise ValueError(f"Unexpected JSON format from LLM: {raw_text[:200]}")


def generate_taglines(
    product_name: str,
    description: str,
    audience_key: str = "fitness enthusiasts",
    tone_key: str = "energetic",
    api_key: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generate exactly 3 valid ad taglines using Gemini API or mock mode.
    
    Validates word counts (<= 8 words) and compliance restrictions. Retries up to 3 times.
    
    Returns:
        dict: {
            "taglines": [tag1, tag2, tag3],
            "attempts": int,
            "is_mock": bool,
            "errors": list
        }
    """
    env_mock = os.getenv("MOCK_LLM", "").lower() in ("true", "1", "yes")
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    
    if mock_mode is None:
        mock_mode = env_mock or not bool(resolved_api_key.strip())

    if mock_mode:
        logger.info("Generating taglines in MOCK mode.")
        mock_tags = get_mock_taglines(product_name, audience_key, tone_key)
        valid, errors = validate_taglines(mock_tags)
        return {
            "taglines": mock_tags,
            "attempts": 1,
            "is_mock": True,
            "errors": errors if not valid else []
        }

    # Live Gemini API path
    if not resolved_api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set in environment or provided to generate_taglines. "
            "Set GEMINI_API_KEY in .env or run with MOCK_LLM=true."
        )

    aud_info = AUDIENCE_PROFILES.get(audience_key, AUDIENCE_PROFILES["fitness enthusiasts"])
    tone_info = TONE_PRESETS.get(tone_key, TONE_PRESETS["energetic"])

    base_user_prompt = TAGLINE_USER_TEMPLATE.format(
        product_name=product_name,
        description=description,
        audience_name=aud_info["name"],
        audience_desc=aud_info["description"],
        tone_name=tone_info["name"],
        tone_voice=tone_info["voice"]
    )

    history_feedback = ""
    last_errors = []

    for attempt in range(1, max_retries + 1):
        prompt = base_user_prompt
        if history_feedback:
            prompt += f"\n\nPREVIOUS ATTEMPT REJECTED FOR REASONS:\n{history_feedback}\nPlease correct these violations and regenerate 3 compliant taglines."

        try:
            raw_response = _call_gemini_api(resolved_api_key, prompt, TAGLINE_SYSTEM_PROMPT)
            taglines = _parse_tagline_json(raw_response)
            valid, errors = validate_taglines(taglines)
            
            if valid:
                return {
                    "taglines": taglines,
                    "attempts": attempt,
                    "is_mock": False,
                    "errors": []
                }
            else:
                last_errors = errors
                history_feedback = "\n".join(f"- {e}" for e in errors)
                logger.warning(f"Attempt {attempt} failed validation: {errors}")
        except Exception as e:
            last_errors = [str(e)]
            logger.error(f"Attempt {attempt} exception: {e}")
            history_feedback = f"API or parsing error: {e}"

    # If all retries failed, fallback gracefully to mock with informative message
    logger.warning("All Gemini tagline attempts failed; falling back to curated compliant taglines.")
    fallback_tags = get_mock_taglines(product_name, audience_key, tone_key)
    return {
        "taglines": fallback_tags,
        "attempts": max_retries,
        "is_mock": True,
        "fallback_used": True,
        "errors": last_errors
    }
