"""Prompt templates, audience profiles, and tone presets for ad creative generation."""

from typing import Dict, Any

# Audience profiles with psychographic characteristics, visual preferences, and pain points
AUDIENCE_PROFILES: Dict[str, Dict[str, Any]] = {
    "students": {
        "name": "Students",
        "description": "Gen-Z and college students, budget-conscious, trend-driven, tech-savvy, valuing convenience and social connection.",
        "visual_style": "Vibrant colors, modern dorm/campus setting, casual, playful aesthetics, dynamic angles.",
        "motivations": "Affordability, portability, stylish look, study boost, social status.",
        "keywords": ["vibrant", "campus life", "fresh", "budget-friendly", "trendy", "social"]
    },
    "young professionals": {
        "name": "Young Professionals",
        "description": "Career-focused millennials/early Gen-Z, urban dwellers, productivity-obsessed, valuing clean aesthetics and efficiency.",
        "visual_style": "Sleek workspace, modern architectural lines, understated neutrals, morning commute vibe, premium tech look.",
        "motivations": "Productivity, seamless daily routine, career elevation, sleek design, sustainability.",
        "keywords": ["minimalist", "sleek", "urban", "executive", "efficient", "modern"]
    },
    "parents": {
        "name": "Parents",
        "description": "Busy caregivers seeking safety, reliability, durability, and stress reduction for family life.",
        "visual_style": "Warm natural lighting, cozy home/outdoor family environment, approachable textures, harmonious tones.",
        "motivations": "Safety, durable materials, time-saving, child wellness, peace of mind.",
        "keywords": ["nurturing", "safe", "reliable", "family-first", "wholesome", "calm"]
    },
    "fitness enthusiasts": {
        "name": "Fitness Enthusiasts",
        "description": "Athletes, gym-goers, and active individuals prioritizing endurance, optimal hydration, performance, and grit.",
        "visual_style": "High-contrast dynamic lighting, energetic athletic environment (gym, trail, track), motion blur accents, vibrant neon accents.",
        "motivations": "Peak performance, hydration tracking, durability, recovery, endurance.",
        "keywords": ["high-energy", "athletic", "endurance", "dynamic", "peak performance", "bold"]
    },
    "seniors": {
        "name": "Seniors",
        "description": "Active older adults seeking health, wellness, ergonomic comfort, simplicity, and trusted quality.",
        "visual_style": "Soft warm sunlight, serene botanical garden or relaxing home setting, clear clean view, peaceful atmosphere.",
        "motivations": "Ease of use, health monitoring, joint-friendly ergonomics, peace of mind, longevity.",
        "keywords": ["wellness", "serene", "comfort", "gentle", "trusted", "clear"]
    }
}

# Tone presets with stylistic guidelines
TONE_PRESETS: Dict[str, Dict[str, Any]] = {
    "playful": {
        "name": "Playful",
        "voice": "Fun, quirky, cheerful, spirited, conversational, and energetic.",
        "visual_mood": "Bright pops of color, whimsical pastel/neon contrast, cheerful ambiance, friendly lighting."
    },
    "luxury": {
        "name": "Luxury",
        "voice": "Sophisticated, exclusive, refined, understated elegance, aspirational.",
        "visual_mood": "Deep dramatic tones (obsidian, rich gold, velvety shadows), studio rim-lighting, pristine surfaces."
    },
    "minimal": {
        "name": "Minimal",
        "voice": "Crisp, uncluttered, direct, essential, quiet confidence.",
        "visual_mood": "Monochromatic clean surfaces, generous negative space, soft ambient shadows, Scandinavian design ethos."
    },
    "energetic": {
        "name": "Energetic",
        "voice": "Bold, punchy, high-octane, motivating, action-oriented.",
        "visual_mood": "Electrifying gradients (cobalt, neon cyan, bright ember), dynamic diagonal lighting, athletic energy."
    },
    "emotional": {
        "name": "Emotional",
        "voice": "Heartfelt, touching, meaningful, human-centered, inspiring.",
        "visual_mood": "Golden hour glow, cinematic warmth, gentle haze, empathetic storytelling backdrop."
    },
    "witty": {
        "name": "Witty",
        "voice": "Clever, sharp, tongue-in-cheek, memorable, smart wordplay.",
        "visual_mood": "Clean editorial style, bold graphic backdrop, modern studio pop-art aesthetic."
    }
}

# Tagline prompt template for LLM
TAGLINE_SYSTEM_PROMPT = """You are an elite advertising copywriter specializing in high-converting ad banners.
Your job is to generate exactly 3 punchy, memorable, and legally compliant taglines for a digital banner ad.

STRICT CONSTRAINTS:
1. Length: Each tagline must be at most 8 words. Never exceed 8 words.
2. Compliance rules:
   - Do NOT use superlative or unsubstantiated words: "best", "#1", "world's best", "finest", "guaranteed", "100%", "miracle", "risk-free".
   - Do NOT make medical or disease claims (e.g. "cures", "treats", "prevents").
   - Do NOT mention competitor brands (e.g. Nike, Hydro Flask, Yeti, Stanley, Apple, CamelBak).
   - Do NOT claim fake endorsements ("doctor approved", "FDA certified").
3. Variety: Provide 3 distinctly different angles (e.g., benefit-driven, action-oriented, lifestyle-focused).
4. Tone & Audience: Tailor closely to the specified audience profile and tone.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{
  "taglines": [
    "First tagline here",
    "Second tagline here",
    "Third tagline here"
  ]
}
"""

TAGLINE_USER_TEMPLATE = """Generate exactly 3 compliant ad taglines for the following product:

Product Name: {product_name}
Description: {description}
Target Audience: {audience_name} ({audience_desc})
Tone: {tone_name} ({tone_voice})

Remember: Maximum 8 words per tagline. Return valid JSON only with 3 taglines.
"""

# Background generation prompt template for SD-Turbo
BACKGROUND_PROMPT_TEMPLATE = (
    "commercial advertising product photography background, {visual_mood}, {visual_style}, "
    "clean empty negative space on the left for typography, subtle surface on the right for product display, "
    "professional studio lighting, 8k resolution, cinematic atmosphere, no people, no text, no words, no watermark"
)

BACKGROUND_NEGATIVE_PROMPT = (
    "text, watermark, typography, letters, logo, blurry, messy, low quality, artifacts, distorted, cluttered, people"
)

# Compliance review prompt template for LLM
COMPLIANCE_SYSTEM_PROMPT = """You are a strict advertising compliance legal reviewer.
Inspect the provided advertising tagline for legal, trademark, and truth-in-advertising risks.

Check for:
1. Trademark misuse or competitor mentions.
2. False or unsubstantiated claims (e.g. "best", "#1", "100%", "guaranteed", "risk-free").
3. Medical or health claims ("cures", "treats", "prevents", "clinically proven").
4. Fake endorsements ("doctor recommended", "expert certified").

Respond ONLY with a JSON object:
{
  "is_compliant": true or false,
  "flags": ["List of identified violations, or empty if compliant"],
  "score": integer from 0 to 100 (100 = fully compliant),
  "suggestion": "Safer compliant revision if non-compliant, or empty"
}
"""

COMPLIANCE_USER_TEMPLATE = """Evaluate this advertising copy for legal compliance:
Product: {product_name}
Tagline: "{tagline}"
Target Audience: {audience}
"""

