"""End-to-end Ad Creative Generation Pipeline.

Orchestrates prompts, tagline generation, background generation + CLIP candidate selection,
banner composition, and compliance verification.
"""

import os
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from PIL import Image

from .prompts import AUDIENCE_PROFILES, TONE_PRESETS
from .tagline import generate_taglines
from .background import generate_and_select_background
from .compose import compose_banner, BANNER_WIDTH, BANNER_HEIGHT
from .compliance import check_compliance, regenerate_compliant_tagline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BANNERS_DIR = OUTPUTS_DIR / "banners"


def run_pipeline(
    product_name: str,
    description: str,
    product_image: Image.Image,
    audience: str = "fitness enthusiasts",
    tone: str = "energetic",
    api_key: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    backend: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """Execute full 5-stage ad creative generation pipeline.
    
    Generates 3 ad banners (1200x628 PNG) with AI background, product cutout,
    auto-contrast tagline, soft shadow, CTA button, and compliance certification.
    
    Returns:
        dict: {
            "banners": [Image, Image, Image],
            "banner_paths": [str, str, str],
            "taglines": [str, str, str],
            "compliance_reports": [dict, dict, dict],
            "clip_candidates_per_banner": [list, list, list],
            "metadata": dict
        }
    """
    BANNERS_DIR.mkdir(parents=True, exist_ok=True)

    def _notify(progress: float, message: str):
        if progress_callback:
            progress_callback(progress, message)
        logger.info(f"[{int(progress * 100)}%] {message}")

    _notify(0.05, "Initializing pipeline and validating audience & tone presets...")
    
    # Normalize audience and tone keys
    aud_key = audience.lower()
    if aud_key not in AUDIENCE_PROFILES:
        aud_key = "fitness enthusiasts"
    tone_key = tone.lower()
    if tone_key not in TONE_PRESETS:
        tone_key = "energetic"

    # Step 1: Generate 3 taglines
    _notify(0.15, "Generating 3 distinct ad taglines (Gemini API / Mock)...")
    tag_result = generate_taglines(
        product_name=product_name,
        description=description,
        audience_key=aud_key,
        tone_key=tone_key,
        api_key=api_key,
        mock_mode=mock_mode
    )
    raw_taglines = tag_result.get("taglines", [])
    if len(raw_taglines) < 3:
        raise RuntimeError(f"Pipeline expected 3 taglines, received {len(raw_taglines)}.")

    banners: List[Image.Image] = []
    banner_paths: List[str] = []
    final_taglines: List[str] = []
    compliance_reports: List[Dict[str, Any]] = []
    clip_candidates_per_banner: List[List[Dict[str, Any]]] = []

    # Process each of the 3 banners
    for idx in range(1, 4):
        banner_tagline = raw_taglines[idx - 1]
        progress_base = 0.20 + (idx - 1) * 0.25

        _notify(progress_base + 0.05, f"Banner {idx}/3: Running compliance verification...")
        comp_report = check_compliance(
            tagline=banner_tagline,
            product_name=product_name,
            audience=aud_key,
            api_key=api_key,
            mock_mode=mock_mode
        )

        # If compliance failed, trigger automatic regeneration (max 2 retries)
        if not comp_report["is_compliant"]:
            _notify(progress_base + 0.08, f"Banner {idx}/3: Non-compliant tagline detected. Regenerating...")
            banner_tagline, comp_report = regenerate_compliant_tagline(
                bad_tagline=banner_tagline,
                reasons=comp_report["flags"],
                product_name=product_name,
                description=description,
                audience_key=aud_key,
                tone_key=tone_key,
                api_key=api_key,
                mock_mode=mock_mode
            )

        final_taglines.append(banner_tagline)
        compliance_reports.append(comp_report)

        # Step 2: Generate 4 background candidates and score with CLIP
        _notify(progress_base + 0.12, f"Banner {idx}/3: Generating 4 background candidates & scoring with CLIP...")
        best_bg, candidates = generate_and_select_background(
            audience=aud_key,
            tone=tone_key,
            banner_index=idx,
            candidate_count=4,
            backend=backend
        )
        clip_candidates_per_banner.append(candidates)

        # Step 3: Compose final 1200x628 banner
        _notify(progress_base + 0.20, f"Banner {idx}/3: Compositing banner (auto-contrast text, cutout, shadow, CTA)...")
        banner_img = compose_banner(
            background_img=best_bg,
            product_img=product_image,
            tagline=banner_tagline,
            product_name=product_name,
            cta_text="Shop Now →"
        )
        
        # Save output banner
        out_filename = f"banner_{idx}.png"
        out_path = BANNERS_DIR / out_filename
        banner_img.save(out_path, format="PNG")
        banners.append(banner_img)
        banner_paths.append(str(out_path))

    _notify(1.0, "All 3 ad banners generated and certified successfully!")

    return {
        "banners": banners,
        "banner_paths": banner_paths,
        "taglines": final_taglines,
        "compliance_reports": compliance_reports,
        "clip_candidates_per_banner": clip_candidates_per_banner,
        "metadata": {
            "product_name": product_name,
            "description": description,
            "audience": aud_key,
            "tone": tone_key,
            "timestamp": datetime.datetime.now().isoformat(),
            "banner_dimensions": f"{BANNER_WIDTH}x{BANNER_HEIGHT}",
            "is_mock": tag_result.get("is_mock", True)
        }
    }


def main():
    """Command-line entrypoint for headless pipeline execution."""
    import argparse
    parser = argparse.ArgumentParser(description="Product Ad Creative Generator Pipeline")
    parser.add_argument("--product", default="AquaPulse", help="Product name")
    parser.add_argument("--desc", default="Reusable smart water bottle with hydration tracking", help="Product description")
    parser.add_argument("--audience", default="fitness enthusiasts", help="Target audience key")
    parser.add_argument("--tone", default="energetic", help="Brand tone preset")
    parser.add_argument("--image", default=None, help="Path to product photo")
    parser.add_argument("--mock", action="store_true", default=None, help="Force mock mode")
    args = parser.parse_args()

    sample_img_path = PROJECT_ROOT / "assets" / "sample_product.png"
    if args.image and Path(args.image).exists():
        prod_img = Image.open(args.image)
    elif sample_img_path.exists():
        prod_img = Image.open(sample_img_path)
    else:
        prod_img = Image.new("RGBA", (400, 700), (0, 180, 216, 255))

    print(f"Starting pipeline for '{args.product}' ({args.audience}, {args.tone})...")
    res = run_pipeline(
        product_name=args.product,
        description=args.desc,
        product_image=prod_img,
        audience=args.audience,
        tone=args.tone,
        mock_mode=args.mock,
        progress_callback=lambda p, msg: print(f"[{int(p*100):3d}%] {msg}")
    )
    print("\nAd Banners Generated:")
    for i, p in enumerate(res["banner_paths"], 1):
        print(f"  {i}. {p} | Tagline: \"{res['taglines'][i-1]}\"")


if __name__ == "__main__":
    main()


