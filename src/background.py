"""Background generation and CLIP scoring module.

Generates 4 background candidates per banner with Stable Diffusion (or Mock Mode)
and selects the best candidate using CLIP multimodal scoring with text/watermark penalties.
Logs all candidate metrics to outputs/logs/clip_candidate_scores.csv.
"""

import os
import csv
import math
import random
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from .prompts import (
    AUDIENCE_PROFILES,
    TONE_PRESETS,
    BACKGROUND_PROMPT_TEMPLATE,
    BACKGROUND_NEGATIVE_PROMPT,
)

logger = logging.getLogger(__name__)

# Base project directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
CANDIDATES_DIR = LOGS_DIR / "candidates"
CSV_LOG_FILE = LOGS_DIR / "clip_candidate_scores.csv"

# Global lazy-loaded models for diffusers/clip
_sd_pipeline = None
_clip_model = None
_clip_processor = None
_device = None


def detect_device() -> str:
    """Auto-detect optimal compute device (MPS on Apple Silicon, CUDA on Nvidia, or CPU)."""
    global _device
    if _device is not None:
        return _device

    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    except ImportError:
        _device = "cpu"

    logger.info(f"Auto-detected compute device: {_device}")
    return _device


def _init_csv_log():
    """Ensure CSV logging file exists with proper headers."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CSV_LOG_FILE.exists():
        with open(CSV_LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "banner_index",
                "candidate_id",
                "audience",
                "tone",
                "relevance_score",
                "text_penalty",
                "net_score",
                "selected",
                "image_path",
                "prompt"
            ])


def log_candidates_to_csv(candidate_records: List[Dict[str, Any]]):
    """Append candidate evaluation results to CSV."""
    _init_csv_log()
    with open(CSV_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for rec in candidate_records:
            writer.writerow([
                rec.get("timestamp", datetime.datetime.now().isoformat()),
                rec.get("banner_index", 1),
                rec.get("candidate_id", "c1"),
                rec.get("audience", ""),
                rec.get("tone", ""),
                f"{rec.get('relevance_score', 0.0):.4f}",
                f"{rec.get('text_penalty', 0.0):.4f}",
                f"{rec.get('net_score', 0.0):.4f}",
                rec.get("selected", False),
                rec.get("image_path", ""),
                rec.get("prompt", "")
            ])


# Color palettes for mock backgrounds tailored to tone and audience
PALETTES = {
    "energetic": [
        ((10, 24, 60), (0, 180, 216), (255, 94, 0)),     # Cobalt to cyan with energetic orange flare
        ((15, 23, 42), (56, 189, 248), (34, 197, 94)),   # Slate to sky blue with electric lime
        ((30, 27, 75), (99, 102, 241), (236, 72, 153)),   # Deep indigo to neon violet / punchy pink
        ((17, 24, 39), (249, 115, 22), (234, 179, 8)),    # Athletic charcoal to blazing amber
    ],
    "playful": [
        ((255, 238, 217), (255, 159, 128), (255, 99, 132)), # Peachy coral cheerful vibe
        ((224, 242, 254), (125, 211, 252), (167, 139, 250)),# Sky blue to pastel lilac
        ((240, 253, 244), (74, 222, 128), (250, 204, 21)),  # Fresh mint with yellow pop
        ((253, 242, 248), (244, 114, 182), (251, 146, 60)), # Playful magenta sunrise
    ],
    "luxury": [
        ((12, 12, 14), (35, 35, 42), (212, 175, 55)),     # Obsidian black with gold studio rim
        ((18, 20, 24), (40, 48, 60), (180, 190, 205)),    # Brushed platinum midnight
        ((20, 10, 15), (60, 25, 35), (220, 180, 140)),    # Deep burgundy and champagne
        ((10, 18, 16), (28, 48, 42), (165, 214, 167)),    # Deep emerald studio
    ],
    "minimal": [
        ((245, 247, 250), (226, 232, 240), (203, 213, 225)),# Clean Scandinavian light grey
        ((255, 255, 255), (241, 245, 249), (148, 163, 184)),# Crisp neutral white with soft shadow
        ((30, 32, 38), (45, 49, 58), (71, 85, 105)),        # Dark architectural charcoal
        ((248, 250, 252), (218, 225, 231), (180, 195, 205)),# Subtle slate studio surface
    ],
    "emotional": [
        ((42, 24, 16), (217, 119, 6), (245, 158, 11)),     # Golden hour cinematic amber
        ((30, 27, 75), (147, 51, 234), (244, 63, 94)),     # Twilight violet & rose
        ((254, 243, 199), (252, 211, 77), (249, 115, 22)), # Gentle warm morning glow
        ((15, 23, 42), (100, 116, 139), (226, 232, 240)),  # Serene contemplative horizon
    ],
    "witty": [
        ((24, 24, 27), (250, 204, 21), (39, 39, 42)),      # High contrast bold yellow & graphite
        ((15, 23, 42), (59, 130, 246), (239, 68, 68)),     # Sharp dynamic cobalt & electric crimson
        ((244, 244, 245), (161, 161, 170), (24, 24, 27)),  # Bold modern monochrome print
        ((23, 37, 84), (56, 189, 248), (250, 204, 21)),    # Clever cyan/amber punch
    ]
}


def _create_mock_gradient(
    width: int,
    height: int,
    palette_stops: Tuple[Tuple[int, int, int], ...],
    style_idx: int = 0
) -> Image.Image:
    """Generate a high-quality 512x512 mock background with studio lighting & subtle vignette."""
    c1, c2, c3 = palette_stops
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Gradient types: 0=diagonal, 1=horizontal studio, 2=radial spotlight on right, 3=bottom spotlight
    if style_idx == 0:
        # Diagonal gradient from top-left to bottom-right
        for y in range(height):
            for x in range(0, width, 2):
                t = (x + y) / (width + height)
                if t < 0.5:
                    sub_t = t * 2.0
                    r = int(c1[0] * (1 - sub_t) + c2[0] * sub_t)
                    g = int(c1[1] * (1 - sub_t) + c2[1] * sub_t)
                    b = int(c1[2] * (1 - sub_t) + c2[2] * sub_t)
                else:
                    sub_t = (t - 0.5) * 2.0
                    r = int(c2[0] * (1 - sub_t) + c3[0] * sub_t)
                    g = int(c2[1] * (1 - sub_t) + c3[1] * sub_t)
                    b = int(c2[2] * (1 - sub_t) + c3[2] * sub_t)
                draw.line([(x, y), (x + 1, y)], fill=(r, g, b))
    elif style_idx == 1:
        # Horizontal gradient: Left is clean negative space (c1), right has spotlight surface (c2->c3)
        for x in range(width):
            t = x / width
            if t < 0.6:
                sub_t = t / 0.6
                r = int(c1[0] * (1 - sub_t) + c2[0] * sub_t)
                g = int(c1[1] * (1 - sub_t) + c2[1] * sub_t)
                b = int(c1[2] * (1 - sub_t) + c2[2] * sub_t)
            else:
                sub_t = (t - 0.6) / 0.4
                r = int(c2[0] * (1 - sub_t) + c3[0] * sub_t)
                g = int(c2[1] * (1 - sub_t) + c3[1] * sub_t)
                b = int(c2[2] * (1 - sub_t) + c3[2] * sub_t)
            draw.line([(x, 0), (x, height)], fill=(r, g, b))
    elif style_idx == 2:
        # Radial spotlight positioned on the right (where product will stand)
        center_x = int(width * 0.75)
        center_y = int(height * 0.55)
        max_dist = math.sqrt(center_x**2 + center_y**2)
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        y_coords, x_coords = np.ogrid[:height, :width]
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        norm_dist = np.clip(dist / max_dist, 0.0, 1.0)
        
        # Center glow (c3) -> mid (c2) -> edge (c1)
        for ch in range(3):
            arr[:, :, ch] = (c3[ch] * (1.0 - norm_dist) + c1[ch] * norm_dist).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        # Two-tone studio horizon with soft reflection line
        horizon_y = int(height * 0.65)
        for y in range(height):
            if y < horizon_y:
                t = y / horizon_y
                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)
            else:
                t = (y - horizon_y) / (height - horizon_y)
                r = int(c2[0] * (1 - t) + c3[0] * t)
                g = int(c2[1] * (1 - t) + c3[1] * t)
                b = int(c2[2] * (1 - t) + c3[2] * t)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Add a soft blur for smooth studio finish
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img


def _generate_mock_candidates(
    prompt: str,
    audience: str,
    tone: str,
    banner_index: int,
    candidate_count: int = 4
) -> List[Dict[str, Any]]:
    """Generate 4 mock background candidates and evaluate with realistic CLIP scoring."""
    palette_list = PALETTES.get(tone, PALETTES["energetic"])
    candidates = []
    _init_csv_log()

    # Pre-calculated deterministic variations for reproducible, realistic scoring
    # Candidate 2 or 3 usually wins because Candidate 4 might simulate slight watermark artifact
    base_relevance = [0.325, 0.362, 0.348, 0.295]
    text_penalties = [0.042, 0.021, 0.035, 0.145]  # Candidate 4 penalized for simulated watermark

    for i in range(candidate_count):
        cand_id = f"b{banner_index}_c{i+1}"
        palette = palette_list[i % len(palette_list)]
        img = _create_mock_gradient(512, 512, palette, style_idx=i)
        
        # Save image candidate
        img_filename = f"banner_{banner_index}_candidate_{i+1}.png"
        img_path = CANDIDATES_DIR / img_filename
        img.save(img_path, format="PNG")

        rel_score = base_relevance[i]
        txt_pen = text_penalties[i]
        # Net score penalizes text/watermark heavily (weight 0.6)
        net_score = rel_score - (0.6 * txt_pen)

        candidates.append({
            "candidate_id": cand_id,
            "banner_index": banner_index,
            "image": img,
            "image_path": str(img_path),
            "relevance_score": rel_score,
            "text_penalty": txt_pen,
            "net_score": net_score,
            "audience": audience,
            "tone": tone,
            "prompt": prompt,
            "selected": False,
            "timestamp": datetime.datetime.now().isoformat()
        })

    # Pick candidate with highest net score
    best_candidate = max(candidates, key=lambda c: c["net_score"])
    best_candidate["selected"] = True

    # Log to CSV
    log_candidates_to_csv(candidates)
    return candidates


def _generate_diffusers_candidates(
    prompt: str,
    audience: str,
    tone: str,
    banner_index: int,
    candidate_count: int = 4
) -> List[Dict[str, Any]]:
    """Generate candidates using diffusers (SD-turbo) and score using real CLIP."""
    global _sd_pipeline, _clip_model, _clip_processor
    import torch
    from diffusers import AutoPipelineForText2Image
    from transformers import CLIPProcessor, CLIPModel

    device = detect_device()
    torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    # Load SD-Turbo if not cached
    if _sd_pipeline is None:
        logger.info(f"Loading SD-Turbo pipeline on {device}...")
        _sd_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch_dtype,
            variant="fp16" if device == "cuda" else None
        )
        _sd_pipeline.to(device)

    # Load CLIP if not cached
    if _clip_model is None or _clip_processor is None:
        logger.info(f"Loading CLIP model (openai/clip-vit-base-patch32) on {device}...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Encode target text and penalty text
    aud_desc = AUDIENCE_PROFILES.get(audience, {}).get("description", audience)
    target_text = f"commercial ad background for {audience}, {tone} style, {aud_desc}"
    penalty_text = "watermark, text, typography, letters, logo, blurry writing"

    text_inputs = _clip_processor(
        text=[target_text, penalty_text],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = _clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        target_embed = text_features[0:1]
        penalty_embed = text_features[1:2]

    candidates = []
    _init_csv_log()

    for i in range(candidate_count):
        cand_id = f"b{banner_index}_c{i+1}"
        generator = torch.Generator(device=device).manual_seed(1000 * banner_index + i * 42)
        
        # SD-Turbo operates best with 1-4 steps and guidance_scale=0.0
        sd_out = _sd_pipeline(
            prompt=prompt,
            negative_prompt=BACKGROUND_NEGATIVE_PROMPT,
            num_inference_steps=2,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=generator
        )
        img = sd_out.images[0]

        # Save image candidate
        img_filename = f"banner_{banner_index}_candidate_{i+1}.png"
        img_path = CANDIDATES_DIR / img_filename
        img.save(img_path, format="PNG")

        # CLIP scoring
        img_inputs = _clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = _clip_model.get_image_features(**img_inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            rel_score = float(torch.matmul(img_features, target_embed.T).squeeze().cpu().item())
            txt_pen = float(torch.matmul(img_features, penalty_embed.T).squeeze().cpu().item())

        net_score = rel_score - (0.5 * txt_pen)

        candidates.append({
            "candidate_id": cand_id,
            "banner_index": banner_index,
            "image": img,
            "image_path": str(img_path),
            "relevance_score": rel_score,
            "text_penalty": txt_pen,
            "net_score": net_score,
            "audience": audience,
            "tone": tone,
            "prompt": prompt,
            "selected": False,
            "timestamp": datetime.datetime.now().isoformat()
        })

    # Pick candidate with highest net score
    best_candidate = max(candidates, key=lambda c: c["net_score"])
    best_candidate["selected"] = True

    # Log to CSV
    log_candidates_to_csv(candidates)
    return candidates


def generate_and_select_background(
    audience: str = "fitness enthusiasts",
    tone: str = "energetic",
    banner_index: int = 1,
    candidate_count: int = 4,
    backend: Optional[str] = None
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Generate 4 candidate backgrounds, score them using CLIP, and return the best image.
    
    Returns:
        (best_image: PIL.Image, all_candidates: list of candidate dicts)
    """
    env_backend = os.getenv("IMAGE_BACKEND", "mock").lower()
    active_backend = backend or env_backend

    aud_info = AUDIENCE_PROFILES.get(audience, AUDIENCE_PROFILES["fitness enthusiasts"])
    tone_info = TONE_PRESETS.get(tone, TONE_PRESETS["energetic"])
    
    prompt = BACKGROUND_PROMPT_TEMPLATE.format(
        visual_mood=tone_info["visual_mood"],
        visual_style=aud_info["visual_style"]
    )

    if active_backend == "diffusers":
        try:
            candidates = _generate_diffusers_candidates(
                prompt=prompt,
                audience=audience,
                tone=tone,
                banner_index=banner_index,
                candidate_count=candidate_count
            )
        except Exception as e:
            logger.warning(f"Diffusers background generation failed ({e}). Falling back to mock generator.")
            candidates = _generate_mock_candidates(
                prompt=prompt,
                audience=audience,
                tone=tone,
                banner_index=banner_index,
                candidate_count=candidate_count
            )
    else:
        candidates = _generate_mock_candidates(
            prompt=prompt,
            audience=audience,
            tone=tone,
            banner_index=banner_index,
            candidate_count=candidate_count
        )

    selected = next(c for c in candidates if c["selected"])
    return selected["image"], candidates
