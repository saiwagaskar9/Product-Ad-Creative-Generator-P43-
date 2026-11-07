"""Banner composition engine.

Combines background, product cutout (with soft shadow), auto-contrast auto-wrapped
tagline, product brand header, and a 'Shop Now' call-to-action button into a 1200x628 PNG.
"""

import os
import math
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

logger = logging.getLogger(__name__)

# Standard social/web ad banner dimensions
BANNER_WIDTH = 1200
BANNER_HEIGHT = 628

# Candidate font locations for cross-platform compatibility
FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "arial.ttf",
    "helvetica.ttf"
]


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Load system font with graceful fallback to PIL default font."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                # index 0 is regular/bold depending on ttc
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def remove_product_background(product_img: Image.Image) -> Image.Image:
    """Remove background from product photo using rembg if available, or return RGBA cleanly."""
    img = product_img.convert("RGBA")

    # If rembg is installed, attempt AI background removal
    try:
        import rembg
        logger.info("Removing product background with rembg...")
        cleaned = rembg.remove(img)
        return cleaned
    except Exception as e:
        logger.debug(f"rembg removal skipped or not installed ({e}); applying fallback alpha processing.")

    # Fallback: If image has transparent pixels, respect them
    alpha = img.split()[-1]
    extrema = alpha.getextrema()
    if extrema[0] < 250:
        # Already has transparency
        return img

    # Fallback thresholding: If nearly solid white background, make white transparent
    arr = np.array(img)
    # Identify near-white background pixels (R>245, G>245, B>245)
    white_mask = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)
    arr[white_mask, 3] = 0
    return Image.fromarray(arr, "RGBA")


def calculate_region_luminance(image: Image.Image, box: Tuple[int, int, int, int]) -> float:
    """Calculate the average perceptual luminance (0-255) of an image within a bounding box.
    
    Formula: L = 0.299*R + 0.587*G + 0.114*B
    """
    cropped = image.crop(box).convert("RGB")
    arr = np.array(cropped, dtype=np.float32)
    # ITU-R BT.601 perceptual luminance formula
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return float(np.mean(lum))


def get_contrast_styling(luminance: float):
    """Determine text and CTA button colors based on background luminance."""
    if luminance < 135.0:
        # Dark background -> White text, vibrant accent button
        return {
            "is_dark_bg": True,
            "brand_color": (56, 189, 248),         # Electric cyan
            "tagline_color": (255, 255, 255),      # Crisp white
            "text_shadow": (0, 0, 0, 160),         # Dark drop shadow
            "button_bg": (0, 180, 216),            # Bright cyan button
            "button_text": (10, 15, 30),           # Deep navy text
            "button_border": (255, 255, 255, 100)
        }
    else:
        # Light background -> Dark text, dark sleek button
        return {
            "is_dark_bg": False,
            "brand_color": (14, 116, 144),        # Deep teal
            "tagline_color": (15, 23, 42),         # Slate black
            "text_shadow": (255, 255, 255, 180),   # Light halo shadow
            "button_bg": (15, 23, 42),             # Deep slate button
            "button_text": (255, 255, 255),        # White text
            "button_border": (0, 0, 0, 40)
        }


def fit_and_wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    initial_font_size: int = 54,
    min_font_size: int = 24
) -> Tuple[Any, List[str], int]:
    """Dynamically determine optimal font size and word wrapping to fit cleanly inside box.
    
    Returns:
        (font, wrapped_lines, total_text_height)
    """
    words = text.strip().split()
    font_size = initial_font_size

    while font_size >= min_font_size:
        font = load_font(font_size, bold=True)
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Single word exceeds max_width
                    lines.append(word)
                    current_line = []

        if current_line:
            lines.append(" ".join(current_line))

        # Measure line heights
        line_height = int(font_size * 1.25)
        total_height = len(lines) * line_height

        if total_height <= max_height and max(draw.textbbox((0, 0), ln, font=font)[2] for ln in lines) <= max_width:
            return font, lines, line_height

        font_size -= 2

    # Minimum size fallback
    font = load_font(min_font_size, bold=True)
    return font, lines, int(min_font_size * 1.25)


def create_soft_shadow(
    image: Image.Image,
    offset: Tuple[int, int] = (15, 22),
    blur_radius: int = 22,
    shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 110)
) -> Image.Image:
    """Create a realistic blurred soft drop shadow for a transparent RGBA image."""
    alpha = image.split()[-1]
    
    # Shadow canvas expanded by blur margin
    pad = blur_radius * 2
    shadow_w = image.width + pad * 2
    shadow_h = image.height + pad * 2
    
    shadow_img = Image.new("RGBA", (shadow_w, shadow_h), (0, 0, 0, 0))
    # Fill mask with shadow color
    color_block = Image.new("RGBA", image.size, shadow_color)
    shadow_img.paste(color_block, (pad, pad), mask=alpha)
    
    # Apply gaussian blur
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return shadow_img, pad


def compose_banner(
    background_img: Image.Image,
    product_img: Image.Image,
    tagline: str,
    product_name: str = "AquaPulse",
    cta_text: str = "Shop Now →",
    target_size: Tuple[int, int] = (BANNER_WIDTH, BANNER_HEIGHT)
) -> Image.Image:
    """Assemble final ad banner (1200x628 PNG).
    
    Layout:
    - Canvas: 1200x628
    - Right ~45%: Product photo with realistic soft drop shadow
    - Left ~55%:
        * Product Brand Header (small caps tracking)
        * Auto-wrapped auto-contrast Tagline (<= 8 words)
        * Sleek CTA Button ("Shop Now →")
    """
    width, height = target_size

    # 1. Resize and crop background to cover 1200x628 exactly
    bg = ImageOps.fit(background_img.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)
    canvas = bg.copy()

    # 2. Process product image
    clean_product = remove_product_background(product_img)
    
    # Scale product to fit comfortably in the right 45% (width ~460, height ~520)
    max_prod_w = int(width * 0.42)
    max_prod_h = int(height * 0.82)
    
    prod_w, prod_h = clean_product.size
    scale = min(max_prod_w / prod_w, max_prod_h / prod_h)
    new_prod_size = (max(10, int(prod_w * scale)), max(10, int(prod_h * scale)))
    scaled_prod = clean_product.resize(new_prod_size, Image.Resampling.LANCZOS)

    # Position product centered horizontally in the right column (x between 680 and 1120)
    prod_center_x = int(width * 0.73)
    prod_center_y = int(height * 0.52)
    prod_top_left = (prod_center_x - scaled_prod.width // 2, prod_center_y - scaled_prod.height // 2)

    # 3. Create and paste soft shadow behind product
    shadow_img, pad = create_soft_shadow(
        scaled_prod,
        offset=(12, 18),
        blur_radius=20,
        shadow_color=(0, 0, 0, 130)
    )
    shadow_top_left = (prod_top_left[0] - pad + 12, prod_top_left[1] - pad + 18)
    canvas.paste(shadow_img, shadow_top_left, mask=shadow_img)

    # Paste scaled product
    canvas.paste(scaled_prod, prod_top_left, mask=scaled_prod)

    # 4. Typography region analysis and styling
    # Left text box: x: 80 -> 640, y: 80 -> 548
    text_region_box = (80, 80, 640, 548)
    luminance = calculate_region_luminance(bg, text_region_box)
    styles = get_contrast_styling(luminance)

    draw = ImageDraw.Draw(canvas, "RGBA")

    # 5. Draw Brand Tag / Product Name
    brand_font = load_font(20, bold=True)
    brand_text = product_name.upper()
    brand_x = 90
    brand_y = 110
    draw.text((brand_x, brand_y), brand_text, fill=styles["brand_color"], font=brand_font)

    # Decorative accent pill or underline next to brand name
    brand_bbox = draw.textbbox((brand_x, brand_y), brand_text, font=brand_font)
    draw.line(
        [(brand_bbox[2] + 16, brand_y + 10), (brand_bbox[2] + 70, brand_y + 10)],
        fill=styles["brand_color"],
        width=3
    )

    # 6. Draw Auto-wrapped Tagline
    tagline_max_w = 540
    tagline_max_h = 240
    font, lines, line_height = fit_and_wrap_text(
        draw=draw,
        text=tagline,
        max_width=tagline_max_w,
        max_height=tagline_max_h,
        initial_font_size=52,
        min_font_size=28
    )

    tagline_start_y = brand_y + 44
    for i, line in enumerate(lines):
        line_y = tagline_start_y + (i * line_height)
        # Drop shadow / outline for crisp legibility
        draw.text(
            (brand_x + 2, line_y + 2),
            line,
            fill=styles["text_shadow"],
            font=font
        )
        draw.text(
            (brand_x, line_y),
            line,
            fill=styles["tagline_color"],
            font=font
        )

    # 7. Draw "Shop Now →" Call-To-Action Button
    button_y = tagline_start_y + (len(lines) * line_height) + 38
    # Ensure button does not overflow bottom
    if button_y > 480:
        button_y = 480

    btn_font = load_font(20, bold=True)
    btn_text = cta_text
    btn_text_bbox = draw.textbbox((0, 0), btn_text, font=btn_font)
    btn_text_w = btn_text_bbox[2] - btn_text_bbox[0]
    btn_text_h = btn_text_bbox[3] - btn_text_bbox[1]

    pad_h = 32
    pad_v = 16
    btn_w = btn_text_w + (pad_h * 2)
    btn_h = btn_text_h + (pad_v * 2)

    btn_rect = [brand_x, button_y, brand_x + btn_w, button_y + btn_h]
    
    # Button drop shadow
    draw.rounded_rectangle(
        [btn_rect[0] + 2, btn_rect[1] + 3, btn_rect[2] + 2, btn_rect[3] + 3],
        radius=10,
        fill=(0, 0, 0, 70)
    )
    # Button background
    draw.rounded_rectangle(
        btn_rect,
        radius=10,
        fill=styles["button_bg"]
    )
    # Centered CTA text
    text_x = brand_x + pad_h
    text_y = button_y + pad_v - 2
    draw.text((text_x, text_y), btn_text, fill=styles["button_text"], font=btn_font)

    return canvas

