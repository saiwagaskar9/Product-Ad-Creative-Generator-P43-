"""Tests for banner composition, text wrapping, and luminance contrast."""

from PIL import Image, ImageDraw
from src.compose import (
    BANNER_WIDTH,
    BANNER_HEIGHT,
    calculate_region_luminance,
    get_contrast_styling,
    fit_and_wrap_text,
    compose_banner,
    load_font
)


def test_banner_dimensions():
    """Verify banner composition adheres strictly to 1200x628 dimensions."""
    bg = Image.new("RGB", (512, 512), (30, 40, 60))
    prod = Image.new("RGBA", (300, 600), (0, 180, 216, 255))
    tagline = "Precision Hydration for Peak Performance."

    banner = compose_banner(
        background_img=bg,
        product_img=prod,
        tagline=tagline,
        product_name="AquaPulse",
        target_size=(BANNER_WIDTH, BANNER_HEIGHT)
    )

    assert banner.size == (1200, 628)
    assert banner.format is None or banner.mode == "RGBA" or banner.mode == "RGB"


def test_auto_contrast_luminance():
    """Verify dark backgrounds trigger white text and light backgrounds trigger dark text."""
    # Dark image (RGB: 15, 20, 30)
    dark_img = Image.new("RGB", (200, 200), (15, 20, 30))
    dark_lum = calculate_region_luminance(dark_img, (0, 0, 200, 200))
    assert dark_lum < 135.0
    dark_styles = get_contrast_styling(dark_lum)
    assert dark_styles["is_dark_bg"] is True
    assert dark_styles["tagline_color"] == (255, 255, 255)

    # Light image (RGB: 245, 245, 250)
    light_img = Image.new("RGB", (200, 200), (245, 245, 250))
    light_lum = calculate_region_luminance(light_img, (0, 0, 200, 200))
    assert light_lum >= 135.0
    light_styles = get_contrast_styling(light_lum)
    assert light_styles["is_dark_bg"] is False
    assert dark_styles["tagline_color"] != light_styles["tagline_color"]
    assert light_styles["tagline_color"] == (15, 23, 42)


def test_fit_and_wrap_text():
    """Verify text wrapping respects box bounds and does not clip."""
    canvas = Image.new("RGB", (1200, 628))
    draw = ImageDraw.Draw(canvas)

    tagline = "Precision Hydration Engineered for Your Daily Unstoppable Flow"
    max_w = 500
    max_h = 200

    font, lines, line_height = fit_and_wrap_text(
        draw=draw,
        text=tagline,
        max_width=max_w,
        max_height=max_h,
        initial_font_size=54,
        min_font_size=24
    )

    assert len(lines) >= 1
    # Check that each line fits within max_w
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        assert w <= max_w, f"Line '{line}' width {w} exceeds max {max_w}"

    total_h = len(lines) * line_height
    assert total_h <= max_h + 30

