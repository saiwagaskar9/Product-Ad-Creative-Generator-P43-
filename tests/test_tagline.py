"""Tests for tagline validation and generation."""

import pytest
from src.tagline import (
    validate_single_tagline,
    validate_taglines,
    generate_taglines,
    BANNED_WORDS,
    COMPETITOR_NAMES
)


def test_tagline_length_constraint():
    """Verify max 8 words limit."""
    valid_text = "Hydrate Smarter. Train Harder. Crush Goals."
    assert len(valid_text.split()) == 6
    is_valid, reason = validate_single_tagline(valid_text)
    assert is_valid
    assert reason is None

    # Exactly 8 words should pass
    eight_words = "One two three four five six seven eight"
    is_valid, _ = validate_single_tagline(eight_words)
    assert is_valid

    # 9 words must fail
    nine_words = "One two three four five six seven eight nine"
    is_valid, reason = validate_single_tagline(nine_words)
    assert not is_valid
    assert "exceeds 8 words" in reason


@pytest.mark.parametrize("banned_term", [
    "best", "#1", "number 1", "guaranteed", "100%", "risk-free", "miracle", "cures"
])
def test_tagline_banned_words(banned_term):
    """Verify rejection of superlatives, false claims, and medical cures."""
    tagline = f"Get the {banned_term} bottle today"
    is_valid, reason = validate_single_tagline(tagline)
    assert not is_valid
    assert "prohibited claim" in reason.lower() or banned_term in reason.lower()


@pytest.mark.parametrize("competitor", [
    "hydro flask", "stanley", "yeti", "camelbak", "nike", "gatorade", "apple"
])
def test_tagline_competitor_names(competitor):
    """Verify rejection of competitor trademarks in taglines."""
    tagline = f"Smarter than {competitor} everyday"
    is_valid, reason = validate_single_tagline(tagline)
    assert not is_valid
    assert "competitor reference" in reason.lower()


def test_taglines_batch_validation():
    """Verify batch validation checks exact count of 3 and uniqueness."""
    # Valid batch of 3 distinct taglines
    good_batch = [
        "Hydrate Smarter Every Day.",
        "Precision Hydration for Active Living.",
        "Track Sips. Stay Energized."
    ]
    is_valid, errors = validate_taglines(good_batch)
    assert is_valid
    assert len(errors) == 0

    # Batch with only 2 items
    is_valid, errors = validate_taglines(good_batch[:2])
    assert not is_valid
    assert any("expected exactly 3" in e.lower() for e in errors)

    # Batch with duplicate items
    dup_batch = [good_batch[0], good_batch[0], good_batch[1]]
    is_valid, errors = validate_taglines(dup_batch)
    assert not is_valid
    assert any("unique" in e.lower() for e in errors)


def test_mock_generation():
    """Verify mock generation produces 3 valid taglines <= 8 words."""
    res = generate_taglines(
        product_name="AquaPulse",
        description="reusable smart water bottle",
        audience_key="fitness enthusiasts",
        tone_key="energetic",
        mock_mode=True
    )
    assert res["is_mock"] is True
    assert len(res["taglines"]) == 3
    valid, errors = validate_taglines(res["taglines"])
    assert valid, f"Errors: {errors}"
    for tag in res["taglines"]:
        assert len(tag.split()) <= 8

