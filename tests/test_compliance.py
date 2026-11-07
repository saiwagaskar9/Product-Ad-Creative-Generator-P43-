"""Tests for advertising compliance rules and automated regeneration."""

import pytest
from src.compliance import (
    check_rule_compliance,
    check_compliance,
    regenerate_compliant_tagline,
    PROTECTED_TRADEMARKS
)


def test_protected_trademarks_count():
    """Verify list includes 50+ famous protected trademarks."""
    assert len(PROTECTED_TRADEMARKS) >= 50


@pytest.mark.parametrize("brand", [
    "Hydro Flask", "Yeti", "Stanley", "Nike", "Adidas", "Lululemon",
    "Gatorade", "Apple", "Samsung", "Rolex", "Fitbit"
])
def test_trademark_detection(brand):
    """Verify unauthorized trademark use is flagged."""
    tagline = f"Better than {brand} today"
    res = check_rule_compliance(tagline)
    assert not res["is_compliant"]
    assert any("trademark" in f.lower() for f in res["flags"])


@pytest.mark.parametrize("comparative_phrase", [
    "Better than the rest",
    "Kills the competition completely",
    "Beats every other bottle"
])
def test_comparative_claims(comparative_phrase):
    """Verify comparative and aggressive claims are flagged."""
    res = check_rule_compliance(comparative_phrase)
    assert not res["is_compliant"]
    assert any("comparative" in f.lower() for f in res["flags"])


@pytest.mark.parametrize("endorsement", [
    "Doctor recommended hydration bottle",
    "FDA approved smart water bottle",
    "Clinically proven hydration system",
    "Dentist recommended daily sip"
])
def test_fake_endorsements(endorsement):
    """Verify fake authority and medical endorsements are flagged."""
    res = check_rule_compliance(endorsement)
    assert not res["is_compliant"]
    assert any("endorsement" in f.lower() for f in res["flags"])


@pytest.mark.parametrize("superlative", [
    "The world's best water bottle",
    "Rated #1 smart water bottle",
    "The finest hydration in history",
    "The ultimate hydration companion"
])
def test_superlatives(superlative):
    """Verify superlative claims are flagged."""
    res = check_rule_compliance(superlative)
    assert not res["is_compliant"]
    assert any("superlative" in f.lower() for f in res["flags"])


@pytest.mark.parametrize("false_guarantee", [
    "100% guaranteed hydration forever",
    "Zero risk and risk-free guarantee",
    "Miracle hydration guaranteed"
])
def test_false_guarantees(false_guarantee):
    """Verify absolute guarantees and risk-free claims are flagged."""
    res = check_rule_compliance(false_guarantee)
    assert not res["is_compliant"]
    assert any("guarantee" in f.lower() for f in res["flags"])


@pytest.mark.parametrize("medical_claim", [
    "Cures chronic dehydration and fatigue",
    "Treats high blood pressure easily",
    "Detoxifies your body every sip",
    "Prevents disease with every drop"
])
def test_medical_claims(medical_claim):
    """Verify medical and therapeutic claims are flagged."""
    res = check_rule_compliance(medical_claim)
    assert not res["is_compliant"]
    assert any("medical" in f.lower() for f in res["flags"])


def test_clean_taglines_pass():
    """Verify compliant, factual marketing claims receive 100 score."""
    clean_taglines = [
        "Precision Hydration for Unstoppable Energy.",
        "Track Every Sip. Feel the Difference.",
        "Smart Hydration Designed for Daily Flow.",
        "Fuel Your Next Workout with Confidence."
    ]
    for tag in clean_taglines:
        res = check_compliance(tag, mock_mode=True)
        assert res["is_compliant"], f"Flagged clean tagline: {tag} ({res['flags']})"
        assert res["score"] == 100


def test_regeneration_on_failure():
    """Verify automated regeneration replaces non-compliant tagline."""
    bad_tagline = "The #1 best bottle beats Hydro Flask"
    res = check_compliance(bad_tagline, mock_mode=True)
    assert not res["is_compliant"]

    new_tagline, new_report = regenerate_compliant_tagline(
        bad_tagline=bad_tagline,
        reasons=res["flags"],
        product_name="AquaPulse",
        description="reusable smart water bottle",
        mock_mode=True
    )
    assert new_report["is_compliant"]
    assert new_report["score"] == 100
    assert len(new_tagline.split()) <= 8

