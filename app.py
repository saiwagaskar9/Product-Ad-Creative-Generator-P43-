"""Streamlit Web Application for Product Ad Creative Generator.

Generative AI Assignment P43.
Interactive interface to input product details, generate compliant taglines,
explore CLIP background candidate selection, preview 1200x628 banners, and download.
"""

import os
import io
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

from src.prompts import AUDIENCE_PROFILES, TONE_PRESETS
from src.pipeline import run_pipeline, PROJECT_ROOT, BANNERS_DIR, OUTPUTS_DIR
from src.background import CSV_LOG_FILE, detect_device

# Page Configuration
st.set_page_config(
    page_title="Product Ad Creative Generator | P43",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0ea5e9, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-compliant {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-flagged {
        background-color: #991b1b;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-winner {
        background-color: #1e3a8a;
        color: #60a5fa;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .card {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 16px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


def load_default_sample_image() -> Image.Image:
    """Load default sample product image."""
    sample_path = PROJECT_ROOT / "assets" / "sample_product.png"
    if sample_path.exists():
        return Image.open(sample_path)
    return Image.new("RGBA", (400, 700), (0, 180, 216, 255))


# Sidebar Controls
st.sidebar.markdown("## ⚙️ Configuration")

env_key = os.getenv("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=env_key,
    type="password",
    help="Optional if running in Mock Mode or already set in .env"
)

mock_mode = st.sidebar.checkbox(
    "Run in MOCK Mode",
    value=os.getenv("MOCK_LLM", "true").lower() in ("true", "1", "yes") or not bool(api_key_input),
    help="Uses deterministic compliant taglines and fast gradient backgrounds (no large models downloaded)."
)

backend_choice = st.sidebar.selectbox(
    "Image Backend",
    options=["mock", "diffusers"],
    index=0 if os.getenv("IMAGE_BACKEND", "mock").lower() == "mock" else 1,
    help="'mock' generates gradient candidates; 'diffusers' loads SD-Turbo + CLIP (requires local weights)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📦 Product Inputs")

product_name = st.sidebar.text_input(
    "Product Name",
    value="AquaPulse",
    help="The brand/product name to appear on the ad banner."
)

product_desc = st.sidebar.text_area(
    "Short Description",
    value="Reusable smart water bottle with hydration tracking, temperature sensors, and glowing LED reminders.",
    height=85
)

audience_options = list(AUDIENCE_PROFILES.keys())
selected_audience = st.sidebar.selectbox(
    "Target Audience",
    options=audience_options,
    index=audience_options.index("fitness enthusiasts") if "fitness enthusiasts" in audience_options else 0,
    format_func=lambda x: AUDIENCE_PROFILES[x]["name"]
)

tone_options = list(TONE_PRESETS.keys())
selected_tone = st.sidebar.selectbox(
    "Ad Tone",
    options=tone_options,
    index=tone_options.index("energetic") if "energetic" in tone_options else 0,
    format_func=lambda x: TONE_PRESETS[x]["name"]
)

uploaded_photo = st.sidebar.file_uploader(
    "Product Photo",
    type=["png", "jpg", "jpeg", "webp"],
    help="Upload a product photo. If empty, the default AquaPulse sample bottle is used."
)

if uploaded_photo is not None:
    product_img = Image.open(uploaded_photo)
    st.sidebar.image(product_img, caption="Uploaded Product Photo", use_container_width=True)
else:
    product_img = load_default_sample_image()
    st.sidebar.image(product_img, caption="Default Sample Product (AquaPulse)", use_container_width=True)

generate_btn = st.sidebar.button("🚀 Generate 3 Ad Banners", type="primary", use_container_width=True)

# Main UI Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<div class='main-title'>Product Ad Creative Generator</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Generative AI Assignment P43 &bull; AI Taglines &bull; CLIP Candidate Selection &bull; 1200x628 Banners</div>",
        unsafe_allow_html=True
    )
with col_h2:
    device = detect_device()
    st.info(f"**Compute:** `{device.upper()}` | **Mode:** `{'MOCK' if mock_mode else 'LIVE'}`")

# Session state initialization for results
if "pipeline_results" not in st.session_state:
    # Check if pre-generated banners exist on disk
    b1_path = BANNERS_DIR / "banner_1.png"
    b2_path = BANNERS_DIR / "banner_2.png"
    b3_path = BANNERS_DIR / "banner_3.png"
    if b1_path.exists() and b2_path.exists() and b3_path.exists():
        st.session_state["pipeline_results"] = {
            "banner_paths": [str(b1_path), str(b2_path), str(b3_path)],
            "banners": [Image.open(b1_path), Image.open(b2_path), Image.open(b3_path)],
            "taglines": [
                "Hydrate Smarter. Train Harder. Crush Goals.",
                "Precision Hydration for Unstoppable Energy.",
                "Fuel Every Rep with Intelligent Flow."
            ],
            "compliance_reports": [
                {"is_compliant": True, "score": 100, "flags": []},
                {"is_compliant": True, "score": 100, "flags": []},
                {"is_compliant": True, "score": 100, "flags": []}
            ],
            "metadata": {
                "product_name": "AquaPulse",
                "audience": "fitness enthusiasts",
                "tone": "energetic"
            }
        }

# Pipeline Execution
if generate_btn:
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(p: float, msg: str):
        progress_bar.progress(p)
        status_text.markdown(f"**Status:** {msg}")

    try:
        with st.spinner("Generating ad creatives..."):
            results = run_pipeline(
                product_name=product_name,
                description=product_desc,
                product_image=product_img,
                audience=selected_audience,
                tone=selected_tone,
                api_key=api_key_input,
                mock_mode=mock_mode,
                backend=backend_choice,
                progress_callback=update_progress
            )
            st.session_state["pipeline_results"] = results
            time.sleep(0.3)
            progress_bar.empty()
            status_text.success("🎉 All 3 ad banners generated successfully!")
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"❌ Error during generation: {str(e)}")

# Display Results View
if "pipeline_results" in st.session_state and st.session_state["pipeline_results"]:
    res = st.session_state["pipeline_results"]
    banners = res.get("banners", [])
    taglines = res.get("taglines", [])
    reports = res.get("compliance_reports", [])
    candidates_per_banner = res.get("clip_candidates_per_banner", [])

    st.markdown("### 🏆 Final Ad Banners (1200x628 PNG)")
    banner_tabs = st.tabs(["Side-by-Side Gallery", "Banner #1 Detail", "Banner #2 Detail", "Banner #3 Detail"])

    with banner_tabs[0]:
        cols = st.columns(3)
        for i in range(min(3, len(banners))):
            with cols[i]:
                st.markdown(f"#### Option {i+1}")
                st.image(banners[i], use_container_width=True)
                
                rep = reports[i] if i < len(reports) else {"is_compliant": True, "score": 100}
                if rep.get("is_compliant", True):
                    st.markdown(f"<span class='badge-compliant'>✅ COMPLIANT ({rep.get('score', 100)}/100)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-flagged'>⚠️ FLAGGED ({rep.get('score', 0)}/100)</span>", unsafe_allow_html=True)
                
                st.caption(f"**Tagline:** \"{taglines[i]}\"")
                
                buf = io.BytesIO()
                banners[i].save(buf, format="PNG")
                st.download_button(
                    label=f"⬇️ Download Banner {i+1}",
                    data=buf.getvalue(),
                    file_name=f"ad_banner_{i+1}_{product_name.lower()}.png",
                    mime="image/png",
                    key=f"dl_btn_{i+1}",
                    use_container_width=True
                )

    for tab_idx in range(1, 4):
        banner_idx = tab_idx - 1
        if banner_idx < len(banners):
            with banner_tabs[tab_idx]:
                col_img, col_info = st.columns([2, 1])
                with col_img:
                    st.image(banners[banner_idx], caption=f"Banner {tab_idx} (1200x628)", use_container_width=True)
                with col_info:
                    st.markdown("#### Banner Specs")
                    st.write(f"**Dimensions:** 1200 x 628 pixels (PNG)")
                    st.write(f"**Tagline:** \"{taglines[banner_idx]}\"")
                    st.write(f"**Word Count:** {len(taglines[banner_idx].split())} words (limit: 8)")
                    
                    rep = reports[banner_idx]
                    st.markdown("#### Legal & Compliance Status")
                    if rep.get("is_compliant", True):
                        st.success("✅ Fully Compliant with Advertising Standards")
                    else:
                        st.error("⚠️ Violations Identified")
                        for fl in rep.get("flags", []):
                            st.write(f"- {fl}")

    st.markdown("---")
    st.markdown("### 🔍 CLIP Candidate Scoring Explorer")
    st.write(
        "For each banner, 4 background candidates are generated and evaluated against the audience/tone text prompt. "
        "Images with potential text/watermarks are penalized, and the highest net score is selected."
    )

    if candidates_per_banner:
        for b_i, cands in enumerate(candidates_per_banner, 1):
            with st.expander(f"Banner #{b_i} Candidate Evaluation Breakdown", expanded=(b_i == 1)):
                c_cols = st.columns(len(cands))
                for c_i, cand in enumerate(cands):
                    with c_cols[c_i]:
                        st.image(cand["image"], caption=f"Candidate {c_i+1}", use_container_width=True)
                        if cand.get("selected", False):
                            st.markdown("<span class='badge-winner'>⭐ SELECTED WINNER</span>", unsafe_allow_html=True)
                        st.metric("Net CLIP Score", f"{cand.get('net_score', 0.0):.4f}")
                        st.caption(f"Relevance: `{cand.get('relevance_score', 0.0):.4f}`")
                        st.caption(f"Text Penalty: `{cand.get('text_penalty', 0.0):.4f}`")

    # CSV Log Download Table
    if CSV_LOG_FILE.exists():
        st.markdown("---")
        st.markdown("### 📊 CLIP Candidate Audit Log (`clip_candidate_scores.csv`)")
        try:
            df = pd.read_csv(CSV_LOG_FILE)
            st.dataframe(df.tail(12), use_container_width=True)
            with open(CSV_LOG_FILE, "rb") as f:
                st.download_button(
                    "📥 Download Complete CLIP Scores CSV",
                    data=f.read(),
                    file_name="clip_candidate_scores.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.warning(f"Could not load CSV log: {e}")

