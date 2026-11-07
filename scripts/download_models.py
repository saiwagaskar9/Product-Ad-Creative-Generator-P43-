"""Model preloading and download utility for Product Ad Creative Generator.

Downloads and caches:
1. Stability AI SD-Turbo (stabilityai/sd-turbo) for fast background generation
2. OpenAI CLIP ViT-B/32 (openai/clip-vit-base-patch32) for multimodal candidate scoring
3. rembg background removal model (u2net)

Run this script once before enabling IMAGE_BACKEND=diffusers.
Usage:
    python scripts/download_models.py
"""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelDownloader")


def check_environment():
    """Check Python environment and available compute devices."""
    logger.info(f"Python version: {sys.version}")
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Silicon GPU (MPS) detected and active.")
        else:
            device = "cpu"
            logger.info("Running on CPU. Inferences will execute in CPU-safe mode.")
        return device
    except ImportError:
        logger.error("PyTorch is not installed. Please install torch: pip install torch torchvision")
        sys.exit(1)


def download_sd_turbo(device: str):
    """Download and cache stabilityai/sd-turbo weights."""
    logger.info("--> [1/3] Downloading Stable Diffusion Turbo (stabilityai/sd-turbo)...")
    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch_dtype,
            variant="fp16" if device == "cuda" else None
        )
        logger.info("✅ SD-Turbo downloaded and cached successfully.")
        return pipe
    except Exception as e:
        logger.error(f"❌ Failed to download SD-Turbo: {e}")
        logger.info("Tip: install diffusers with: pip install diffusers accelerate")
        return None


def download_clip(device: str):
    """Download and cache openai/clip-vit-base-patch32 weights."""
    logger.info("--> [2/3] Downloading OpenAI CLIP (openai/clip-vit-base-patch32)...")
    try:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("✅ CLIP ViT-B/32 downloaded and cached successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"❌ Failed to download CLIP model: {e}")
        logger.info("Tip: install transformers with: pip install transformers")
        return None, None


def download_rembg():
    """Pre-cache rembg u2net model."""
    logger.info("--> [3/3] Testing rembg background removal model...")
    try:
        import rembg
        from PIL import Image
        dummy = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        _ = rembg.remove(dummy)
        logger.info("✅ rembg (u2net) verified and cached successfully.")
    except Exception as e:
        logger.warning(f"⚠️ rembg cache skipped or not installed ({e}). (compose.py will use alpha fallback)")


def main():
    logger.info("=== Starting Model Download & Verification ===")
    start_time = time.time()
    
    device = check_environment()
    download_sd_turbo(device)
    download_clip(device)
    download_rembg()
    
    elapsed = time.time() - start_time
    logger.info(f"=== Download Process Completed in {elapsed:.1f}s ===")
    logger.info("You can now set IMAGE_BACKEND=diffusers and MOCK_LLM=false in .env!")


if __name__ == "__main__":
    main()

