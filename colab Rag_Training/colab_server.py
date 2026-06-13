"""
Colab Multi-Modal RAG Server

Flask API server designed to run inside Google Colab.
Hosts MedCPT (text embeddings), SigLIP (image embeddings),
and MedGemma (text generation) behind ngrok.

Usage in Colab:
    # Cell 1: Install deps
    !pip install -r requirements.txt

    # Cell 2: Mount Drive and start server
    from google.colab import drive
    drive.mount('/content/drive')
    from colab_server import app, start_server
    start_server()
"""

import io
import os
import sys
import time
import base64
import logging
import traceback
from typing import Dict, Any, List, Optional

import numpy as np
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("colab_server")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global model holders (lazy-loaded)
# ---------------------------------------------------------------------------
_models: Dict[str, Any] = {
    "medcpt_query": None,
    "medcpt_article": None,
    "medcpt_tokenizer": None,
    "siglip_model": None,
    "siglip_processor": None,
    "medgemma_model": None,
    "medgemma_tokenizer": None,
}

_device = None


def _get_device():
    """Detect best available device."""
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {_device}")
    return _device


# ---------------------------------------------------------------------------
# Model Loaders (lazy)
# ---------------------------------------------------------------------------

def _load_medcpt():
    """Load MedCPT query + article encoders from local Drive or HuggingFace."""
    if _models["medcpt_query"] is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModel
    from config import get_config

    cfg = get_config().models
    model_path = cfg.text_model_path  # local Drive copy first
    article_path = cfg.text_article_model_path

    logger.info(f"Loading MedCPT from {model_path}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    query_encoder = AutoModel.from_pretrained(model_path)
    article_encoder = AutoModel.from_pretrained(article_path)

    device = _get_device()
    query_encoder = query_encoder.to(device).eval()
    article_encoder = article_encoder.to(device).eval()

    _models["medcpt_tokenizer"] = tokenizer
    _models["medcpt_query"] = query_encoder
    _models["medcpt_article"] = article_encoder

    logger.info(f"MedCPT loaded in {time.time() - t0:.1f}s (dim=768)")


def _load_siglip():
    """Load SigLIP vision encoder from local Drive or HuggingFace."""
    if _models["siglip_model"] is not None:
        return
    from transformers import AutoModel, AutoProcessor
    from config import get_config

    cfg = get_config().models
    model_path = cfg.image_model_path

    logger.info(f"Loading SigLIP from {model_path}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    device = _get_device()
    model = model.to(device).eval()

    _models["siglip_processor"] = processor
    _models["siglip_model"] = model

    logger.info(f"SigLIP loaded in {time.time() - t0:.1f}s (dim=1152)")


def _ensure_medgemma_downloaded():
    """
    Download MedGemma 1.5 4B to Google Drive on first run.

    Persists the model at /content/drive/MyDrive/IDP/Medical/medgemma-1.5-4b-it
    so subsequent Colab sessions skip the ~8 GB download.
    """
    from config import get_config
    cfg = get_config().models
    local_dir = os.path.join(cfg.drive_root, cfg.generation_model_local)

    # Quick check: if config_json or model weights already exist, skip
    if os.path.isdir(local_dir) and any(
        f.endswith((".safetensors", ".bin", ".json"))
        for f in os.listdir(local_dir)
    ):
        logger.info(f"MedGemma already cached at {local_dir}")
        return local_dir

    logger.info(
        f"Downloading {cfg.generation_model_hf} to {local_dir} "
        f"(first-run only, ~8 GB)..."
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=cfg.generation_model_hf,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    logger.info(f"MedGemma download complete → {local_dir}")
    return local_dir


def _load_medgemma():
    """Load MedGemma for generation (multimodal image-text-to-text)."""
    if _models["medgemma_model"] is not None:
        return
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    # Ensure model is on Drive
    model_path = _ensure_medgemma_downloaded()

    logger.info(f"Loading MedGemma from {model_path}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if _get_device() == "cuda" else torch.float32,
        device_map="auto",
    )

    _models["medgemma_tokenizer"] = processor  # processor acts as tokenizer
    _models["medgemma_model"] = model

    logger.info(f"MedGemma loaded in {time.time() - t0:.1f}s")


def _unload_medcpt():
    """Unload MedCPT models from GPU and free VRAM."""
    import torch
    import gc

    if _models["medcpt_query"] is None:
        return

    logger.info("Unloading MedCPT models...")
    _models["medcpt_query"] = None
    _models["medcpt_article"] = None
    _models["medcpt_tokenizer"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("MedCPT unloaded, GPU memory freed.")


def _unload_siglip():
    """Unload SigLIP model from GPU and free VRAM."""
    import torch
    import gc

    if _models["siglip_model"] is None:
        return

    logger.info("Unloading SigLIP model...")
    _models["siglip_model"] = None
    _models["siglip_processor"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SigLIP unloaded, GPU memory freed.")


def _unload_medgemma():
    """Unload MedGemma model from GPU and free VRAM."""
    import torch
    import gc

    if _models["medgemma_model"] is None:
        return

    logger.info("Unloading MedGemma model...")
    _models["medgemma_model"] = None
    _models["medgemma_tokenizer"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("MedGemma unloaded, GPU memory freed.")


# ---------------------------------------------------------------------------
# Embedding Helpers
# ---------------------------------------------------------------------------

def _embed_texts_medcpt(texts: List[str], mode: str = "query") -> List[List[float]]:
    """Embed texts with MedCPT. mode='query' for queries, 'article' for docs."""
    import torch

    _load_medcpt()
    tokenizer = _models["medcpt_tokenizer"]
    encoder = _models["medcpt_query"] if mode == "query" else _models["medcpt_article"]
    device = _get_device()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = encoder(**encoded)
        # MedCPT uses [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]

    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().tolist()


def _embed_image_siglip(image_bytes: bytes) -> List[float]:
    """Embed a single image with SigLIP."""
    import torch
    from PIL import Image

    _load_siglip()
    processor = _models["siglip_processor"]
    model = _models["siglip_model"]
    device = _get_device()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.get_image_features(**inputs)
        else:
            outputs = model.get_image_features(**inputs)

        # outputs may be a BaseModelOutputWithPooling, not a raw tensor
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            image_features = outputs.last_hidden_state[:, 0, :]
        else:
            image_features = outputs  # some versions return tensor directly

        # L2 normalize → cast to float32 (ChromaDB/hnswlib rejects bfloat16)
        embedding = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        embedding = embedding.to(torch.float32)

    return embedding.squeeze(0).cpu().numpy().tolist()


def _embed_images_siglip_batch(
    image_bytes_list: List[bytes],
    batch_size: int = 12,
) -> List[List[float]]:
    """
    Batch-embed multiple images with SigLIP using the pre-loaded model.

    Uses AMP (mixed precision) for faster inference on CUDA GPUs.
    4-8x faster than calling _embed_image_siglip() in a loop.

    Handles the BaseModelOutputWithPooling object that some transformers
    versions return from ``get_image_features()``.

    Args:
        image_bytes_list: List of raw PNG/JPEG image bytes.
        batch_size: Images per forward pass (8-16 safe on T4).

    Returns:
        List of 1152-dim L2-normalized embeddings.
    """
    import torch
    from PIL import Image

    _load_siglip()
    processor = _models["siglip_processor"]
    model = _models["siglip_model"]
    device = _get_device()

    embeddings: List[List[float]] = []

    for i in range(0, len(image_bytes_list), batch_size):
        batch_bytes = image_bytes_list[i : i + batch_size]

        # Convert bytes → PIL images
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch_bytes]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.inference_mode():
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model.get_image_features(**inputs)
            else:
                outputs = model.get_image_features(**inputs)

            # outputs may be a BaseModelOutputWithPooling, not a raw tensor
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                image_features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                image_features = outputs.last_hidden_state[:, 0, :]
            else:
                image_features = outputs  # some versions return tensor directly

            emb = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            emb = emb.to(torch.float32)  # ChromaDB/hnswlib rejects bfloat16

        embeddings.extend(emb.cpu().numpy().tolist())

    return embeddings


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check reporting loaded models."""
    return jsonify({
        "status": "ok",
        "device": _get_device(),
        "models": {
            "medcpt": _models["medcpt_query"] is not None,
            "siglip": _models["siglip_model"] is not None,
            "medgemma": _models["medgemma_model"] is not None,
        },
        "dimensions": {
            "text": 768,
            "image": 1152,
        },
    })


@app.route("/embed_text", methods=["POST"])
def embed_text():
    """
    Embed text(s) with MedCPT.

    JSON body:
        text: str | list[str]  — text(s) to embed
        mode: str              — 'query' (default) or 'article'

    Returns:
        embedding: list[float] | list[list[float]]
        dimension: int
    """
    try:
        data = request.get_json(force=True)
        texts = data.get("text") or data.get("texts")
        mode = data.get("mode", "query")

        if texts is None:
            return jsonify({"error": "Missing 'text' field"}), 400

        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = _embed_texts_medcpt(texts, mode=mode)

        return jsonify({
            "embedding": embeddings[0] if single else embeddings,
            "dimension": 768,
            "count": len(embeddings),
        })
    except Exception as e:
        logger.error(f"/embed_text error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/embed_image", methods=["POST"])
def embed_image():
    """
    Embed image(s) with SigLIP.

    Accepts:
        - multipart/form-data with 'image' file(s)
        - JSON with base64-encoded 'image' field

    Returns:
        embedding: list[float] | list[list[float]]
        dimension: int
    """
    try:
        embeddings = []

        # Handle file upload
        if request.files:
            for key in request.files:
                file = request.files[key]
                img_bytes = file.read()
                emb = _embed_image_siglip(img_bytes)
                embeddings.append(emb)
        else:
            # Handle JSON with base64
            data = request.get_json(force=True)
            image_data = data.get("image")
            if image_data is None:
                return jsonify({"error": "Missing 'image' field or file upload"}), 400

            if isinstance(image_data, list):
                for b64 in image_data:
                    img_bytes = base64.b64decode(b64)
                    emb = _embed_image_siglip(img_bytes)
                    embeddings.append(emb)
            else:
                img_bytes = base64.b64decode(image_data)
                emb = _embed_image_siglip(img_bytes)
                embeddings.append(emb)

        single = len(embeddings) == 1
        return jsonify({
            "embedding": embeddings[0] if single else embeddings,
            "dimension": 1152,
            "count": len(embeddings),
        })
    except Exception as e:
        logger.error(f"/embed_image error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate text with MedGemma.

    JSON body:
        prompt: str              — user prompt
        context: str (optional)  — RAG context to prepend
        max_tokens: int          — max new tokens (default 1024)
        temperature: float       — sampling temperature (default 0.3)

    Returns:
        response: str
        tokens_used: int
    """
    try:
        import torch

        data = request.get_json(force=True)
        prompt = data.get("prompt")
        context = data.get("context", "")
        max_tokens = data.get("max_tokens", 2000)
        temperature = data.get("temperature", 0.3)

        if not prompt:
            return jsonify({"error": "Missing 'prompt' field"}), 400

        _load_medgemma()
        processor = _models["medgemma_tokenizer"]  # AutoProcessor
        model = _models["medgemma_model"]

        # Build chat messages
        if context:
            user_text = (
                f"Based on the following medical context, answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {prompt}"
            )
        else:
            user_text = prompt

        messages = [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
            )
            output_tokens = generation[0][input_len:]

        response_text = processor.decode(output_tokens, skip_special_tokens=True)

        return jsonify({
            "response": response_text.strip(),
            "tokens_used": len(output_tokens),
        })
    except Exception as e:
        logger.error(f"/generate error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Server Startup
# ---------------------------------------------------------------------------

def start_server(
    port: int = 5000,
    ngrok_token: Optional[str] = None,
    preload: bool = True,
    preload_medcpt: bool = True,
    preload_siglip: bool = True,
    preload_medgemma: bool = True,
):
    """
    Start the Flask server with ngrok tunnel.

    Args:
        port: Port to serve on (default 5000).
        ngrok_token: Ngrok auth token. Falls back to NGROK_AUTH_TOKEN env var.
        preload: If True AND per-model flags are True, load models upfront.
        preload_medcpt: Load MedCPT at startup (default True).
        preload_siglip: Load SigLIP at startup (default True).
        preload_medgemma: Load MedGemma at startup (default True).
    """
    # Optionally preload models (respects per-model flags)
    if preload:
        logger.info("Pre-loading models...")
        if preload_medcpt:
            _load_medcpt()
        if preload_siglip:
            _load_siglip()
        if preload_medgemma:
            _load_medgemma()
        logger.info("Requested models loaded. Starting server...")

    # Setup ngrok
    token = ngrok_token or os.getenv("NGROK_AUTH_TOKEN", "")
    if token:
        try:
            from pyngrok import ngrok as pyngrok
            pyngrok.set_auth_token(token)
            tunnel = pyngrok.connect(port, "http")
            public_url = tunnel.public_url
            logger.info(f"╔══════════════════════════════════════════╗")
            logger.info(f"║  ngrok tunnel active                     ║")
            logger.info(f"║  Public URL: {public_url:<28s}║")
            logger.info(f"╚══════════════════════════════════════════╝")
            print(f"\n🔗 COLAB_API_URL={public_url}\n")
        except Exception as e:
            logger.warning(f"ngrok setup failed: {e}. Serving locally only.")
    else:
        logger.warning("No NGROK_AUTH_TOKEN — serving locally only on port %d", port)

    # Run Flask
    app.run(host="0.0.0.0", port=port, debug=False)


# ---------------------------------------------------------------------------
# Direct execution (for Colab notebook cells)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start_server()
