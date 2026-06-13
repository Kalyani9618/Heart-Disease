"""
Data Ingestion Script for Colab-based Multi-Modal RAG

Reads PDFs and JSON files from Google Drive, chunks text, generates
embeddings via local Colab models, and stores vectors in ChromaDB
persisted to Drive.

Creates two collections:
  - medical_text_768   — text chunks embedded via MedCPT (768-dim)
  - medical_images_1152 — image descriptions embedded via SigLIP (1152-dim)

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    from data_ingestion import run_ingestion
    run_ingestion()
"""

import io
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
    min_size: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks using sentence-aware boundaries.

    Prefers splitting at paragraph breaks, then sentence endings, then
    whitespace, to avoid cutting mid-sentence.
    """
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_size else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if len(chunk) >= min_size:
                chunks.append(chunk)
            break

        # Find best split point (paragraph > sentence > word)
        segment = text[start:end]

        # Try paragraph break
        split_at = segment.rfind("\n\n")
        if split_at == -1 or split_at < chunk_size // 2:
            # Try sentence ending
            for delim in [". ", ".\n", "? ", "! "]:
                pos = segment.rfind(delim)
                if pos > chunk_size // 2:
                    split_at = pos + len(delim)
                    break

        if split_at == -1 or split_at < chunk_size // 2:
            # Fall back to whitespace
            split_at = segment.rfind(" ")
            if split_at == -1:
                split_at = chunk_size

        chunk = text[start : start + split_at].strip()
        if len(chunk) >= min_size:
            chunks.append(chunk)

        start = start + split_at - overlap
        if start < 0:
            start = 0

    return chunks


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_pdf_text_and_images(
    pdf_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract text and images from a PDF file.

    Returns:
        (text_pages, images) where:
        - text_pages: [{"page": int, "text": str}, ...]
        - images: [{"page": int, "image_bytes": bytes, "ext": str}, ...]
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    text_pages = []
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Text
        text = page.get_text("text")
        if text.strip():
            text_pages.append({"page": page_num + 1, "text": text})

        # Images
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_bytes = pix.tobytes("png")
                images.append({
                    "page": page_num + 1,
                    "image_bytes": img_bytes,
                    "ext": "png",
                    "idx": img_idx,
                })
            except Exception as e:
                logger.debug(f"Skipping image on page {page_num + 1}: {e}")

    doc.close()
    return text_pages, images


# ---------------------------------------------------------------------------
# JSON Extraction
# ---------------------------------------------------------------------------

def load_json_documents(json_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON or JSONL file."""
    docs = []
    path = Path(json_path)

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                docs = data
            else:
                docs = [data]

    return docs


# ---------------------------------------------------------------------------
# Embedding Generators (in-process, no ngrok needed)
# ---------------------------------------------------------------------------

def embed_texts_medcpt(texts: List[str]) -> List[List[float]]:
    """Generate 768-dim text embeddings using MedCPT article encoder."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Use the server loader if available, otherwise load directly
    try:
        from colab_server import _embed_texts_medcpt
        return _embed_texts_medcpt(texts, mode="article")
    except ImportError:
        pass

    from config import get_config
    cfg = get_config().models
    model_path = cfg.text_article_model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().tolist()


def embed_image_siglip(image_bytes: bytes) -> List[float]:
    """Generate 1152-dim image embedding using SigLIP (single image fallback)."""
    try:
        from colab_server import _embed_image_siglip
        return _embed_image_siglip(image_bytes)
    except ImportError:
        pass

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from config import get_config

    cfg = get_config().models
    model_path = cfg.image_model_path

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.get_image_features(**inputs)
        else:
            outputs = model.get_image_features(**inputs)

        # outputs may be BaseModelOutputWithPooling, not a raw tensor
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            image_features = outputs.last_hidden_state[:, 0, :]
        else:
            image_features = outputs

        embedding = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        embedding = embedding.to(torch.float32)  # ChromaDB/hnswlib rejects bfloat16

    return embedding.squeeze(0).cpu().numpy().tolist()


def embed_images_siglip_batch(
    image_bytes_list: List[bytes],
    batch_size: int = 16,
) -> List[List[float]]:
    """
    Batch version of SigLIP embedding — 4-8x faster on GPU.

    Uses AMP (mixed precision) and torch.compile for maximum throughput
    on free-tier Colab T4 GPUs.

    Args:
        image_bytes_list: List of raw PNG/JPEG bytes for each image.
        batch_size: Number of images to process per forward pass.

    Returns:
        List of 1152-dim normalized embeddings.
    """
    import torch

    # Try using the colab_server batch version if available (uses pre-loaded model)
    try:
        from colab_server import _embed_images_siglip_batch
        return _embed_images_siglip_batch(image_bytes_list, batch_size=batch_size)
    except (ImportError, AttributeError):
        pass

    # Fallback: load model locally (slower, only used outside colab_server context)
    from PIL import Image
    from transformers import AutoProcessor, AutoModel
    from config import get_config

    cfg = get_config().models
    model_path = cfg.image_model_path

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Attempt torch.compile for extra speedup (requires PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        logger.info("SigLIP model compiled with torch.compile ✅")
    except Exception as e:
        logger.debug(f"torch.compile not available, skipping: {e}")

    embeddings: List[List[float]] = []
    for i in range(0, len(image_bytes_list), batch_size):
        batch_bytes = image_bytes_list[i : i + batch_size]

        # Convert bytes → PIL images
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch_bytes]
        inputs = processor(images=images, return_tensors="pt").to(device)

        # Use AMP (Automatic Mixed Precision) for faster inference on CUDA
        with torch.inference_mode():
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model.get_image_features(**inputs)
            else:
                outputs = model.get_image_features(**inputs)

            # outputs may be BaseModelOutputWithPooling, not a raw tensor
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                image_features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                image_features = outputs.last_hidden_state[:, 0, :]
            else:
                image_features = outputs

            emb = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            emb = emb.to(torch.float32)  # ChromaDB/hnswlib rejects bfloat16

        embeddings.extend(emb.cpu().numpy().tolist())

    return embeddings


# ---------------------------------------------------------------------------
# ChromaDB Storage
# ---------------------------------------------------------------------------

def get_chroma_client(persist_dir: str):
    """Get ChromaDB client with persistence to Google Drive."""
    import chromadb
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def store_text_chunks(
    client,
    chunks: List[Dict[str, Any]],
    collection_name: str = "medical_text_768",
    batch_size: int = 32,
):
    """
    Embed and store text chunks in ChromaDB.

    Each chunk dict must have: 'text', 'metadata' (dict), 'id' (str).
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "dimension": 768},
    )

    total = len(chunks)
    stored = 0

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    batch_iter = range(0, total, batch_size)
    if tqdm:
        batch_iter = tqdm(
            batch_iter,
            desc="    └─ Embedding batches",
            total=(total + batch_size - 1) // batch_size,
            unit="batch",
        )

    for i in batch_iter:
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c.get("metadata", {}) for c in batch]

        embeddings = embed_texts_medcpt(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        stored += len(batch)

    logger.info(f"  Stored {stored}/{total} text chunks")
    return stored


def store_image_embeddings(
    client,
    images: List[Dict[str, Any]],
    collection_name: str = "medical_images_1152",
    batch_size: int = 16,
):
    """
    Embed and store images in ChromaDB using batched SigLIP inference.

    Each image dict must have: 'image_bytes', 'metadata' (dict), 'id' (str).

    Args:
        client: ChromaDB client.
        images: List of image dicts with image_bytes, metadata, id.
        collection_name: Name of the ChromaDB collection.
        batch_size: Number of images to embed per forward pass (8-16 safe on T4).

    Returns:
        Number of images successfully stored.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "dimension": 1152},
    )

    stored = 0
    total = len(images)

    try:
        from tqdm.auto import tqdm
        batch_iter = tqdm(
            range(0, total, batch_size),
            desc="    └─ Embedding image batches",
            total=(total + batch_size - 1) // batch_size,
            unit="batch",
        )
    except ImportError:
        batch_iter = range(0, total, batch_size)

    for i in batch_iter:
        batch = images[i : i + batch_size]
        img_bytes_list = [img["image_bytes"] for img in batch]

        try:
            # Use the new batched embedding function (4-8x faster)
            embeddings = embed_images_siglip_batch(img_bytes_list, batch_size=len(img_bytes_list))

            for img, emb in zip(batch, embeddings):
                description = img.get("metadata", {}).get(
                    "description",
                    f"Image from page {img.get('metadata', {}).get('page', '?')}",
                )
                collection.add(
                    documents=[description],
                    embeddings=[emb],
                    metadatas=[img.get("metadata", {})],
                    ids=[img["id"]],
                )
                stored += 1
        except Exception as e:
            logger.warning(f"Failed to embed image batch starting at index {i}: {e}")
            # Fallback: try one at a time for this batch
            for img in batch:
                try:
                    embedding = embed_image_siglip(img["image_bytes"])
                    description = img.get("metadata", {}).get(
                        "description",
                        f"Image from page {img.get('metadata', {}).get('page', '?')}",
                    )
                    collection.add(
                        documents=[description],
                        embeddings=[embedding],
                        metadatas=[img.get("metadata", {})],
                        ids=[img["id"]],
                    )
                    stored += 1
                except Exception as e2:
                    logger.warning(f"Failed to embed image {img.get('id', '?')}: {e2}")

    logger.info(f"  Stored {stored}/{total} image embeddings")
    return stored


# ---------------------------------------------------------------------------
# Main Ingestion Pipeline
# ---------------------------------------------------------------------------

def ingest_pdfs(
    pdf_dir: str,
    chroma_client,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> Dict[str, int]:
    """Ingest all PDFs from a directory."""
    stats = {"files": 0, "text_chunks": 0, "images": 0}
    pdf_dir = Path(pdf_dir)

    if not pdf_dir.exists():
        logger.warning(f"PDF directory not found: {pdf_dir}")
        return stats

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    try:
        from tqdm.auto import tqdm
        file_iter = tqdm(pdf_files, desc="📄 Processing PDFs", unit="file")
    except ImportError:
        file_iter = pdf_files

    for pdf_path in file_iter:
        if hasattr(file_iter, 'set_postfix'):
            file_iter.set_postfix(file=pdf_path.name[:30])
        logger.info(f"Processing: {pdf_path.name}")
        try:
            text_pages, images = extract_pdf_text_and_images(str(pdf_path))

            # Chunk text
            all_text = "\n\n".join(p["text"] for p in text_pages)
            chunks = chunk_text(all_text, chunk_size=chunk_size, overlap=chunk_overlap)

            # Prepare chunk records
            file_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
            chunk_records = [
                {
                    "text": chunk,
                    "id": f"{file_hash}_chunk_{i}",
                    "metadata": {
                        "source": pdf_path.name,
                        "chunk_index": i,
                        "type": "text",
                    },
                }
                for i, chunk in enumerate(chunks)
            ]

            stored = store_text_chunks(chroma_client, chunk_records)
            stats["text_chunks"] += stored

            # Prepare image records
            image_records = [
                {
                    "image_bytes": img["image_bytes"],
                    "id": f"{file_hash}_img_p{img['page']}_{img['idx']}",
                    "metadata": {
                        "source": pdf_path.name,
                        "page": img["page"],
                        "type": "image",
                    },
                }
                for img in images
            ]

            if image_records:
                img_stored = store_image_embeddings(chroma_client, image_records)
                stats["images"] += img_stored

            stats["files"] += 1
            print(f"  ✅ {pdf_path.name}: {len(chunks)} chunks, {len(images)} images")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    return stats


def ingest_json_files(
    json_dir: str,
    chroma_client,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> Dict[str, int]:
    """Ingest all JSON/JSONL files from a directory."""
    stats = {"files": 0, "text_chunks": 0}
    json_dir = Path(json_dir)

    if not json_dir.exists():
        logger.warning(f"JSON directory not found: {json_dir}")
        return stats

    json_files = list(json_dir.glob("**/*.json")) + list(json_dir.glob("**/*.jsonl"))
    logger.info(f"Found {len(json_files)} JSON files in {json_dir}")

    try:
        from tqdm.auto import tqdm
        file_iter = tqdm(json_files, desc="📋 Processing JSON", unit="file")
    except ImportError:
        file_iter = json_files

    for json_path in file_iter:
        if hasattr(file_iter, 'set_postfix'):
            file_iter.set_postfix(file=json_path.name[:30])
        logger.info(f"Processing: {json_path.name}")
        try:
            docs = load_json_documents(str(json_path))
            file_hash = hashlib.md5(json_path.name.encode()).hexdigest()[:8]

            chunk_records = []
            for doc_idx, doc in enumerate(docs):
                content = doc.get("content") or doc.get("text") or doc.get("contents") or ""
                title = doc.get("title", "")

                if not content.strip():
                    continue

                full_text = f"{title}\n\n{content}" if title else content
                chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=chunk_overlap)

                for i, chunk in enumerate(chunks):
                    chunk_records.append({
                        "text": chunk,
                        "id": f"{file_hash}_doc{doc_idx}_chunk{i}",
                        "metadata": {
                            "source": json_path.name,
                            "doc_index": doc_idx,
                            "title": title,
                            "chunk_index": i,
                            "type": "text",
                        },
                    })

            stored = store_text_chunks(chroma_client, chunk_records)
            stats["text_chunks"] += stored
            stats["files"] += 1
            print(f"  ✅ {json_path.name}: {len(docs)} docs → {len(chunk_records)} chunks")

        except Exception as e:
            logger.error(f"Failed to process {json_path.name}: {e}")

    return stats


# ---------------------------------------------------------------------------
# Two-Pass Functions: Text-Only and Image-Only
# ---------------------------------------------------------------------------

def ingest_pdfs_text_only(
    pdf_dir: str,
    chroma_client,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> Dict[str, int]:
    """
    PASS 1: Extract text from PDFs, chunk it, embed with MedCPT, store in ChromaDB.

    This function does NOT touch images at all — no SigLIP loaded.
    Returns stats dict with keys: files, text_chunks, total_pages.
    """
    stats = {"files": 0, "text_chunks": 0, "total_pages": 0}
    pdf_dir = Path(pdf_dir)

    if not pdf_dir.exists():
        logger.warning(f"PDF directory not found: {pdf_dir}")
        return stats

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    try:
        from tqdm.auto import tqdm
        file_iter = tqdm(pdf_files, desc="📄 Pass 1 — Text extraction", unit="file")
    except ImportError:
        file_iter = pdf_files

    for pdf_path in file_iter:
        if hasattr(file_iter, 'set_postfix'):
            file_iter.set_postfix(file=pdf_path.name[:30])
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            text_pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    text_pages.append({"page": page_num + 1, "text": text})
            doc.close()

            if not text_pages:
                print(f"  ⏭️  {pdf_path.name}: No text found, skipping")
                continue

            stats["total_pages"] += len(text_pages)

            # Chunk text
            all_text = "\n\n".join(p["text"] for p in text_pages)
            chunks = chunk_text(all_text, chunk_size=chunk_size, overlap=chunk_overlap)

            # Prepare chunk records
            file_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
            chunk_records = [
                {
                    "text": chunk,
                    "id": f"{file_hash}_chunk_{i}",
                    "metadata": {
                        "source": pdf_path.name,
                        "chunk_index": i,
                        "type": "text",
                    },
                }
                for i, chunk in enumerate(chunks)
            ]

            stored = store_text_chunks(chroma_client, chunk_records)
            stats["text_chunks"] += stored
            stats["files"] += 1
            print(f"  ✅ {pdf_path.name}: {len(text_pages)} pages → {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    return stats


def ingest_pdfs_images_only(
    pdf_dir: str,
    chroma_client,
    batch_size: int = 12,
    use_pre_extracted: bool = True,
    extracted_folder: str = "/content/extracted_images",
) -> Dict[str, int]:
    """
    PASS 2: Image-only ingestion — uses pre-extracted images when available.

    Fast path (default):
        Reads PNG files + manifest.json from ``extracted_folder`` on local SSD.
        No PDFs are re-opened from Google Drive.

    Fallback:
        If no pre-extracted folder is found, falls back to on-the-fly
        extraction from PDFs (slower, but still batched).

    Args:
        pdf_dir: Path to PDF directory (used only by the fallback path).
        chroma_client: ChromaDB client instance.
        batch_size: Images per SigLIP forward pass (8-16 safe on T4).
        use_pre_extracted: Whether to try the fast pre-extracted path first.
        extracted_folder: Path where ``pre_extract_all_images()`` saved PNGs.

    Returns:
        Stats dict with keys: files_with_images, images_embedded, images_skipped.
    """
    stats = {"files_with_images": 0, "images_embedded": 0, "images_skipped": 0}

    # ------------------------------------------------------------------
    # FAST PATH: load pre-extracted images from local SSD
    # ------------------------------------------------------------------
    manifest_path = os.path.join(extracted_folder, "manifest.json")

    if use_pre_extracted and os.path.isfile(manifest_path):
        logger.info(f"⚡ Fast path: loading pre-extracted images from {extracted_folder}")
        print(f"  ⚡ Using pre-extracted images from {extracted_folder} (fast path)")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        logger.info(f"  Manifest contains {len(manifest)} image entries")

        # Group by source PDF for per-file progress & stats
        from collections import defaultdict
        source_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in manifest:
            source_groups[entry["source"]].append(entry)

        try:
            from tqdm.auto import tqdm
            source_iter = tqdm(
                source_groups.items(),
                desc="🖼️  Pass 2 — Embedding pre-extracted images",
                unit="file",
            )
        except ImportError:
            source_iter = source_groups.items()

        for source_name, entries in source_iter:
            if hasattr(source_iter, 'set_postfix'):
                source_iter.set_postfix(file=source_name[:30])

            image_records: List[Dict[str, Any]] = []
            for entry in entries:
                img_path = os.path.join(extracted_folder, entry["filename"])
                if not os.path.isfile(img_path):
                    stats["images_skipped"] += 1
                    continue
                try:
                    with open(img_path, "rb") as img_f:
                        img_bytes = img_f.read()
                    image_records.append({
                        "image_bytes": img_bytes,
                        "id": entry["id"],
                        "metadata": {
                            "source": entry["source"],
                            "page": entry["page"],
                            "type": "image",
                        },
                    })
                except Exception:
                    stats["images_skipped"] += 1

            if not image_records:
                continue

            img_stored = store_image_embeddings(
                chroma_client, image_records, batch_size=batch_size,
            )
            stats["images_embedded"] += img_stored
            stats["files_with_images"] += 1
            print(f"  ✅ {source_name}: {img_stored} images embedded")

        return stats

    # ------------------------------------------------------------------
    # FALLBACK: on-the-fly extraction from PDFs (slower)
    # ------------------------------------------------------------------
    if use_pre_extracted:
        logger.warning("⚠️  No pre-extracted images found — falling back to PDF extraction")
        print("  ⚠️  No pre-extracted images found — falling back to slow PDF extraction")

    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        logger.warning(f"PDF directory not found: {pdf_dir}")
        return stats

    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files for image extraction")

    try:
        from tqdm.auto import tqdm
        file_iter = tqdm(pdf_files, desc="🖼️  Pass 2 — Image extraction (slow)", unit="file")
    except ImportError:
        file_iter = pdf_files

    for pdf_path in file_iter:
        if hasattr(file_iter, 'set_postfix'):
            file_iter.set_postfix(file=pdf_path.name[:30])
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            images = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n > 4:  # CMYK → RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_bytes = pix.tobytes("png")
                        images.append({
                            "page": page_num + 1,
                            "image_bytes": img_bytes,
                            "idx": img_idx,
                        })
                    except Exception:
                        stats["images_skipped"] += 1
            doc.close()

            if not images:
                continue

            file_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
            image_records = [
                {
                    "image_bytes": img["image_bytes"],
                    "id": f"{file_hash}_img_p{img['page']}_{img['idx']}",
                    "metadata": {
                        "source": pdf_path.name,
                        "page": img["page"],
                        "type": "image",
                    },
                }
                for img in images
            ]

            img_stored = store_image_embeddings(
                chroma_client, image_records, batch_size=batch_size,
            )
            stats["images_embedded"] += img_stored
            stats["files_with_images"] += 1
            print(f"  ✅ {pdf_path.name}: {img_stored} images embedded")

        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path.name}: {e}")

    return stats


def pre_extract_all_images(
    books_dir: str,
    output_folder: str = "/content/extracted_images",
) -> str:
    """
    Pre-extract every image from every PDF and save as PNG files on local SSD.

    Also writes a ``manifest.json`` mapping each filename back to its source
    PDF, page number, and image index so that Phase 3 can restore correct
    metadata without re-opening any PDFs from Google Drive.

    This is a one-time I/O cost.  Run it once (e.g. between Phase 2 and
    Phase 3) and subsequent image-embedding runs read from local disk.

    Args:
        books_dir: Path to directory containing PDF files (may be on Drive).
        output_folder: Path on *local* SSD to save extracted PNG images.

    Returns:
        Path to the output folder (same as ``output_folder``).
    """
    import fitz  # PyMuPDF

    os.makedirs(output_folder, exist_ok=True)
    pdf_files = list(Path(books_dir).glob("**/*.pdf"))
    logger.info(f"Pre-extracting images from {len(pdf_files)} PDFs → {output_folder}")

    total_images = 0
    skipped = 0
    manifest: List[Dict[str, Any]] = []  # metadata for every saved image

    try:
        from tqdm.auto import tqdm
        file_iter = tqdm(pdf_files, desc="📦 Pre-extracting images to disk", unit="file")
    except ImportError:
        file_iter = pdf_files

    for pdf_path in file_iter:
        if hasattr(file_iter, 'set_postfix'):
            file_iter.set_postfix(file=pdf_path.name[:30])
        try:
            doc = fitz.open(str(pdf_path))
            file_hash = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]

            for page_num, page in enumerate(doc):
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n > 4:  # CMYK → RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        img_filename = f"{file_hash}_p{page_num + 1}_{img_idx}.png"
                        save_path = os.path.join(output_folder, img_filename)
                        pix.save(save_path)

                        manifest.append({
                            "filename": img_filename,
                            "id": f"{file_hash}_img_p{page_num + 1}_{img_idx}",
                            "source": pdf_path.name,
                            "page": page_num + 1,
                            "img_idx": img_idx,
                            "file_hash": file_hash,
                        })
                        total_images += 1
                    except Exception:
                        skipped += 1
            doc.close()
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path.name}: {e}")

    # Write manifest so the fast-path knows source/page for each image
    manifest_path = os.path.join(output_folder, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    logger.info(f"  Manifest written → {manifest_path} ({len(manifest)} entries)")

    logger.info(f"✅ Extracted {total_images} images to {output_folder} (skipped {skipped})")
    print(f"✅ Extracted {total_images} images to {output_folder} (skipped {skipped})")
    return output_folder


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run_ingestion(
    books_dir: str = "/content/drive/MyDrive/IDP/Medical/Books",
    json_dir: str = "/content/drive/MyDrive/IDP/Medical/Books/JsonData",
    chromadb_dir: str = "/content/drive/MyDrive/IDP/Medical/Chromadb",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline.

    Args:
        books_dir: Path to PDF medical textbooks on Google Drive.
        json_dir: Path to JSON datasets on Google Drive.
        chromadb_dir: Path for persisted ChromaDB on Google Drive.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Combined statistics dict.
    """
    logger.info("=" * 60)
    logger.info("Starting Multi-Modal Data Ingestion")
    logger.info("=" * 60)
    t0 = time.time()

    client = get_chroma_client(chromadb_dir)

    # Ingest PDFs
    logger.info(f"\n📄 Ingesting PDFs from {books_dir}")
    pdf_stats = ingest_pdfs(books_dir, client, chunk_size, chunk_overlap)

    # Ingest JSON
    logger.info(f"\n📋 Ingesting JSON from {json_dir}")
    json_stats = ingest_json_files(json_dir, client, chunk_size, chunk_overlap)

    elapsed = time.time() - t0

    summary = {
        "pdf": pdf_stats,
        "json": json_stats,
        "total_text_chunks": pdf_stats["text_chunks"] + json_stats["text_chunks"],
        "total_images": pdf_stats.get("images", 0),
        "total_files": pdf_stats["files"] + json_stats["files"],
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete")
    logger.info(f"  Files processed: {summary['total_files']}")
    logger.info(f"  Text chunks:     {summary['total_text_chunks']}")
    logger.info(f"  Images:          {summary['total_images']}")
    logger.info(f"  Time:            {summary['elapsed_seconds']}s")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    run_ingestion()
