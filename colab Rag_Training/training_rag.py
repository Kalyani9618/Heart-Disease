# -*- coding: utf-8 -*-
"""Training_Rag.ipynb — Two-Pass Sequential Pipeline with Progress

Run this file in Google Colab to:
  Phase 1: Setup & validate paths
  Phase 2: Load MedCPT → Extract & embed TEXT from PDFs + JSON → Unload MedCPT
  Phase 3: Load SigLIP → Extract & embed IMAGES from PDFs → Unload SigLIP
  Phase 4: Load MedGemma → Start API server via ngrok

Models are loaded ONE AT A TIME to avoid GPU OOM on free Colab (15 GB).
PDFs are read TWICE: once for text, once for images — never both models on GPU.
"""

# ============================================================================
# Cell 1: Install Dependencies
# ============================================================================
# !pip install flask pyngrok torch transformers sentence-transformers accelerate chromadb PyMuPDF Pillow unstructured requests numpy tqdm

# ============================================================================
# Cell 2: Full Training Pipeline with Progress
# ============================================================================

import os
import sys
import time
import gc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_banner(phase_num, title, emoji="🔷"):
    """Print a decorative phase banner visible in Colab output."""
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {emoji}  PHASE {phase_num}: {title}")
    print(f"{'═' * width}\n")


def print_gpu_status():
    """Show current GPU memory usage — works on Colab with CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            free = total - allocated
            bar_len = 30
            used_ratio = allocated / total if total > 0 else 0
            filled = int(bar_len * used_ratio)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  🖥️  GPU Memory: [{bar}] {allocated:.1f}GB / {total:.1f}GB  (Free: {free:.1f}GB)")
        else:
            print("  ⚠️  No GPU detected — running on CPU")
    except Exception:
        print("  ⚠️  Could not read GPU status")


def print_step(msg):
    """Print a timestamped step message."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def clear_gpu():
    """Force-clear GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================================
# PHASE 1: Setup & Configuration
# ============================================================================

print_banner(1, "SETUP & CONFIGURATION", "⚙️")
pipeline_start = time.time()

# Mount Google Drive (only in Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print_step("Google Drive mounted ✅")
except ImportError:
    print_step("Not running in Colab — skipping drive mount")

# Add code dir to path
sys.path.append('/content/drive/MyDrive/IDP/Medical/Code')

# Import key modules
from config import get_config
from data_ingestion import (
    get_chroma_client,
    ingest_pdfs_text_only,
    ingest_pdfs_images_only,
    ingest_json_files,
    pre_extract_all_images,
)
from colab_server import (
    _load_medcpt, _unload_medcpt,
    _load_siglip, _unload_siglip,
    _load_medgemma,
    start_server,
)

config = get_config()
config.validate()
print_step("Config loaded and validated ✅")

# Paths
books_dir = '/content/drive/MyDrive/IDP/Medical/Books'
json_dir = '/content/drive/MyDrive/IDP/Medical/Books/JsonData'

# ChromaDB: use LOCAL SSD during ingestion (Google Drive FUSE breaks SQLite WAL)
# The finished database is copied to Drive after all ingestion is complete.
chromadb_local = '/content/chromadb'          # Fast local SSD
chromadb_drive = '/content/drive/MyDrive/IDP/Medical/Chromadb'  # Persistent backup

# Clean any stale local DB from a previous crashed run
import shutil
if os.path.exists(chromadb_local):
    shutil.rmtree(chromadb_local, ignore_errors=True)
    print_step("Cleared stale local ChromaDB ♻️")

# Verify paths exist
for label, path in [("Books", books_dir), ("JSON", json_dir)]:
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print_step(f"{label} directory: {path} ({count} items) ✅")
    else:
        print_step(f"⚠️  {label} directory NOT found: {path}")

# Check model directories
models_root = '/content/drive/MyDrive/IDP/Medical'
for model_name in ['MedCPT-Query-Encoder', 'siglip-so400m']:
    model_path = os.path.join(models_root, model_name)
    status = "✅ Found" if os.path.isdir(model_path) else "❌ Missing"
    print_step(f"Model [{model_name}]: {status}")

print_gpu_status()

# ============================================================================
# PHASE 2: Text-Only Pass (MedCPT)
# ============================================================================

print_banner(2, "TEXT INGESTION — MedCPT Only", "📄")
phase2_start = time.time()

# Step 2a: Load MedCPT (the ONLY model on GPU during this phase)
print_step("Loading MedCPT text encoder...")
_load_medcpt()
print_step("MedCPT loaded ✅")
print_gpu_status()

# Step 2b: Text-only pass on PDFs (no images extracted, no SigLIP needed)
print_step("Starting PDF text-only ingestion (Pass 1 of 2)...")
print_step(f"ChromaDB: using LOCAL path {chromadb_local} (faster, no Drive locking issues)")
chroma_client = get_chroma_client(chromadb_local)

pdf_text_stats = ingest_pdfs_text_only(
    books_dir, chroma_client,
    chunk_size=1500, chunk_overlap=200,
)

# Step 2c: JSON ingestion (text only, no images)
print_step("Starting JSON ingestion...")
json_stats = ingest_json_files(
    json_dir, chroma_client,
    chunk_size=1500, chunk_overlap=200,
)

phase2_time = time.time() - phase2_start
total_text_chunks = pdf_text_stats["text_chunks"] + json_stats["text_chunks"]
total_files = pdf_text_stats["files"] + json_stats["files"]

print(f"\n  📊 Phase 2 Summary:")
print(f"     PDF files processed:  {pdf_text_stats['files']}")
print(f"     JSON files processed: {json_stats['files']}")
print(f"     Total text chunks:    {total_text_chunks}")
print(f"     Total pages:          {pdf_text_stats.get('total_pages', 'N/A')}")
print(f"     Time:                 {phase2_time:.1f}s")

# Step 2d: Unload MedCPT to free GPU completely
print_step("Unloading MedCPT to free GPU memory...")
_unload_medcpt()
clear_gpu()
print_step("MedCPT unloaded ✅")
print_gpu_status()

# ============================================================================
# PHASE 2.5: Pre-extract images to local SSD (one-time I/O cost)
# ============================================================================
# This avoids re-opening all 50+ PDFs from Google Drive FUSE in Phase 3.
# Images are saved as PNGs on the fast Colab local SSD with a manifest.json
# mapping each image back to its source PDF, page number, and index.

extracted_images_dir = '/content/extracted_images'

print_step("Pre-extracting images from PDFs → local SSD (one-time cost)...")
pre_extract_all_images(books_dir, output_folder=extracted_images_dir)
print_step("Pre-extraction complete ✅")

# ============================================================================
# PHASE 3: Image-Only Pass (SigLIP)
# ============================================================================

print_banner(3, "IMAGE INGESTION — SigLIP Only", "🖼️")
phase3_start = time.time()

# Re-obtain chroma_client (safe if running Phase 3 in a separate Colab cell)
chroma_client = get_chroma_client(chromadb_local)

# Step 3a: Load SigLIP (the ONLY model on GPU during this phase)
print_step("Loading SigLIP image encoder...")
_load_siglip()
print_step("SigLIP loaded ✅")
print_gpu_status()

# Step 3b: Image-only pass — reads pre-extracted PNGs from local SSD (fast!)
print_step("Starting image embedding from pre-extracted PNGs (fast path)...")
print_step("Using batched SigLIP inference (batch_size=12) + AMP for 4-8x speedup ⚡")
image_stats = ingest_pdfs_images_only(
    books_dir,
    chroma_client,
    batch_size=12,
    use_pre_extracted=True,
    extracted_folder=extracted_images_dir,
)

phase3_time = time.time() - phase3_start
total_images = image_stats["images_embedded"]

print(f"\n  📊 Phase 3 Summary:")
print(f"     PDFs with images:     {image_stats['files_with_images']}")
print(f"     Images embedded:      {total_images}")
print(f"     Images skipped:       {image_stats['images_skipped']}")
print(f"     Time:                 {phase3_time:.1f}s")

# Step 3c: Unload SigLIP to free GPU completely
print_step("Unloading SigLIP to free GPU memory...")
_unload_siglip()
clear_gpu()
print_step("SigLIP unloaded ✅")
print_gpu_status()

# Step 3d: Copy ChromaDB from local SSD → Google Drive for persistence
print_step(f"Copying ChromaDB to Drive: {chromadb_drive} ...")
if os.path.exists(chromadb_drive):
    shutil.rmtree(chromadb_drive, ignore_errors=True)
shutil.copytree(chromadb_local, chromadb_drive)
print_step("ChromaDB copied to Drive ✅")
# Clean up local copy to free Colab disk
shutil.rmtree(chromadb_local, ignore_errors=True)
print_step("Local ChromaDB cleaned up ♻️")

# Clean up pre-extracted images to free Colab disk space
if os.path.exists(extracted_images_dir):
    shutil.rmtree(extracted_images_dir, ignore_errors=True)
    print_step("Pre-extracted images cleaned up ♻️")

# ============================================================================
# PHASE 4: Load MedGemma & Start Server
# ============================================================================

print_banner(4, "LOAD MEDGEMMA & START API SERVER", "🚀")

print_step("Loading MedGemma for text generation (this may take a few minutes)...")
print_step("If first run, model will be downloaded to Drive (~8 GB)...")

# Set your ngrok token via environment variable before running.
# Example in Colab:
#   import os
#   os.environ["NGROK_AUTH_TOKEN"] = "<your-token>"
ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "")
if not ngrok_token:
    print_step("NGROK_AUTH_TOKEN not set; server will run without public ngrok tunnel")

# ============================================================================
# PHASE 5: Final Summary (printed before server blocks)
# ============================================================================

total_time = time.time() - pipeline_start

print_banner(5, "PIPELINE SUMMARY", "📊")
print(f"  ┌─────────────────────────────────────────────────────┐")
print(f"  │  Two-Pass Ingestion Complete!                        │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │  PDF files (text):      {pdf_text_stats['files']:>6}                  │")
print(f"  │  JSON files:            {json_stats['files']:>6}                  │")
print(f"  │  Total text chunks:     {total_text_chunks:>6}                  │")
print(f"  │  Total images embedded: {total_images:>6}                  │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │  Phase 2 (Text/MedCPT):  {phase2_time:>7.1f}s                 │")
print(f"  │  Phase 3 (Imgs/SigLIP):  {phase3_time:>7.1f}s                 │")
print(f"  │  Total pipeline time:    {total_time:>7.1f}s                 │")
print(f"  └─────────────────────────────────────────────────────┘")
print()
print_gpu_status()
print()

# Now start the server — loads ONLY MedGemma (MedCPT & SigLIP stay unloaded)
print_step("Starting API server with MedGemma only...")
print_step("Server will be accessible via ngrok tunnel below ⬇️")
print()

start_server(
    port=5000,
    ngrok_token=ngrok_token,
    preload=True,
    preload_medcpt=False,    # Already done & unloaded
    preload_siglip=False,    # Already done & unloaded
    preload_medgemma=True,   # Load for generation
)