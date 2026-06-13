# Colab RAG Training and Serving

This folder contains the Google Colab side of your multi-modal RAG pipeline.

It is designed to run on a Colab GPU with Google Drive storage and performs:
- Text ingestion from PDFs/JSON using MedCPT
- Image ingestion from PDFs using SigLIP
- ChromaDB persistence to Drive
- Optional MedGemma generation server exposed through ngrok

## What This Pipeline Does

`training_rag.py` runs a two-pass ingestion strategy to avoid GPU OOM:
- Pass 1: load MedCPT and ingest text chunks
- Pass 2: unload MedCPT, load SigLIP, ingest image embeddings
- Final: unload SigLIP, optionally start Flask API with MedGemma

This keeps only one heavy model in GPU memory at a time.

## Folder Files

- `training_rag.py`: End-to-end Colab training + server bootstrap.
- `data_ingestion.py`: PDF/JSON parsing, chunking, and Chroma writes.
- `colab_server.py`: Flask API (`/health`, `/embed_text`, `/embed_image`, `/generate`).
- `config.py`: Central config for model IDs/paths and runtime settings.
- `remote_embeddings.py`: LangChain-compatible remote embeddings client.
- `embedding_colab.py`: Utility service wrapper for Colab embedding endpoints.
- `rag_pipeline.py`: Retrieval and generation orchestration.
- `rag_integration_example.py`: Integration examples with Chroma/LangChain.
- `test_remote_embeddings.py`: End-to-end connectivity and dimension tests.
- `NGROK_SERVER.ipynb`: Notebook helper for launching API from Colab.
- `requirements.txt`: Python dependencies for this folder.

## Prerequisites

- Google Colab runtime with GPU enabled.
- Google Drive mounted in Colab.
- Medical source files under Drive:
	- `/content/drive/MyDrive/IDP/Medical/Books`
	- `/content/drive/MyDrive/IDP/Medical/Books/JsonData`
- Optional local Drive copies of models:
	- `MedCPT-Query-Encoder`
	- `siglip-so400m`

## Install Dependencies

In Colab:

```bash
pip install -r requirements.txt
```

or as notebook magic:

```python
!pip install -r requirements.txt
```

## Configure ngrok Token Safely

Do not hardcode tokens in source files.

In Colab before starting server:

```python
import os
os.environ["NGROK_AUTH_TOKEN"] = "<your-ngrok-token>"
```

`training_rag.py` and `config.py` now read token from `NGROK_AUTH_TOKEN` only.

## Run Full Training Pipeline

```bash
python training_rag.py
```

Expected high-level flow:
- Mount Drive
- Validate data/model paths
- Ingest text chunks into ChromaDB (local SSD first)
- Pre-extract images to local SSD
- Ingest image embeddings into ChromaDB
- Copy final ChromaDB to Drive
- Start Flask server (with ngrok if token exists)

## API Endpoints (from `colab_server.py`)

- `GET /health`: Model/server status and dimensions.
- `POST /embed_text`: MedCPT text embeddings.
- `POST /embed_image`: SigLIP image embeddings.
- `POST /generate`: MedGemma generation with optional context.

## Integration from Main App

Set env var in your app environment:

```bash
COLAB_API_URL=https://<your-ngrok-domain>
```

Then use `remote_embeddings.py` for LangChain-compatible embedding calls.

## Testing

After server is up and `COLAB_API_URL` is set:

```bash
python test_remote_embeddings.py
```

This validates:
- Health endpoint
- Text embedding dimensions
- Image embedding dimensions
- ChromaDB integration
- Optional LangChain integration

## Pre-Push Safety Checklist

- No hardcoded secrets/tokens in tracked files.
- `NGROK_AUTH_TOKEN` is supplied via environment only.
- `__pycache__` and other generated files are excluded from push.
- `COLAB_API_URL` is not hardcoded in source.
- Large temporary artifacts are not tracked.

## Notes on Paths

Defaults are tailored to:
- `/content/drive/MyDrive/IDP/Medical/...`

If your Drive structure differs, update `config.py` and/or paths in `training_rag.py`.

## Troubleshooting

- OOM during model loading:
	- Ensure only one model is loaded at a time.
	- Restart runtime and rerun from top.
- ChromaDB locking/performance issues:
	- Keep ingestion on local SSD (`/content/chromadb`) then copy to Drive.
- ngrok URL missing:
	- Verify `NGROK_AUTH_TOKEN` is set before `start_server()`.
- Empty retrieval results:
	- Confirm ingestion completed and persisted database was copied to Drive.

## Backup and Cleanup

Unwanted cache artifacts from this folder are moved to:
- `chatbot_service/backup/colab_backup`

This keeps the `colab` folder clean for GitHub push while preserving files if needed.
