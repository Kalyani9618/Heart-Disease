# Setup Guide

## HeartGuard AI — Installation & Configuration

> **Version:** 2.1.0  
> **Last Updated:** February 2026  
> **Author:** HeartGuard AI Team

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start (Docker)](#quick-start-docker)
3. [Manual Setup](#manual-setup)
4. [Environment Variables](#environment-variables)
5. [Database Setup](#database-setup)
6. [AI Model Setup](#ai-model-setup)
7. [Frontend Setup](#frontend-setup)
8. [Colab Integration](#colab-integration)
9. [Verification](#verification)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Hardware

```
┌──────────────────────────────────────────────────┐
│  MINIMUM REQUIREMENTS                            │
│                                                  │
│  CPU:     4 cores                                │
│  RAM:     16 GB (32 GB recommended)              │
│  Disk:    250 GB+ (SSD recommended)              │
│  GPU:     Optional (speeds up MedGemma)          │
│  Network: Internet for OpenFDA + web search      │
└──────────────────────────────────────────────────┘
```

### Software Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Backend runtime |
| Node.js | 18+ | Frontend build |
| Docker | 24+ | Container deployment |
| Docker Compose | 2.20+ | Multi-service orchestration |
| PostgreSQL | 15+ | Primary database |
| Redis | 7+ | Cache & job queue |
| Git | 2.30+ | Version control |

---

## Quick Start (Docker)

**Fastest way to run everything:**

```bash
# 1. Clone the repository
git clone <repo-url> Heart
cd Heart

# 2. Copy environment template
cp chatbot_service/.env.example chatbot_service/.env

# 3. Edit .env with your settings (see Environment Variables section)

# 4. Start all services
cd chatbot_service/Docker
docker-compose up -d
```

### What Docker Starts

```
┌──────────────────────────────────────────────────────────────┐
│                   DOCKER COMPOSE STACK                        │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  medgemma (8 GB RAM)                                  │   │
│  │  LLM server on port 8090                              │   │
│  │  Image: llama.cpp with MedGemma-4B-IT model           │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  api (4 GB RAM)                                       │   │
│  │  FastAPI backend on port 8000                         │   │
│  │  Depends on: db, redis, medgemma                      │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐                                   │
│  │  db (1 GB)  │  │ redis        │                                   │
│  │  PostgreSQL │  │ (512 MB)     │                                   │
│  │  Port 5432  │  │ Port 6379    │                                   │
│  └─────────────┘  │              │                                   │
│                   └──────────────┘                                   │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  chroma (1 GB)                                        │   │
│  │  Vector database on port 8001                         │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  Total RAM: ~14 GB minimum                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Manual Setup

### Step 1: Python Environment

```bash
# Create virtual environment
cd chatbot_service
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: spaCy Models

```bash
# Download medical NLP models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

### Step 3: Start Services

```bash
# Start PostgreSQL, Redis, ChromaDB
# (install and run separately, or use Docker for just these)
docker-compose up -d db redis chroma

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Environment Variables

Create a `.env` file in `chatbot_service/`:

### Database

```env
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://heartguard:password@localhost:5432/heartguard
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

### AI / LLM

```env
# MedGemma (local via llama.cpp)
LLM_API_URL=http://localhost:8090/v1
LLM_MODEL=medgemma-4b-it
LLM_API_KEY=not-needed

# Remote Embeddings (Colab via ngrok)
USE_REMOTE_EMBEDDINGS=true
COLAB_API_URL=https://your-ngrok-url.ngrok.io
EMBEDDING_MODEL=ncbi/MedCPT-Query-Encoder
EMBEDDING_DIM=768

# Optional: OpenAI (for Memori fact extraction)
OPENAI_API_KEY=sk-...
```

### Security

```env
# JWT
JWT_SECRET=your-256-bit-secret
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=1440

# Encryption
ENCRYPTION_KEY=your-32-byte-base64-key
```

### Application

```env
# General
APP_ENV=production
APP_DEBUG=false
LOG_LEVEL=INFO

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Web Search
TAVILY_API_KEY=tvly-...
```

---

## Database Setup

### PostgreSQL Schema

```bash
# Option 1: Run Alembic migrations (recommended)
cd chatbot_service
alembic upgrade head

# Option 2: Load schema directly
psql -U heartguard -d heartguard -f core_postgresql_schema.sql
```

### Migration History

```
┌──────────────────────────────────────────────────────────────┐
│                   ALEMBIC MIGRATIONS                         │
│                                                              │
│  001_initial ──▶ Base tables (users, conversations)          │
│       │                                                      │
│  002_sessions ──▶ Chat sessions                              │
│       │                                                      │
│  003_memories ──▶ Memory system tables                       │
│       │                                                      │
│  004_health_data ──▶ Vitals, medications, goals              │
│       │                                                      │
│  005_pgvector ──▶ Vector columns + indexes                   │
│       │                                                      │
│  006_drug_interactions ──▶ Drug pair tables                   │
│       │                                                      │
│  007_audit_logs ──▶ Audit trail + compliance                 │
└──────────────────────────────────────────────────────────────┘
```

### Seed Data

```bash
# Load drug data
python scripts/seed_drugs.py

# Load symptom data
python scripts/seed_symptoms.py

# Load sample interactions
python scripts/seed_interactions.py
```

---

## AI Model Setup

### MedGemma (Local LLM)

```bash
# Option 1: Docker (easiest)
# Already included in docker-compose.yml

# Option 2: Manual llama.cpp setup
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Download MedGemma model (GGUF format)
# Place in models/ directory
./server -m models/medgemma-4b-it.gguf \
         --host 0.0.0.0 \
         --port 8090 \
         --ctx-size 8192 \
         --n-gpu-layers 35
```

### Embedding Models

```bash
# MedCPT downloads automatically on first use
# Pre-download if needed:
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('ncbi/MedCPT-Query-Encoder')"
```

### ChromaDB Collection Setup

```bash
# Collections are created automatically on first use
# To pre-populate with medical knowledge:
python scripts/ingest_medical_docs.py
```

---

## Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Development server
npm run dev
# Opens at http://localhost:5173

# Production build
npm run build

# Mobile (Android)
npx cap sync android
npx cap open android
```

### Frontend Architecture

```
Frontend connects to backend:

  Browser ──▶ http://localhost:5173 (Vite dev server)
                    │
                    │ API calls
                    ▼
  Backend ──▶ http://localhost:8000 (FastAPI)
```

---

## Colab Integration

HeartGuard uses Google Colab as a remote GPU for embeddings. This is the **default mode** — the app sends text/images to a Colab-hosted Flask server via ngrok.

### On Google Colab

```python
# 1. Upload the colab/ folder to Colab
# 2. Install dependencies
!pip install -r requirements.txt

# 3. Start the embedding server (auto-creates ngrok tunnel)
from colab_server import start_server
start_server()
# → Prints ngrok URL (copy this)
```

The server hosts 3 models:
- **MedCPT** — text embeddings (768-dim) at `/embed_text`
- **SigLIP** — image embeddings (1152-dim) at `/embed_image`
- **MedGemma** — text generation at `/generate`

### On Local Machine

```bash
# Set the ngrok URL from Colab
export COLAB_API_URL=https://your-ngrok-url.ngrok.io

# Enable remote embeddings (default is true)
export USE_REMOTE_EMBEDDINGS=true

# Test connection
python colab/test_remote_embeddings.py
```

See `colab/REMOTE_EMBEDDINGS_GUIDE.md` for the full integration guide.

---

## Verification

### Check All Services

```bash
# API health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "2.1.0", "services": {...}}
```

### Service Status Checklist

```
┌──────────────────┬────────────────────────────────────┐
│ Service          │ How to verify                      │
├──────────────────┼────────────────────────────────────┤
│ API              │ curl localhost:8000/health          │
│ PostgreSQL       │ psql -c "SELECT 1"                 │
│ Redis            │ redis-cli ping → PONG              │
│ MedGemma         │ curl localhost:8090/v1/models       │
│ ChromaDB         │ curl localhost:8001/api/v1/heartbeat│
│ Frontend         │ Open localhost:5173 in browser      │
└──────────────────┴────────────────────────────────────┘
```

### Run Tests

```bash
cd chatbot_service

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test suite
pytest tests/test_agents.py -v
pytest tests/test_routes.py -v
pytest tests/test_tools.py -v
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| MedGemma not responding | Check port 8090, verify model is loaded, check RAM (needs 8GB) |
| Database connection error | Verify PostgreSQL is running, check DATABASE_URL |
| Redis connection refused | Start Redis: `docker-compose up -d redis` |
| ChromaDB timeout | Check port 8001, verify disk space for vector data |
| spaCy model not found | Run: `python -m spacy download en_core_web_sm` |
| Embeddings slow | Use Colab GPU integration or reduce batch size |
| JWT token expired | Default TTL is 24h, check JWT_EXPIRY_MINUTES |
| Out of memory | MedGemma needs 8GB alone; total stack needs 16GB+ |
| CORS errors | Check allowed origins in core/security.py |
| Migration fails | Run `alembic downgrade -1` then `alembic upgrade head` |

### Logs

```bash
# API logs
docker-compose logs -f api

# All service logs
docker-compose logs -f

# Application-level logs
tail -f chatbot_service/logs/app.log
```

### Reset Everything

```bash
# WARNING: This deletes all data

# Stop all services
docker-compose down -v

# Remove database volumes
docker volume prune

# Restart fresh
docker-compose up -d

# Re-run migrations
alembic upgrade head

# Re-seed data
python scripts/seed_drugs.py
```

---

## Ports Summary

```
┌──────────────┬───────────────────────────────┐
│ Port         │ Service                       │
├──────────────┼───────────────────────────────┤
│ 5173         │ Frontend (Vite dev server)    │
│ 8000         │ Backend API (FastAPI)         │
│ 8090         │ MedGemma LLM (llama.cpp)     │
│ 5432         │ PostgreSQL                    │
│ 6379         │ Redis                         │
│ 8001         │ ChromaDB                      │
└──────────────┴───────────────────────────────┘
```

---

*This document describes the Setup process for HeartGuard AI v2.1.0*
