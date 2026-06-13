# HeartGuard AI

### AI-Powered Cardiac Health Assistant

```
    ♥ HeartGuard AI v2.1.0 ♥

    Intelligent cardiac health chatbot with
    medical-grade knowledge retrieval and
    multi-agent AI reasoning.
```

---

## What is HeartGuard AI?

HeartGuard AI is a medical chatbot that helps patients and healthcare providers with heart health questions. It uses a team of AI agents backed by real medical databases to provide personalized cardiac health guidance.

**It is NOT a replacement for a doctor** — it's a smart assistant that helps users understand their heart health better.

---

## Features

| Feature | Description |
|---------|-------------|
| **AI Chat** | Conversational assistant for heart health questions |
| **Heart Risk Prediction** | ML-powered heart disease risk assessment |
| **Drug Interaction Check** | Warns about dangerous drug combinations |
| **Symptom Triage** | Urgency-level assessment of symptoms |
| **Patient Memory** | Remembers your medications, conditions, and preferences |
| **Deep Research** | Multi-source investigation for complex medical queries |
| **Medical Standards** | FHIR, OpenFDA, DICOM integration |
| **Secure & Compliant** | HIPAA, GDPR, AES-256 encryption, PII scrubbing |
| **Mobile Ready** | Android app via Capacitor |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      HeartGuard AI                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  FRONTEND — React 19 + TypeScript + Capacitor 8      │  │
│  │  Web & Android app with real-time chat               │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │ REST API                          │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │  BACKEND — FastAPI + Python 3.9+                      │  │
│  │                                                       │  │
│  │  ┌───────────────────────────────────────────────┐    │  │
│  │  │  AI Engine                                    │    │  │
│  │  │  • LangGraph Orchestrator (multi-agent)       │    │  │
│  │  │  • 27 Medical Tools (drugs, FHIR, FDA...)     │    │  │
│  │  │  • RAG Pipeline (125K+ medical documents)     │    │  │
│  │  │  • Memori (patient memory system)             │    │  │
│  │  └───────────────────────────────────────────────┘    │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │  DATA — PostgreSQL + Redis + ChromaDB              │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │  LLM — MedGemma-4B-IT (local, via llama.cpp)         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend

| Technology | Purpose |
|-----------|---------|
| FastAPI | Web framework |
| Python 3.9+ | Runtime |
| LangGraph | Multi-agent orchestration |
| LangChain | AI toolkit |
| MedGemma-4B-IT | Medical LLM (local) |
| llama.cpp | LLM inference server |
| MedCPT | Medical text embeddings |
| scikit-learn / XGBoost | Heart disease ML models |
| spaCy | Medical NLP |
| Alembic | Database migrations |

### Frontend

| Technology | Purpose |
|-----------|---------|
| React 19 | UI framework |
| TypeScript 5.8 | Type safety |
| Zustand 5 | State management |
| Tailwind CSS 4 | Styling |
| Vite 6 | Build tool |
| Capacitor 8 | Mobile (Android) |

### Databases

| Technology | Purpose |
|-----------|---------|
| PostgreSQL 15 | Primary database (27+ tables) |
| Redis 7 | Cache, sessions, job queue |
| ChromaDB | Vector database (125K+ docs) |

---

## Quick Start

### Docker (Recommended)

```bash
# 1. Clone
git clone <repo-url> Heart
cd Heart

# 2. Configure
cp chatbot_service/.env.example chatbot_service/.env
# Edit .env with your settings

# 3. Start
cd chatbot_service/Docker
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
```

### Manual

```bash
# Backend
cd chatbot_service
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
alembic upgrade head
uvicorn main:app --port 8000 --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

See [chatbot_service/docs/SETUP.md](chatbot_service/docs/SETUP.md) for full setup instructions.

---

## Project Structure

```
Heart/
├── chatbot_service/           # Backend application
│   ├── main.py                # FastAPI entry point
│   ├── agents/                # AI agents (LangGraph orchestrator)
│   ├── core/                  # Core services (config, security, LLM)
│   ├── routes/                # 37 API route files (core/health/admin)
│   ├── tools/                 # 27 agent tools (drugs, FHIR, FDA)
│   ├── rag/                   # RAG pipeline (embeddings, retrieval)
│   ├── memori/                # Memory system (v2.3.0)
│   ├── data/                  # Static data (drugs, symptoms)
│   ├── models/                # ML models
│   ├── tests/                 # Test suite
│   ├── Docker/                # Docker compose & Dockerfiles
│   ├── docs/                  # Architecture documentation
│   └── alembic/               # Database migrations
│
├── frontend/                  # React + TypeScript frontend
│   ├── App.tsx                # Root component
│   ├── screens/               # UI screens
│   ├── components/            # Reusable components
│   ├── services/              # API client
│   ├── store/                 # Zustand state
│   └── android/               # Capacitor Android project
│
└── README.md                  # This file
```

---

## Services & Ports

| Port | Service |
|------|---------|
| 5173 | Frontend (Vite) |
| 8000 | Backend API (FastAPI) |
| 8090 | MedGemma LLM (llama.cpp) |
| 5432 | PostgreSQL |
| 6379 | Redis |
| 8001 | ChromaDB |

---

## Documentation

Detailed architecture reports are in `chatbot_service/docs/`:

| Document | Description |
|----------|-------------|
| [COMPLETE_ARCHITECTURE_REPORT.md](chatbot_service/docs/COMPLETE_ARCHITECTURE_REPORT.md) | Full system overview |
| [AGENTS_ARCHITECTURE_REPORT.md](chatbot_service/docs/AGENTS_ARCHITECTURE_REPORT.md) | AI agents & orchestrator |
| [CORE_ARCHITECTURE_REPORT.md](chatbot_service/docs/CORE_ARCHITECTURE_REPORT.md) | Core services & security |
| [DATABASE_ARCHITECTURE_REPORT.md](chatbot_service/docs/DATABASE_ARCHITECTURE_REPORT.md) | Database & storage |
| [RAG_ARCHITECTURE_REPORT.md](chatbot_service/docs/RAG_ARCHITECTURE_REPORT.md) | RAG pipeline |
| [ROUTES_ARCHITECTURE_REPORT.md](chatbot_service/docs/ROUTES_ARCHITECTURE_REPORT.md) | API routes & endpoints |
| [TOOLS_ARCHITECTURE_REPORT.md](chatbot_service/docs/TOOLS_ARCHITECTURE_REPORT.md) | Agent tools |
| [MEMORI_ARCHITECTURE_REPORT.md](chatbot_service/docs/MEMORI_ARCHITECTURE_REPORT.md) | Memory system |
| [SETUP.md](chatbot_service/docs/SETUP.md) | Setup & installation |

---

## API Endpoints (Highlights)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message to AI |
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Login |
| GET | `/api/health/predict` | Heart disease prediction |
| GET | `/api/drugs/interactions` | Check drug interactions |
| POST | `/api/symptoms/check` | Symptom triage |
| GET | `/api/vitals` | Get patient vitals |
| GET | `/api/medications` | List medications |
| GET | `/health` | Service health check |

---

## System Requirements

```
Minimum:  16 GB RAM, 4 CPU cores, 250 GB disk
Recommended: 32 GB RAM, 8 CPU cores, 500 GB SSD, GPU
```

---

## Contributing

See [chatbot_service/docs/CONTRIBUTING.md](chatbot_service/docs/CONTRIBUTING.md) for contribution guidelines.

---

## License

Private — All rights reserved.

---

*HeartGuard AI v2.1.0 — AI-powered cardiac health assistant*
