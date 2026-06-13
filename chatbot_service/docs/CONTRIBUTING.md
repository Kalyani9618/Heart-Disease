# Contributing to HeartGuard AI

> Guidelines for contributing to the HeartGuard AI codebase.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack & Models](#tech-stack--models)
3. [Project Structure](#project-structure)
4. [Development Setup](#development-setup)
5. [Code Style](#code-style)
6. [Workflow](#workflow)
7. [Testing](#testing)
8. [Documentation](#documentation)

---

## Project Overview

HeartGuard AI is a medical chatbot for cardiac health. The backend is a **FastAPI** application with a **LangGraph** multi-agent orchestrator, **RAG** pipeline, and **Memori** patient memory system. The frontend is **React 19 + TypeScript** with **Capacitor** for Android.

---

## Tech Stack & Models

### AI / ML Models

| Model | Purpose | Dimensions | Runtime |
|-------|---------|------------|---------|
| **MedGemma-4B-IT** | Primary LLM (generation, reasoning, routing) | — | llama.cpp on port 8090 |
| **MedCPT-Query-Encoder** | Medical text embeddings | 768 | Remote Colab or local |
| **SigLIP** | Medical image embeddings | 1,152 | Remote Colab or local |
| **MS-MARCO Cross-Encoder** | Reranking retrieved documents | — | Local CPU |
| **scikit-learn / XGBoost** | Heart disease risk prediction | — | Local CPU |
| **spaCy (en_core_web_sm)** | Medical NLP (NER, negation) | — | Local CPU |

### Backend

| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Runtime |
| FastAPI | Web framework (port 8000) |
| LangGraph / LangChain | Multi-agent orchestration |
| PostgreSQL 15 | Primary database (27+ tables) |
| Redis 7 | Cache, sessions, job queue |
| ChromaDB | Vector database (125K+ docs) |
| Alembic | Database migrations |

### Frontend

| Technology | Purpose |
|-----------|---------|
| React 19 + TypeScript 5.8 | UI framework |
| Zustand 5 | State management |
| Tailwind CSS 4 | Styling |
| Vite 6 | Build tool |
| Capacitor 8 | Android mobile |

---

## Project Structure

```
chatbot_service/
├── main.py                    # FastAPI entry point
├── agents/                    # AI agents (23 files)
│   ├── langgraph_orchestrator.py   # Central LangGraph orchestrator
│   ├── heart_predictor.py          # ML heart disease prediction
│   ├── evaluation.py               # LLM-as-judge evaluator
│   ├── components/                 # Agent components (thinking, vision, triage, etc.)
│   └── deep_research_agent/        # Multi-source deep research
├── core/                      # Core services (55 files)
│   ├── config/                     # AppConfig, RAG config
│   ├── llm/                        # LLM gateway (MedGemma), guardrails
│   ├── prompts/                    # System prompts + registry (lazy-loaded)
│   ├── database/                   # PostgreSQL connector, ORM models
│   ├── services/                   # Cache, chat history, encryption, WebSocket
│   ├── safety/                     # Hallucination grading
│   ├── compliance/                 # PII scrubbing
│   ├── monitoring/                 # Prometheus metrics
│   └── observability/              # Tracing
├── routes/                    # API routes (37 files)
│   ├── core/                       # Auth, chat, user, session, feedback
│   ├── health/                     # Vitals, medications, symptoms, drugs, FHIR
│   └── admin/                      # Analytics, monitoring, WebSocket, webhooks
├── tools/                     # Agent tools (27 files)
│   ├── openfda/                    # FDA drug/device lookup
│   ├── fhir/                       # FHIR EHR integration
│   └── *.py                        # Drug interaction, symptom, nutrition, etc.
├── rag/                       # RAG pipeline (79 files)
│   ├── embedding/                  # MedCPT + SigLIP embedding service
│   ├── retrieval/                  # Tiered, RAPTOR, fusion retrieval
│   ├── store/                      # ChromaDB + vector store
│   ├── nlp/                        # Medical NLP (tokenizer, NER, negation)
│   ├── trust/                      # Source validation, conflict detection
│   ├── knowledge_graph/            # Graph RAG, medical ontology
│   ├── multimodal/                 # Image + document processing
│   ├── memory/                     # Memori ↔ RAG bridge
│   └── pipeline/                   # Self-RAG, CRAG, query optimization
├── memori/                    # Patient memory system (63 files)
│   ├── agents/                     # Memory, retrieval, conscious agents
│   ├── config/                     # Memory manager, pool config
│   ├── core/                       # Conversation, database, memory core
│   ├── database/                   # PostgreSQL adapters, migrations, queries
│   ├── integrations/               # OpenAI, Anthropic, LiteLLM
│   ├── long_term/                  # Fact extraction
│   ├── short_term/                 # Redis buffer
│   ├── security/                   # Auth
│   ├── tools/                      # Memory tool
│   └── utils/                      # Helpers, validators, security
├── data/                      # Static data (drugs.json, symptoms.json)
├── models/                    # ML model files
├── tests/                     # Test suite
├── colab/                     # Google Colab embedding server
├── Docker/                    # Docker Compose & Dockerfiles
├── docs/                      # Architecture documentation
└── alembic/                   # Database migrations

frontend/
├── App.tsx                    # Root component
├── screens/                   # UI screens
├── components/                # Reusable components
├── services/                  # API client
├── store/                     # Zustand state
├── contexts/                  # React contexts
└── android/                   # Capacitor Android project
```

---

## Development Setup

```bash
# 1. Clone
git clone <repo-url> Heart
cd Heart/chatbot_service

# 2. Python environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt

# 3. spaCy models
python -m spacy download en_core_web_sm

# 4. Start services (PostgreSQL, Redis, ChromaDB)
cd Docker && docker-compose up -d db redis chroma && cd ..

# 5. Database migrations
alembic upgrade head

# 6. Start backend
uvicorn main:app --port 8000 --reload

# 7. Frontend (new terminal)
cd ../frontend
npm install
npm run dev
```

See [SETUP.md](SETUP.md) for full setup, environment variables, and model configuration.

---

## Code Style

This project enforces consistent formatting and linting via **Black** and **Ruff**.

### Setup

```bash
pip install black ruff pre-commit
pre-commit install
```

### Commands

```bash
# Format
black .
black --check .          # check only

# Lint
ruff check --fix .
ruff check .             # check only

# Pre-commit (runs on git commit automatically)
pre-commit run --all-files
```

### Configuration

- **Black** and **Ruff** settings live in `pyproject.toml`.
- **Pre-commit** hook versions are pinned in `.pre-commit-config.yaml`.

| Tool | Purpose |
|------|---------|
| Black | Opinionated code formatter (line-length 100) |
| Ruff | Fast linter — pycodestyle, Pyflakes, isort, bugbear, and more |
| pre-commit | Git hook runner — enforces style on every commit |

---

## Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** — follow the code style and architecture patterns.

3. **Run tests** before committing:
   ```bash
   pytest
   ```

4. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add new drug interaction alert"
   ```
   Prefix conventions: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

5. **Push** and open a pull request against `main`.

---

## Testing

```bash
cd chatbot_service

# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific suites
pytest tests/test_agents.py -v
pytest tests/test_routes.py -v
pytest tests/test_tools.py -v
```

---

## Documentation

Architecture docs live in `chatbot_service/docs/`:

| Document | Covers |
|----------|--------|
| [COMPLETE_ARCHITECTURE_REPORT.md](COMPLETE_ARCHITECTURE_REPORT.md) | Full system overview |
| [AGENTS_ARCHITECTURE_REPORT.md](AGENTS_ARCHITECTURE_REPORT.md) | AI agents & orchestrator |
| [CORE_ARCHITECTURE_REPORT.md](CORE_ARCHITECTURE_REPORT.md) | Core services & security |
| [DATABASE_ARCHITECTURE_REPORT.md](DATABASE_ARCHITECTURE_REPORT.md) | Database & storage |
| [RAG_ARCHITECTURE_REPORT.md](RAG_ARCHITECTURE_REPORT.md) | RAG pipeline |
| [ROUTES_ARCHITECTURE_REPORT.md](ROUTES_ARCHITECTURE_REPORT.md) | API routes & endpoints |
| [TOOLS_ARCHITECTURE_REPORT.md](TOOLS_ARCHITECTURE_REPORT.md) | Agent tools |
| [MEMORI_ARCHITECTURE_REPORT.md](MEMORI_ARCHITECTURE_REPORT.md) | Memory system |
| [SETUP.md](SETUP.md) | Setup & installation |

When you add or change files, update the relevant doc's **File Reference** table.

---

*HeartGuard AI v2.1.0*
