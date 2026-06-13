# Complete Architecture Report

## HeartGuard AI — Full System Architecture

> **Version:** 2.1.0  
> **Last Updated:** February 2026  
> **Author:** HeartGuard AI Team

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Map](#component-map)
4. [Data Flow](#data-flow)
5. [AI Pipeline](#ai-pipeline)
6. [Infrastructure](#infrastructure)
7. [Security Architecture](#security-architecture)
8. [Technology Stack](#technology-stack)
9. [Detailed Reports](#detailed-reports)

---

## System Overview

HeartGuard AI is a medical chatbot specializing in heart disease. It combines a multi-agent AI system with retrieval-augmented generation (RAG) to provide personalized cardiac health guidance.

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                     HeartGuard AI v2.1.0                          │
│                                                                  │
│    "AI-powered cardiac health assistant with medical-grade       │
│     knowledge retrieval and multi-agent reasoning"               │
│                                                                  │
│    Users: Patients, healthcare providers                         │
│    Input: Text, medical images, health data                      │
│    Output: Personalized cardiac health guidance                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### What It Does

- **Chat** — Conversational AI for heart health questions
- **Predict** — Heart disease risk assessment using ML models
- **Alert** — Drug interaction warnings and symptom triage
- **Remember** — Persistent patient memory across sessions
- **Research** — Deep research mode for complex medical queries
- **Integrate** — FHIR, OpenFDA, DICOM medical standards

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          HeartGuard AI                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                         FRONTEND                                    │ │
│  │                                                                     │ │
│  │   React 19 + TypeScript + Capacitor 8 (Web + Android)              │ │
│  │   Zustand state management + Tailwind CSS                          │ │
│  │                                                                     │ │
│  │   Screens: Chat, Dashboard, Medications, Vitals, Profile           │ │
│  └────────────────────────────────┬────────────────────────────────────┘ │
│                                   │ REST API + WebSocket                 │
│  ┌────────────────────────────────▼────────────────────────────────────┐ │
│  │                         BACKEND (FastAPI)                           │ │
│  │                                                                     │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │ │
│  │  │ 37 Route│  │ Core     │  │ Security │  │ Rate Limiter     │   │ │
│  │  │ Files    │  │ Services │  │ JWT+PII  │  │ Redis-backed     │   │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────────────┘   │ │
│  │       │              │             │              │                 │ │
│  │  ┌────▼──────────────▼─────────────▼──────────────▼─────────────┐  │ │
│  │  │                   AI ENGINE                                   │  │ │
│  │  │                                                               │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │ │
│  │  │  │           LangGraph Orchestrator (1409 lines)           │  │  │ │
│  │  │  │                                                         │  │  │ │
│  │  │  │  ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐ │  │  │ │
│  │  │  │  │ Drug     │ │ Heart   │ │ Thinking │ │ Clinical   │ │  │  │ │
│  │  │  │  │ Expert   │ │ Predict │ │ Agent    │ │ Reasoning  │ │  │  │ │
│  │  │  │  └──────────┘ └─────────┘ └──────────┘ └────────────┘ │  │  │ │
│  │  │  │  ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐ │  │  │ │
│  │  │  │  │ Medical  │ │ Data    │ │ Deep     │ │ Researcher │ │  │  │ │
│  │  │  │  │ Analyst  │ │ Analyst │ │ Research │ │            │ │  │  │ │
│  │  │  │  └──────────┘ └─────────┘ └──────────┘ └────────────┘ │  │  │ │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │ │
│  │  │                                                               │  │ │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │  │ │
│  │  │  27 Tools    │  │  RAG Pipeline │  │  Memori v2.3.0  │    │  │ │
│  │  │  Search,FHIR,│  │  Self-RAG     │  │  Patient memory │    │  │ │
│  │  │  │  FDA, DICOM  │  │  CRAG, HyDE   │  │  Fact extraction│    │  │ │
│  │  │  └──────────────┘  └──────────────┘  └──────────────────┘    │  │ │
│  │  └───────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                         DATA LAYER                                  │ │
│  │                                                                     │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │ │
│  │  │PostgreSQL│  │  Redis   │  │ ChromaDB │                              │ │
│  │  │          │  │          │  │          │                              │ │
│  │  │ 27+      │  │ Cache    │  │ 125K+    │                              │ │
│  │  │ tables   │  │ Queue    │  │ medical  │                              │ │
│  │  │ pgvector │  │ Sessions │  │ docs     │                              │ │
│  │  └──────────┘  └──────────┘  └──────────┘                              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     LLM LAYER                                       │ │
│  │                                                                     │ │
│  │  MedGemma-4B-IT (local, via llama.cpp, port 8090)                  │ │
│  │  MedCPT embeddings (768-dim) + SigLIP images (1152-dim)           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Component Map

```
HeartGuard AI
├── Frontend (React 19 + Capacitor 8)
│   ├── Screens: Chat, Dashboard, Medications, Vitals, Profile, Settings
│   ├── State: Zustand stores (auth, chat, health, UI)
│   ├── API: Axios client with retry + offline queue
│   └── Mobile: Capacitor for Android
│
├── Backend (FastAPI + Python 3.9+)
│   ├── Routes (37 files, nested in core/health/admin/)
│   │   ├── Core: chat, auth, users, documents, feedback, memory, profile
│   │   ├── Health: predictions, vision, smartwatch, tools, medical_ai
│   │   └── Admin: db_health, evaluation, models, nlp_debug, rag_memory
│   │
│   ├── Core Services
│   │   ├── Config: Layered settings (TOML + env + DB)
│   │   ├── Security: JWT, Argon2, AES-256, PII scrubbing
│   │   ├── LLM: MedGemma gateway with fallback
│   │   ├── DI: Dependency injection container
│   │   └── Resilience: Circuit breaker, rate limiter, cache
│   │
│   ├── AI Agents (LangGraph)
│   │   ├── Orchestrator: State machine routing
│   │   ├── Specialists: Drug, Heart, Data, Clinical, Research
│   │   ├── ThinkingAgent: Chain-of-thought reasoning
│   │   └── Deep Research: Multi-source investigation
│   │
│   ├── Tools (27 Python files)
│   │   ├── Core: agentic_tools, entity_validator, text_to_sql, web_search
│   │   ├── External: OpenFDA (8), FHIR (2), DICOM (1)
│   │   └── Specialized: Medical coding, clinical guidelines, medical search
│   │
│   ├── RAG Pipeline
│   │   ├── Embeddings: MedCPT (text) + SigLIP (images)
│   │   ├── Vector stores: ChromaDB + pgvector
│   │   ├── Retrieval: Tiered, HyDE, RAPTOR, Intent Router
│   │   └── Verification: Self-RAG + CRAG trust scoring
│   │
│   └── Memori (Memory System v2.3.0)
│       ├── Fact extraction via LLM (12 medical categories)
│       ├── Multi-tier cache (L1 memory + L2 Redis + L3 DB)
│       ├── Search: Full-text + semantic
│       └── Multi-tenant isolation
│
├── Databases
│   ├── PostgreSQL 15 (primary, 27+ tables, pgvector)
│   ├── Redis 7 (cache, sessions, job queue)
│   └── ChromaDB (125K+ medical document vectors)
│
└── LLM Infrastructure
    ├── MedGemma-4B-IT via llama.cpp (local, port 8090)
    ├── MedCPT-Query-Encoder (768-dim text embeddings)
    ├── SigLIP (1152-dim image embeddings)
    └── MS-MARCO cross-encoder (reranking)
```

---

## Data Flow

### Chat Request Flow

```
┌────────┐     ┌──────────┐     ┌──────────────────────────────────┐
│  User  │────▶│ Frontend │────▶│  POST /api/chat                  │
│        │     │ React    │     │                                  │
└────────┘     └──────────┘     └──────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │              MIDDLEWARE PIPELINE              │
                    │                                              │
                    │  1. Rate Limiter (60 req/min per user)       │
                    │  2. JWT Authentication                       │
                    │  3. Input Sanitization (XSS, injection)      │
                    │  4. PII Detection (SSN, phone, email)        │
                    │  5. Request Logging                          │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │           LANGGRAPH ORCHESTRATOR              │
                    │                                              │
                    │  1. Classify query intent                    │
                    │     ├── SemanticRouter (fast, embedding)     │
                    │     └── Supervisor (slow, LLM fallback)      │
                    │                                              │
                    │  2. Route to specialist agent                 │
                    │     ├── drug_expert_node                     │
                    │     ├── heart_analyst_node                   │
                    │     ├── medical_analyst_node                 │
                    │     ├── data_analyst_node                    │
                    │     └── thinking_node                        │
                    │                                              │
                    │  3. Agent uses tools + RAG                   │
                    │     ├── Retrieve from ChromaDB/pgvector      │
                    │     ├── Call tools (drug check, vitals...)   │
                    │     └── Get patient memory from Memori       │
                    │                                              │
                    │  4. Generate response via MedGemma            │
                    │                                              │
                    │  5. Safety check (medical disclaimer)        │
                    │                                              │
                    │  6. Store conversation in Memori              │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │  Response to user with:                      │
                    │  • Answer text                               │
                    │  • Confidence score                          │
                    │  • Sources/citations                         │
                    │  • Safety disclaimers                        │
                    │  • Suggested follow-up questions              │
                    └──────────────────────────────────────────────┘
```

### Heart Disease Prediction Flow

```
Patient Data ─▶ Feature Engineering ─▶ ML Models ─▶ Risk Score

Input:                    Models:              Output:
├── Age                   ├── XGBoost          ├── Risk: 0-100%
├── Sex                   ├── Random Forest    ├── Category: Low/Med/High
├── Blood Pressure        ├── Logistic Reg     ├── Key factors
├── Cholesterol           └── Ensemble vote    ├── Recommendations
├── Heart Rate                                 └── Confidence
├── Chest Pain Type
├── Fasting Blood Sugar
└── ECG Results
```

---

## AI Pipeline

### Query Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI PIPELINE                                   │
│                                                                     │
│  STEP 1: UNDERSTAND                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  NLP Pipeline (spaCy)                                        │   │
│  │  • Tokenize                                                  │   │
│  │  • Entity recognition (drugs, conditions, symptoms)          │   │
│  │  • Negation detection ("no chest pain" ≠ "chest pain")       │   │
│  │  • Intent classification                                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│  STEP 2: RETRIEVE                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  RAG Pipeline                                                │   │
│  │  • Embed query with MedCPT (768-dim)                         │   │
│  │  • Search ChromaDB (125K+ medical docs)                      │   │
│  │  • Rerank with MS-MARCO cross-encoder                        │   │
│  │  • Verify relevance with Self-RAG grader                     │   │
│  │  • Fallback to web search (CRAG) if low confidence           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│  STEP 3: REMEMBER                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Memori                                                      │   │
│  │  • Retrieve patient's stored memories                        │   │
│  │  • Add medications, conditions, preferences to context       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│  STEP 4: REASON                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Specialist Agent + ThinkingAgent                            │   │
│  │  • Chain-of-thought reasoning                                │   │
│  │  • Tool calls (drug checks, vitals, etc.)                    │   │
│  │  • Multi-step analysis for complex queries                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│  STEP 5: GENERATE                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  MedGemma LLM                                                │   │
│  │  • Generate personalized response                            │   │
│  │  • Include citations from RAG                                │   │
│  │  • Add medical safety disclaimers                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│  STEP 6: LEARN                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Post-Response                                               │   │
│  │  • Extract new facts from conversation → Store in Memori     │   │
│  │  • Log for evaluation metrics                                │   │
│  │  • Update user engagement analytics                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure

### Docker Deployment

```
┌──────────────────────────────────────────────────────────────────┐
│                    DOCKER COMPOSE STACK                           │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  medgemma        │  8 GB RAM  │  Port 8090               │   │
│  │  llama.cpp + MedGemma-4B-IT GGUF model                   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                         ▲                                        │
│                         │ OpenAI-compatible API                  │
│  ┌──────────────────────┴────────────────────────────────────┐   │
│  │  api             │  4 GB RAM  │  Port 8000               │   │
│  │  FastAPI + Uvicorn + all Python deps                      │   │
│  └───────────────────────────────────────────────────────────┘   │
│        │                    │                    │                │
│        ▼                    ▼                    ▼                │
│  ┌──────────┐       ┌──────────┐       ┌──────────────────┐     │
│  │ db       │       │ redis    │       │ chroma           │     │
│  │ 1 GB     │       │ 512 MB   │       │ 1 GB             │     │
│  │ PG 15    │       │ Redis 7  │       │ ChromaDB         │     │
│  │ :5432    │       │ :6379    │       │ :8001            │     │
│  └──────────┘       └──────────┘       └──────────────────┘     │
│                                                                  │
│  Total: ~14 GB RAM minimum                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                                │
│                                                                  │
│  Layer 1: NETWORK                                                │
│  ├── HTTPS/TLS encryption                                        │
│  ├── CORS origin whitelist                                       │
│  └── Rate limiting (Redis-backed, 60 req/min)                    │
│                                                                  │
│  Layer 2: AUTHENTICATION                                         │
│  ├── JWT (HS256, 24h expiry)                                     │
│  ├── Argon2 password hashing                                     │
│  └── Session management                                          │
│                                                                  │
│  Layer 3: INPUT SAFETY                                           │
│  ├── SQL injection detection                                     │
│  ├── XSS payload detection                                       │
│  ├── Command injection prevention                                │
│  └── Input length limits                                         │
│                                                                  │
│  Layer 4: DATA PROTECTION                                        │
│  ├── AES-256-GCM field encryption (SSN, medical records)         │
│  ├── 4-layer PII scrubber (regex + spaCy NER + Presidio + LLM)  │
│  ├── Audit logging (all data access)                             │
│  └── Data retention policies                                     │
│                                                                  │
│  Layer 5: COMPLIANCE                                             │
│  ├── HIPAA (healthcare data rules)                               │
│  ├── GDPR (EU privacy rights)                                    │
│  └── SOC2 (security controls)                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend

| Category | Technology | Version |
|----------|-----------|---------|
| Framework | FastAPI | 0.104+ |
| Runtime | Python | 3.9+ |
| ASGI Server | Uvicorn | Latest |
| AI Orchestration | LangGraph | Latest |
| AI Toolkit | LangChain | Latest |
| LLM | MedGemma-4B-IT | Latest |
| LLM Server | llama.cpp | Latest |
| Embeddings | MedCPT | Latest |
| ML | scikit-learn, XGBoost | Latest |
| NLP | spaCy | 3.x |
| Job Queue | ARQ (Redis) | Latest |
| Migrations | Alembic | Latest |

### Frontend

| Category | Technology | Version |
|----------|-----------|---------|
| Framework | React | 19.2 |
| Language | TypeScript | 5.8 |
| State | Zustand | 5.0 |
| CSS | Tailwind CSS | 4.1 |
| Build | Vite | 6.2 |
| Mobile | Capacitor | 8.0 |
| HTTP | Axios | Latest |
| Charts | Chart.js / Recharts | Latest |

### Data

| Category | Technology | Version |
|----------|-----------|---------|
| Primary DB | PostgreSQL | 15 |
| Cache | Redis | 7 |
| Vector DB | ChromaDB | Latest |
| Vectors | pgvector | Latest |
| Async Driver | asyncpg | Latest |

---

## Detailed Reports

For deeper information on each subsystem, see these reports:

| Report | What It Covers |
|--------|---------------|
| [AGENTS_ARCHITECTURE_REPORT.md](AGENTS_ARCHITECTURE_REPORT.md) | AI agents, orchestrator, routing, reasoning |
| [CORE_ARCHITECTURE_REPORT.md](CORE_ARCHITECTURE_REPORT.md) | Core services, config, security, LLM, DI |
| [DATABASE_ARCHITECTURE_REPORT.md](DATABASE_ARCHITECTURE_REPORT.md) | Database schema, migrations, caching |
| [RAG_ARCHITECTURE_REPORT.md](RAG_ARCHITECTURE_REPORT.md) | RAG pipeline, embeddings, vector stores |
| [ROUTES_ARCHITECTURE_REPORT.md](ROUTES_ARCHITECTURE_REPORT.md) | All 37 API route files and endpoints |
| [TOOLS_ARCHITECTURE_REPORT.md](TOOLS_ARCHITECTURE_REPORT.md) | 27 agent tools (search, FHIR, FDA, etc.) |
| [MEMORI_ARCHITECTURE_REPORT.md](MEMORI_ARCHITECTURE_REPORT.md) | Memory system, fact extraction, caching |
| [SETUP.md](SETUP.md) | Installation, configuration, deployment |

---

## Project Statistics

```
┌──────────────────────────────────────────────────┐
│  CODEBASE METRICS                                │
│                                                  │
│  Backend Python files:     ~284                  │
│  Frontend TypeScript:      ~50+                  │
│  Total route files:        37                    │
│  Total agent files:        23                    │
│  Total tool files:         27                    │
│  Total RAG files:          ~79                   │
│  Total Memori files:       ~63                   │
│  Database tables:          27+                   │
│  Alembic migrations:       7                     │
│  Docker services:          5                     │
│  ChromaDB documents:       125,000+              │
│  Drug interactions:        25+                   │
│  Symptoms tracked:         91                    │
│  Drugs in database:        110+                  │
│  API endpoints:            80+                   │
└──────────────────────────────────────────────────┘
```

---

*This document provides the Complete Architecture Overview of HeartGuard AI v2.1.0*
