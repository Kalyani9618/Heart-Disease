# Core Architecture Report

## HeartGuard AI — Core Services Layer

> **Version:** 2.1.0  
> **Last Updated:** February 2026  
> **Author:** HeartGuard AI Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Configuration Management](#configuration-management)
4. [Security Layer](#security-layer)
5. [LLM Integration](#llm-integration)
6. [Dependency Injection](#dependency-injection)
7. [Resilience Patterns](#resilience-patterns)
8. [NLP Services](#nlp-services)
9. [Prompt Management](#prompt-management)
10. [Monitoring & Observability](#monitoring--observability)
11. [Supporting Services](#supporting-services)
12. [File Reference](#file-reference)

---

## Overview

The **Core** layer is the foundation of HeartGuard AI. It provides shared services that every other layer depends on — configuration, security, LLM access, database connections, monitoring, and more. Think of it as the "utility belt" of the application.

**Key Principle:** Every service is accessed through a singleton pattern with lazy initialization. Nothing is loaded until it's actually needed.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CORE LAYER                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    APP CONFIG                             │   │
│  │  Single source of truth for all settings                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │   │
│  │  │LLMConfig │ │RAGConfig │ │DBConfig  │ │APIConfig │    │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  DI CONTAINER │   │  SECURITY    │   │  LLM GATEWAY │        │
│  │              │   │              │   │              │        │
│  │ Lazy-loads   │   │ JWT Auth     │   │ MedGemma     │        │
│  │ all services │   │ Argon2 Hash  │   │ via OpenAI   │        │
│  │ on demand    │   │ Rate Limit   │   │ compatible   │        │
│  │              │   │ Audit Log    │   │ local API    │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  RESILIENCE  │   │  COMPLIANCE  │   │  GUARDRAILS  │        │
│  │              │   │              │   │              │        │
│  │ Circuit      │   │ PII Scrubber │   │ Safety       │        │
│  │ Breaker      │   │ (4-layer)    │   │ Validation   │        │
│  │ Graceful     │   │ Encryption   │   │ Hallucination│        │
│  │ Degradation  │   │ (AES-256)    │   │ Grading      │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  MONITORING  │   │   PROMPTS    │   │ NLP SERVICES │        │
│  │              │   │              │   │              │        │
│  │ Prometheus   │   │ Registry     │   │ spaCy NER    │        │
│  │ Metrics      │   │ Templates    │   │ Drug Detect  │        │
│  │ Tracing      │   │ Hot Reload   │   │ Multilingual │        │
│  │ Grafana      │   │ Versioning   │   │              │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Management

All settings flow from a single `AppConfig` class:

```
┌────────────────────────────────────────────────────┐
│                 AppConfig                          │
│          (Single Source of Truth)                   │
│                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │ LLMConfig  │  │ RAGConfig  │  │ DBConfig   │   │
│  │            │  │            │  │            │   │
│  │ model_name │  │ top_k: 5   │  │ host       │   │
│  │ base_url   │  │ embed_model│  │ port: 5432 │   │
│  │ temperature│  │ relevance  │  │ pool: 10-30│   │
│  │ max_tokens │  │ threshold  │  │ database   │   │
│  └────────────┘  └────────────┘  └────────────┘   │
│                                                    │
│  ┌────────────┐  ┌────────────┐                    │
│  │ APIConfig  │  │SpaCyConfig │                    │
│  │            │  │            │                    │
│  │ host       │  │ model      │                    │
│  │ port: 5001 │  │ med_ruler  │                    │
│  │ workers    │  │            │                    │
│  └────────────┘  └────────────┘                    │
│                                                    │
│  Loads from: .env file → environment variables     │
│  Legacy mapping: LLAMA_LOCAL_* → MEDGEMMA_*        │
└────────────────────────────────────────────────────┘
```

**Key Files:**
| File | Purpose |
|------|---------|
| `config/app_config.py` | Main AppConfig (Pydantic model, thread-safe singleton) |
| `config/rag_config.py` | RAG-specific settings (embedding model, weights) |
| `config/rag_paths.py` | Filesystem path resolution (drugs.json, models, logs) |
| `config/rag_settings.py` | RAG settings singleton with runtime updates |
| `config/compat.py` | Backward compatibility shim for old Settings interface |
| `config/legacy_settings.py` | DEPRECATED — scheduled for removal in v2.0 |

---

## Security Layer

HeartGuard implements **defense-in-depth** security:

```
┌─────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                           │
│                                                             │
│  Layer 1: API BOUNDARY                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ JWT Authentication (HS256)                           │    │
│  │ • 32+ char secret key with entropy check            │    │
│  │ • Token expiry + refresh flow                       │    │
│  │ • Forbidden key patterns blocked                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Layer 2: RATE LIMITING                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Redis Sliding Window (primary)                       │    │
│  │ • 100 requests/minute per user                      │    │
│  │ • 5,000 requests/hour per user                      │    │
│  │ • In-memory fallback if Redis is down               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Layer 3: INPUT SANITIZATION                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ HTML escaping, null bytes removed                    │    │
│  │ Control characters filtered                          │    │
│  │ Pattern validation on IDs                            │    │
│  │ Prompt injection defense in system prompts           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Layer 4: PII PROTECTION (4-stage pipeline)                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Stage 1: Presidio NER detection                      │    │
│  │ Stage 2: spaCy medical NER                           │    │
│  │ Stage 3: Regex patterns (SSN, phone, email, etc.)    │    │
│  │ Stage 4: Custom medical rules                        │    │
│  │ + 500-term medical whitelist (avoids false positives) │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Layer 5: ENCRYPTION AT REST                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AES-256-GCM for Protected Health Information        │    │
│  │ PBKDF2 key derivation (100K iterations)             │    │
│  │ encrypt_dict()/decrypt_dict() for sensitive fields  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Layer 6: AUDIT LOGGING                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ PII-scrubbed JSON event logs                        │    │
│  │ HIPAA-compliant audit trail                         │    │
│  │ Preference change tracking (GDPR)                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Password Hashing:** Argon2 via `passlib` (memory-hard, GPU-resistant)

**HIPAA Compliance:**
- Local-only LLM processing (no patient data sent to cloud)
- Encrypted chat metadata
- Audit logging for all data access
- PII scrubbing on all outputs

**GDPR Support:**
- `export_user_data()` — data portability
- `delete_user_data()` — right to be forgotten
- Preference audit trail

---

## LLM Integration

All AI generation flows through a **single gateway**:

```
┌──────────────────────────────────────────────────────────┐
│                    LLM GATEWAY                            │
│              (Single Entry Point)                         │
│                                                          │
│    Any Component ──▶ LLMGateway ──▶ MedGemma-4B-IT      │
│                         │            (localhost:8090)     │
│                         │                                │
│    Features:            │    Safety Pipeline:             │
│    ├─ LangChain         │    ├─ PII detection pre-check  │
│    │  ChatOpenAI        │    ├─ LLM generation           │
│    │  adapter           │    ├─ PII redaction post-check  │
│    ├─ Streaming         │    ├─ Medical disclaimer        │
│    │  support           │    ├─ Hallucination grading     │
│    ├─ Multimodal        │    └─ Audit logging             │
│    │  (text + image)    │                                │
│    ├─ Circuit breaker   │    Token Budget:                │
│    │  protection        │    ├─ 4 chars ≈ 1 token        │
│    └─ Langfuse          │    ├─ 8,192 context window     │
│       tracing           │    └─ Dynamic top_k calc       │
└──────────────────────────────────────────────────────────┘
```

**Token Budget Calculator** pre-allocates context space:
- System prompt budget
- User query budget  
- RAG context budget (remainder)
- Calculates `max_documents` based on available tokens

---

## Dependency Injection

The `DIContainer` is the **factory for all services**:

```
┌──────────────────────────────────────────────────┐
│              DIContainer (Singleton)              │
│                                                  │
│  Lazy-loaded services:                           │
│                                                  │
│  ├─ embedding_service    (MedCPT embeddings)     │
│  ├─ vector_store         (ChromaDB)              │
│  ├─ reranker             (MS-MARCO cross-encoder)│
│  ├─ llm_gateway          (MedGemma)              │
│  ├─ pii_scrubber         (Presidio + spaCy)      │
│  ├─ db_manager           (PostgreSQL asyncpg)    │
│  ├─ redis_client         (Redis)                 │
│  ├─ memory_manager       (Memori)                │
│  ├─ interaction_checker  (Drug interactions)     │
│  ├─ spacy_service        (NLP pipeline)          │
│  └─ ... (feature flags for optional deps)        │
│                                                  │
│  Access: get_di_container()                      │
│  Each service initialized ONLY when first used   │
└──────────────────────────────────────────────────┘
```

---

## Resilience Patterns

### Circuit Breaker

```
┌─────────┐     3 failures     ┌──────┐    30s timeout    ┌───────────┐
│ CLOSED  │───────────────────▶│ OPEN │───────────────────▶│ HALF_OPEN │
│ (Normal)│                    │(Fail)│                    │  (Test)   │
│         │◀───────────────────│      │◀───────────────────│           │
│         │    success reset   │      │    failure retry   │           │
└─────────┘                    └──────┘                    └───────────┘

Per-service configs:  LLM, Tavily, Redis, PostgreSQL
Backend: Redis (Lua scripts for atomic transitions)
```

### Graceful Degradation

```python
@with_fallback(default="I'm having trouble connecting. Please try again.")
async def generate_response(query):
    return await llm_gateway.generate(query)
```

- `@with_fallback` decorator returns a safe default on failure
- `fallback_chain()` tries multiple alternatives with timeout
- `ServiceStatusTracker` counts consecutive failures and applies recovery windows

---

## NLP Services

```
┌──────────────────────────────────────────────────┐
│                NLP PIPELINE                       │
│                                                  │
│  ┌────────────────┐                              │
│  │  SpaCyService  │  Thread-safe singleton       │
│  │                │                              │
│  │  get_entities()    ──▶ Named entities          │
│  │  get_medications() ──▶ Drug names              │
│  │  get_medical_summary() ──▶ Full NLP analysis   │
│  └────────┬───────┘                              │
│           │                                      │
│           ▼                                      │
│  ┌────────────────┐  ┌───────────────────┐       │
│  │ PhraseMatcher  │  │ InteractionDetect │       │
│  │                │  │                   │       │
│  │ Loads terms:   │  │ 10 critical drug  │       │
│  │ • drugs.json   │  │ pairs hardcoded   │       │
│  │ • symptoms.json│  │ (warfarin+aspirin │       │
│  │ • interactions │  │  etc.)            │       │
│  └────────────────┘  │ Fuzzy matching    │       │
│                      └───────────────────┘       │
│                                                  │
│  ┌────────────────────────────────────────┐      │
│  │        Multilingual Support            │      │
│  │                                        │      │
│  │  Languages: en, es, fr, de, pt, zh, ar │      │
│  │  Auto-detection via langdetect         │      │
│  │  Lazy model loading per language       │      │
│  └────────────────────────────────────────┘      │
└──────────────────────────────────────────────────┘
```

---

## Prompt Management

```
┌──────────────────────────────────────────────────┐
│              PROMPT REGISTRY                      │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  system_prompts.py                        │    │
│  │                                          │    │
│  │  30+ prompt templates for:               │    │
│  │  • LLM Gateway (medical/nutrition/etc.)  │    │
│  │  • Supervisor routing & synthesizing     │    │
│  │  • Agent personas (Medical Analyst,      │    │
│  │    Drug Expert, Heart Analyst, etc.)     │    │
│  │  • Multimodal extractors                 │    │
│  │  • SQL Expert, Medical Coding            │    │
│  │  • Memori agents                         │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  PromptRegistry                           │    │
│  │                                          │    │
│  │  • Centralized prompt management         │    │
│  │  • Version hashing for change detection  │    │
│  │  • Injection pattern detection           │    │
│  │  • Audit logging of prompt usage         │    │
│  │  • Hot reload without restart            │    │
│  │  • YAML override system for per-env      │    │
│  │    customization                         │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

---

## Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────┐
│                 OBSERVABILITY STACK                           │
│                                                             │
│  ┌───────────────────────────────────────────────────┐      │
│  │  Prometheus Metrics                                │      │
│  │                                                   │      │
│  │  20+ custom metrics:                              │      │
│  │  • Memory cache hit/miss rates                    │      │
│  │  • Vector search latency                          │      │
│  │  • LLM token usage & latency                      │      │
│  │  • Health check results                           │      │
│  │  • Compression ratios                             │      │
│  │                                                   │      │
│  │  Export: GET /metrics (text format)                │      │
│  └───────────────────────────────────────────────────┘      │
│                                                             │
│  ┌───────────────────────────────────────────────────┐      │
│  │  Agent Tracing                                     │      │
│  │                                                   │      │
│  │  Span types:                                      │      │
│  │  LLM_CALL | TOOL_CALL | RAG_RETRIEVAL | DB_QUERY  │      │
│  │  AGENT_STEP | ROUTING | GUARDRAIL | MEMORY | OTHER │      │
│  │                                                   │      │
│  │  Backends: Langfuse | OpenTelemetry | Local        │      │
│  └───────────────────────────────────────────────────┘      │
│                                                             │
│  ┌───────────────────────────────────────────────────┐      │
│  │  Database Monitoring                               │      │
│  │                                                   │      │
│  │  • Slow query logger (>100ms threshold)           │      │
│  │  • Query performance percentiles (p50/p95/p99)    │      │
│  │  • Connection pool monitoring                     │      │
│  │  • Periodic health checks (PG + Redis)            │      │
│  │  • Query timeout tiers (5s → 300s)                │      │
│  └───────────────────────────────────────────────────┘      │
│                                                             │
│  ┌───────────────────────────────────────────────────┐      │
│  │  Grafana Dashboard + Prometheus Alert Rules        │      │
│  └───────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Supporting Services

### Advanced Caching (3-Tier)

```
Request ──▶ L1 (In-Memory LRU)  ──miss──▶ L2 (Redis + LZ4)  ──miss──▶ L3 (Database)
            │                              │                             │
            │ 1,000-10,000 entries         │ TTL 300s                    │ Permanent
            │ TTL 60s                      │ LZ4 compressed              │
            └──────────────────────────────┴─────────────────────────────┘
```

### WebSocket Manager

- Real-time message delivery with 15s heartbeat keep-alive
- Per-user/per-job subscriptions
- Redis pub/sub for multi-instance coordination

### Job Store

- Redis-based async job tracking (create, progress, complete)
- Priority queues
- 24-hour TTL auto-cleanup

### Webhook Service

- HMAC-signed webhook delivery
- Exponential backoff retry
- Delivery tracking and logging

### User Preferences

- CRUD operations with type-safe serialization
- GDPR export/delete support
- HIPAA audit logging for all changes
- Allergy storage (used by safety guardrails)

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `core/circuit_breaker.py` | 385 | Redis-backed distributed circuit breaker |
| `core/dependencies.py` | 702 | DIContainer — lazy service factory |
| `core/graceful_degradation.py` | 322 | Fallback decorators & service tracking |
| `core/models.py` | 394 | Pydantic v2 request/response models |
| `core/rate_limiter_redis.py` | ~190 | Redis sliding window rate limiter |
| `core/security.py` | 515 | JWT, Argon2, rate limit, audit logger |
| `core/compliance/pii_scrubber_v2.py` | 877 | 4-layer PII detection pipeline |
| `core/config/app_config.py` | 437 | Central AppConfig singleton |
| `core/config/rag_config.py` | 175 | RAG-specific configuration |
| `core/config/rag_paths.py` | 323 | Filesystem path resolution |
| `core/config/rag_settings.py` | — | RAG settings dataclass |
| `core/config/compat.py` | — | Backward compatibility shims |
| `core/config/legacy_settings.py` | — | Legacy settings migration |
| `core/llm/llm_gateway.py` | 458 | Single LLM entry point (MedGemma-4B-IT, selective prompt loading) |
| `core/llm/guardrails.py` | 561 | Safety validation & filtering |
| `core/llm/medgemma_service.py` | ~100 | Direct llama.cpp client |
| `core/llm/token_budget.py` | ~120 | Context window management |
| `core/monitoring/prometheus_metrics.py` | ~310 | Custom Prometheus metrics |
| `core/observability/tracing.py` | ~370 | Span-based agent tracing |
| `core/prompts/system_prompts.py` | 525 | Complete prompt library |
| `core/prompts/registry.py` | 556 | Centralized prompt management (lazy per-category loading) |
| `core/prompts/config_loader.py` | — | YAML/JSON prompt config loader |
| `core/prompts/medical_prompts.py` | — | Medical domain prompt templates |
| `core/prompts/migrate_prompts.py` | — | Prompt migration utility |
| `core/safety/hallucination_grader.py` | ~80 | LLM grounding verification |
| `core/services/encryption_service.py` | ~350 | AES-256-GCM PHI encryption |
| `core/services/chat_history.py` | 948 | Persistent chat with LRU cache |
| `core/services/advanced_cache.py` | 808 | 3-tier cache (L1+L2+L3) |
| `core/services/spacy_service.py` | 314 | Medical NLP pipeline |
| `core/services/multilingual_spacy_service.py` | — | Multilingual NLP support |
| `core/services/websocket_manager.py` | 474 | Real-time WebSocket management |
| `core/services/job_store.py` | 516 | Redis async job store |
| `core/services/webhook_service.py` | 550 | HMAC-signed webhook delivery |
| `core/services/health_service.py` | — | Health check service |
| `core/services/interaction_detector.py` | — | Drug interaction detection service |
| `core/services/medical_phrase_matcher.py` | — | Medical phrase matching |
| `core/services/performance_monitor.py` | — | Performance monitoring |
| `core/user/user_preferences.py` | 712 | GDPR/HIPAA user preferences |

---

*This document describes the Core Architecture of HeartGuard AI v2.1.0*
