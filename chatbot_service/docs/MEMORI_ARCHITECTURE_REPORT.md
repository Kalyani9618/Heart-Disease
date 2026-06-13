# Memori Architecture Report

## HeartGuard AI — Memory System (Memori v2.3.0)

> **Version:** 2.3.0  
> **Last Updated:** February 2026  
> **Author:** HeartGuard AI Team

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Memory Storage](#memory-storage)
5. [Search & Retrieval](#search-retrieval)
6. [Caching Architecture](#caching-architecture)
7. [Fact Extraction](#fact-extraction)
8. [Security & Multi-Tenancy](#security-multi-tenancy)
9. [LLM Integrations](#llm-integrations)
10. [File Reference](#file-reference)

---

## Overview

**Memori** is the memory system powering HeartGuard AI. It remembers what patients tell it across conversations — medications, conditions, preferences, and personal context — so the AI doesn't ask the same questions twice and provides better, personalized care.

**Key Idea:** Every conversation is analyzed for important facts. Those facts are stored, indexed, and automatically retrieved in future conversations to give the AI full context about each patient.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       MEMORI v2.3.0                              │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                     API LAYER                             │   │
│  │                                                           │   │
│  │  REST Endpoints:                                          │   │
│  │  POST /v1/memories     — Add new memory                   │   │
│  │  GET  /v1/memories     — Search memories                  │   │
│  │  GET  /v1/memories/:id — Get specific memory              │   │
│  │  PUT  /v1/memories/:id — Update memory                    │   │
│  │  DELETE /v1/memories   — Delete memories                  │   │
│  │  GET  /v1/entities     — List remembered entities         │   │
│  │  POST /v1/memories/:id/feedback — Rate memory accuracy    │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                              │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐   │
│  │                   CORE ENGINE                             │   │
│  │                                                           │   │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐     │   │
│  │  │  Memori    │  │  Memory    │  │  Search          │     │   │
│  │  │  (3240 ln) │  │  Manager   │  │  Service         │     │   │
│  │  │            │  │  (1158 ln) │  │                  │     │   │
│  │  │  Main      │  │            │  │  FTS5 (SQLite)   │     │   │
│  │  │  facade    │  │  Singleton │  │  tsvector (PG)   │     │   │
│  │  │  class     │  │  lifecycle │  │  Semantic search │     │   │
│  │  └────────────┘  └────────────┘  └─────────────────┘     │   │
│  │                                                           │   │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐     │   │
│  │  │  Memory    │  │  Retrieval │  │  Conscious       │     │   │
│  │  │  Agent     │  │  Agent     │  │  Agent           │     │   │
│  │  │            │  │            │  │                  │     │   │
│  │  │  LLM-based │  │  Semantic  │  │  Smart recall    │     │   │
│  │  │  fact      │  │  retrieval │  │  with LLM        │     │   │
│  │  │  extraction│  │  + ranking │  │  relevance       │     │   │
│  │  └────────────┘  └────────────┘  └─────────────────┘     │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                   STORAGE LAYER                           │   │
│  │                                                           │   │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ L1      │  │ L2       │  │ L3       │  │ Vector   │  │   │
│  │  │ In-Mem  │  │ Redis    │  │ Database │  │ Store    │  │   │
│  │  │ LRU     │  │ ZSTD     │  │ PG/SQL   │  │ Chroma   │  │   │
│  │  │ (1000)  │  │ compress │  │ /MySQL   │  │ /PGvec   │  │   │
│  │  └─────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Memori Class (Main Facade)

The central class (~3240 lines) that coordinates all memory operations:

```
┌──────────────────────────────────────────────────┐
│                    Memori                         │
│                                                  │
│  add(messages, user_id)                          │
│    │                                             │
│    ├── 1. Extract facts from messages via LLM    │
│    ├── 2. Deduplicate against existing memories  │
│    ├── 3. Store in database                      │
│    ├── 4. Update search index                    │
│    ├── 5. Cache in L1/L2                         │
│    └── 6. Return stored memory IDs               │
│                                                  │
│  search(query, user_id, limit=100)               │
│    │                                             │
│    ├── 1. Check L1 cache (in-memory)             │
│    ├── 2. Check L2 cache (Redis)                 │
│    ├── 3. Full-text search (FTS5/tsvector)       │
│    ├── 4. Semantic search (embeddings)           │
│    ├── 5. Merge and rank results                 │
│    └── 6. Return top-K memories                  │
│                                                  │
│  get_all(user_id) → all memories for user        │
│  update(memory_id, data) → update memory         │
│  delete(memory_id) → remove memory               │
│  delete_all(user_id) → remove all for user       │
│  history(memory_id) → version history            │
│  reset() → clear everything                      │
└──────────────────────────────────────────────────┘
```

### MemoryManager (Singleton)

```
┌──────────────────────────────────────────────────┐
│                MemoryManager                     │
│                                                  │
│  Singleton that manages Memori lifecycle:        │
│                                                  │
│  get_instance(config)                            │
│    └── Creates Memori if not exists              │
│                                                  │
│  Connection pooling:                             │
│    └── PostgreSQL: asyncpg pool (2-20)           │
│    └── SQLite: aiosqlite                         │
│    └── MySQL: aiomysql                           │
│                                                  │
│  Cleanup:                                        │
│    └── Graceful shutdown with pool drain         │
└──────────────────────────────────────────────────┘
```

---

## Memory Storage

### Database Schema

```
┌────────────────────────────────────────────────────────────┐
│                     MEMORIES TABLE                         │
│                                                            │
│  ┌──────────────────┬───────────────────────────────────┐  │
│  │ Column           │ Description                       │  │
│  ├──────────────────┼───────────────────────────────────┤  │
│  │ id               │ UUID primary key                  │  │
│  │ user_id          │ Patient identifier                │  │
│  │ assistant_id     │ Bot instance                      │  │
│  │ session_id       │ Conversation session              │  │
│  │ content          │ The remembered fact (text)        │  │
│  │ hash             │ SHA256 for deduplication          │  │
│  │ metadata_        │ JSON blob (category, tags, etc.)  │  │
│  │ embedding        │ Vector for semantic search        │  │
│  │ created_at       │ Timestamp                         │  │
│  │ updated_at       │ Last modification                 │  │
│  │ is_deleted       │ Soft delete flag                  │  │
│  └──────────────────┴───────────────────────────────────┘  │
│                                                            │
│  Indexes:                                                  │
│  • idx_memories_user_id                                    │
│  • idx_memories_hash (unique)                              │
│  • idx_memories_created_at                                 │
│  • idx_memories_fts (full-text search)                     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                    HISTORY TABLE                           │
│                                                            │
│  Tracks every change to a memory:                          │
│                                                            │
│  ┌──────────────────┬───────────────────────────────────┐  │
│  │ memory_id        │ Reference to memory               │  │
│  │ old_content      │ Previous text                     │  │
│  │ new_content      │ Updated text                      │  │
│  │ event            │ ADD / UPDATE / DELETE              │  │
│  │ timestamp        │ When change occurred              │  │
│  └──────────────────┴───────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Supported Database Backends

```
┌──────────────┬─────────────────────────────────────────────┐
│ Backend      │ Use Case                                    │
├──────────────┼─────────────────────────────────────────────┤
│ PostgreSQL   │ Production (recommended)                    │
│              │ Full-text: tsvector + gin index             │
│              │ Pool: asyncpg (2-20 connections)            │
├──────────────┼─────────────────────────────────────────────┤
│ SQLite       │ Development / testing                       │
│              │ Full-text: FTS5 virtual table               │
│              │ File: local .db file                        │
├──────────────┼─────────────────────────────────────────────┤
│ MySQL        │ Alternative production                      │
│              │ Full-text: FULLTEXT index                   │
│              │ Pool: aiomysql                              │
└──────────────┴─────────────────────────────────────────────┘
```

---

## Search & Retrieval

```
┌──────────────────────────────────────────────────────────────┐
│                    SEARCH PIPELINE                            │
│                                                              │
│  Query                                                       │
│    │                                                         │
│    ├── 1. CACHE CHECK                                        │
│    │   ├── L1: In-memory LRU (sub-millisecond)               │
│    │   └── L2: Redis (1-5ms)                                 │
│    │                                                         │
│    ├── 2. FULL-TEXT SEARCH                                    │
│    │   ├── PostgreSQL: ts_query + ts_rank                    │
│    │   ├── SQLite: FTS5 MATCH + bm25()                       │
│    │   └── Returns: keyword-matched memories                 │
│    │                                                         │
│    ├── 3. SEMANTIC SEARCH                                    │
│    │   ├── Embed query via MedCPT                            │
│    │   ├── Cosine similarity against stored embeddings       │
│    │   └── Returns: meaning-matched memories                 │
│    │                                                         │
│    ├── 4. MERGE & RANK                                       │
│    │   ├── Combine FTS + semantic results                    │
│    │   ├── Score = FTS_score * 0.4 + semantic_score * 0.6    │
│    │   └── Deduplicate by memory ID                          │
│    │                                                         │
│    └── 5. RESULT                                             │
│        └── Top-K memories with scores                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Caching Architecture

```
                          MULTI-TIER CACHE

        ┌──────────────────────────────────────────┐
        │         L1: In-Memory LRU                │
        │                                          │
        │  • Capacity: 1000 entries                │
        │  • TTL: 5 minutes                        │
        │  • Latency: < 0.1ms                      │
        │  • Hit ratio: ~60%                       │
        │  • Scope: per-process                    │
        └─────────────────┬────────────────────────┘
                          │ miss
                          ▼
        ┌──────────────────────────────────────────┐
        │         L2: Redis Cache                  │
        │                                          │
        │  • Capacity: limited by Redis memory     │
        │  • TTL: 1 hour                           │
        │  • Latency: 1-5ms                        │
        │  • Compression: ZSTD                     │
        │  • Scope: shared across processes        │
        └─────────────────┬────────────────────────┘
                          │ miss
                          ▼
        ┌──────────────────────────────────────────┐
        │         L3: Database                     │
        │                                          │
        │  • Source of truth                        │
        │  • Latency: 5-50ms                       │
        │  • Full-text + semantic index            │
        └──────────────────────────────────────────┘
```

### ZSTD Compression (Redis)

Memories stored in Redis are compressed with ZSTD:

```
Memory text ──▶ ZSTD compress ──▶ Redis SET (bytes)
Redis GET ──▶ ZSTD decompress ──▶ Memory text

Compression ratio: ~4:1 for medical text
```

---

## Fact Extraction

When a user sends a message, the MemoryAgent uses the LLM to extract important facts:

```
┌──────────────────────────────────────────────────────────────┐
│                   FACT EXTRACTION                            │
│                                                              │
│  User message: "I've been taking metformin 500mg twice       │
│  daily for my type 2 diabetes. My blood sugar was            │
│  130 this morning."                                          │
│                                                              │
│       ┌──────────────────────────────────────┐               │
│       │  LLM: Extract facts from message     │               │
│       └──────────────────┬───────────────────┘               │
│                          │                                   │
│                          ▼                                   │
│  Extracted facts:                                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 1. Takes metformin 500mg BID                          │  │
│  │    Category: MEDICATION                               │  │
│  │                                                       │  │
│  │ 2. Has type 2 diabetes                                │  │
│  │    Category: CONDITION                                │  │
│  │                                                       │  │
│  │ 3. Fasting blood sugar: 130 mg/dL                     │  │
│  │    Category: LAB_RESULT                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  12 Medical Fact Categories:                                 │
│  ├── MEDICATION      ├── ALLERGY       ├── VITAL_SIGN       │
│  ├── CONDITION       ├── FAMILY_HISTORY├── LIFESTYLE         │
│  ├── LAB_RESULT      ├── PROCEDURE     ├── PREFERENCE        │
│  ├── SYMPTOM         ├── DEMOGRAPHIC   ├── GOAL              │
└──────────────────────────────────────────────────────────────┘
```

### Deduplication

```
New fact: "Takes metformin 500mg"
   │
   ├── Hash: SHA256("takes metformin 500mg") = abc123...
   │
   ├── Check: Does hash already exist in DB?
   │   ├── Yes ──▶ Skip (duplicate)
   │   └── No  ──▶ Continue
   │
   ├── Semantic check: Similar memory exists?
   │   ├── Similarity > 0.95 ──▶ Update existing
   │   └── Similarity < 0.95 ──▶ Insert new
   │
   └── Store with hash for future dedup
```

---

## Security & Multi-Tenancy

### Multi-Tenant Isolation

```
┌──────────────────────────────────────────────────────────────┐
│                   MULTI-TENANCY                              │
│                                                              │
│  Every memory is scoped by 3 dimensions:                     │
│                                                              │
│  ┌─────────────────────────────────────────────────┐         │
│  │  user_id       │ Patient (e.g., "user_42")      │         │
│  │  assistant_id  │ Bot instance (e.g., "heart_v2")│         │
│  │  session_id    │ Conversation (e.g., "sess_99") │         │
│  └─────────────────────────────────────────────────┘         │
│                                                              │
│  Query: SELECT * FROM memories                               │
│         WHERE user_id = ?                                    │
│         AND assistant_id = ?                                 │
│                                                              │
│  → User A NEVER sees User B's memories                       │
└──────────────────────────────────────────────────────────────┘
```

### Authentication

```
┌──────────────┬───────────────────────────────────────┐
│ Method       │ Description                           │
├──────────────┼───────────────────────────────────────┤
│ JWT          │ Standard token-based auth             │
│ API Key      │ Static key for service-to-service     │
│ NoAuth       │ Development mode (no auth required)   │
└──────────────┴───────────────────────────────────────┘
```

### Input Validation

```
All inputs are checked for:
  ├── SQL injection patterns
  ├── XSS (cross-site scripting) payloads
  ├── Maximum length limits
  └── Content sanitization
```

---

## LLM Integrations

Memori supports multiple LLM providers for fact extraction:

```
┌──────────────────────────────────────────────────────────────┐
│                  LLM PROVIDERS                               │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │  OpenAI          │  │  Anthropic       │                  │
│  │  GPT-4o          │  │  Claude 3.5      │                  │
│  │  GPT-4o-mini     │  │                  │                  │
│  └──────────────────┘  └──────────────────┘                  │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │  LiteLLM         │  │  Local           │                  │
│  │  (universal      │  │  MedGemma        │                  │
│  │  proxy for 100+  │  │  via llama.cpp   │                  │
│  │  models)         │  │  OpenAI-compat   │                  │
│  └──────────────────┘  └──────────────────┘                  │
│                                                              │
│  HeartGuard default: MedGemma-4B-IT via LiteLLM              │
└──────────────────────────────────────────────────────────────┘
```

---

## Memory Agents

### MemoryAgent

```
Analyzes conversation messages and extracts facts:

  Message → LLM prompt → Structured facts → Store

  Prompt: "Extract important medical facts from this 
  conversation. Categorize each fact."
```

### RetrievalAgent

```
Retrieves relevant memories for current context:

  Current query → Search → Rank → Top-K relevant memories

  Used by: ThinkingAgent, Orchestrator for patient context
```

### ConsciousAgent

```
Higher-level memory management:

  • Merges overlapping memories
  • Resolves conflicts (old vs. new info)
  • Prioritizes recent memories
  • Forgets low-relevance old memories
```

---

## Data Flow Example

```
User: "My doctor changed my lisinopril to 20mg"

  1. MemoryAgent extracts:
     → "Lisinopril dosage changed to 20mg"
     → Category: MEDICATION

  2. Deduplication check:
     → Found existing: "Takes lisinopril 10mg"
     → Action: UPDATE (not insert)

  3. History tracked:
     → Event: UPDATE
     → Old: "Takes lisinopril 10mg"
     → New: "Takes lisinopril 20mg"

  4. Cache invalidated:
     → L1: remove stale entry
     → L2: remove stale entry

  5. Next conversation:
     → AI knows current dose is 20mg
     → Drug interactions checked against 20mg
```

---

## File Reference

### Top-Level

| File | Purpose |
|------|---------|
| `memori/__init__.py` | Package exports |
| `memori/memory_manager.py` | Singleton lifecycle manager |
| `memori/memory_middleware.py` | FastAPI middleware for memory |
| `memori/memory_aware_agents.py` | Memory-aware agent integration |
| `memori/memory_observability.py` | Memory system metrics |
| `memori/memory_performance.py` | Performance monitoring |
| `memori/memory_query_optimizer.py` | Query optimization |

### Agents

| File | Purpose |
|------|---------|
| `memori/agents/memory_agent.py` | LLM-based fact extraction |
| `memori/agents/retrieval_agent.py` | Semantic memory retrieval |
| `memori/agents/conscious_agent.py` | Smart memory management |

### Config

| File | Purpose |
|------|---------|
| `memori/config/settings.py` | Configuration (MemoriConfig) |
| `memori/config/manager.py` | Config manager |
| `memori/config/memory_manager.py` | Memory manager config |
| `memori/config/pool_config.py` | Connection pool settings |

### Core

| File | Purpose |
|------|---------|
| `memori/core/conversation.py` | Conversation memory management |
| `memori/core/database.py` | Core database operations |
| `memori/core/memory.py` | Memory CRUD operations |
| `memori/core/providers.py` | Service providers |

### Database

| File | Purpose |
|------|---------|
| `memori/database/models.py` | SQLAlchemy models |
| `memori/database/search_service.py` | Unified search service |
| `memori/database/sqlalchemy_manager.py` | SQLAlchemy session management |
| `memori/database/adapters/postgresql_adapter.py` | PostgreSQL backend |
| `memori/database/connectors/postgres_connector.py` | PostgreSQL connection |
| `memori/database/connectors/base_connector.py` | Base connector interface |
| `memori/database/queries/memory_queries.py` | Memory SQL queries |
| `memori/database/queries/chat_queries.py` | Chat SQL queries |
| `memori/database/queries/base_queries.py` | Base query interface |
| `memori/database/migrations/` | Migration scripts |

### LLM Integrations

| File | Purpose |
|------|---------|
| `memori/integrations/openai_integration.py` | OpenAI integration |
| `memori/integrations/anthropic_integration.py` | Anthropic integration |
| `memori/integrations/litellm_integration.py` | Universal LLM proxy |

### Memory Layers

| File | Purpose |
|------|---------|
| `memori/long_term/fact_extractor.py` | Long-term fact extraction |
| `memori/short_term/redis_buffer.py` | Short-term Redis buffer (L2 cache) |

### Security

| File | Purpose |
|------|---------|
| `memori/security/auth.py` | JWT / API Key / NoAuth |

### Tools

| File | Purpose |
|------|---------|
| `memori/tools/memory_tool.py` | Memory tool for agents |

### Utils

| File | Purpose |
|------|---------|
| `memori/utils/helpers.py` | Helper utilities |
| `memori/utils/validators.py` | Input validation |
| `memori/utils/exceptions.py` | Custom exceptions |
| `memori/utils/pydantic_models.py` | Pydantic schemas |
| `memori/utils/query_builder.py` | SQL query builder |
| `memori/utils/security.py` | Security utilities |
| `memori/utils/security_audit.py` | Security audit logging |
| `memori/utils/security_integration.py` | Security integration |
| `memori/utils/input_validator.py` | Input sanitization |
| `memori/utils/log_sanitizer.py` | Log sanitization |
| `memori/utils/logging.py` | Logging configuration |
| `memori/utils/rate_limiter.py` | Rate limiting |
| `memori/utils/transaction_manager.py` | Transaction management |
| `memori/utils/database/db_helpers.py` | Database helpers |

---

*This document describes the Memori Memory Architecture of HeartGuard AI v2.1.0*
