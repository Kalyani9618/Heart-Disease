# Database Architecture Report

## HeartGuard AI — Database & Storage Layer

> **Version:** 2.1.0  
> **Last Updated:** February 2026  
> **Author:** HeartGuard AI Team

---

## Table of Contents

1. [Overview](#overview)
2. [Storage Stack](#storage-stack)
3. [PostgreSQL Schema](#postgresql-schema)
4. [Migration History](#migration-history)
5. [Connection Management](#connection-management)
6. [Query Optimization](#query-optimization)
7. [Caching Strategy](#caching-strategy)
8. [Vector Storage](#vector-storage)
9. [Database Monitoring](#database-monitoring)
10. [File Reference](#file-reference)

---

## Overview

HeartGuard AI uses a **multi-database architecture** optimized for different workloads. PostgreSQL serves as the primary relational database, Redis handles caching and real-time features, and ChromaDB stores vector embeddings for the RAG pipeline.

---

## Storage Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     STORAGE ARCHITECTURE                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐      │
│  │                 PostgreSQL 15                          │      │
│  │                 (Primary Database)                     │      │
│  │                                                       │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │      │
│  │  │  Users   │ │  Vitals  │ │   Chat   │ │  Meds   │ │      │
│  │  │  Devices │ │  Alerts  │ │ Sessions │ │ Appoint │ │      │
│  │  │  Consent │ │  Goals   │ │ Messages │ │ Predict │ │      │
│  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │      │
│  │                                                       │      │
│  │  Features: RLS, pgvector, HNSW indexes,               │      │
│  │           partitioned tables, materialized views       │      │
│  │  Pool: asyncpg (10-30 connections)                    │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  │
│  │    Redis 7       │  │   ChromaDB       │  │
│  │  (Cache/Queue)   │  │ (Vector Store)   │  │
│  │                  │  │                  │  │
│  │ • Rate limiting  │  │ • 125K+ docs     │  │
│  │ • Session cache  │  │ • MedCPT 768-dim │  │
│  │ • ARQ job queue  │  │ • SigLIP 1152-dim│  │
│  │ • Circuit breaker│  │ • Cosine search  │  │
│  │ • Token blocklist│  │                  │  │
│  │ • LZ4 compressed │  │                  │  │
│  │ • 512MB max      │  │                  │  │
│  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## PostgreSQL Schema

### Entity Relationship Overview

```
┌──────────┐       ┌──────────────┐       ┌──────────────┐
│  users   │──────┐│ chat_sessions│       │  providers   │
│          │      ││              │       │              │
│ id (PK)  │      ││ id (PK)     │       │ id (PK)      │
│ email    │      ││ user_id (FK)│       │ name         │
│ name     │      ││ title       │       │ specialty    │
│ password │      ││ created_at  │       │ rating       │
│ phone    │      │└──────┬───────┘       └──────┬───────┘
│ avatar   │      │       │                      │
└────┬─────┘      │       ▼                      ▼
     │            │┌──────────────┐       ┌──────────────┐
     │            ││chat_messages │       │ appointments │
     │            ││              │       │              │
     │            ││ id (PK)     │       │ id (PK)      │
     │            ││ session_id  │       │ user_id (FK) │
     │            ││ role        │       │ provider_id  │
     │            ││ content     │       │ date/time    │
     │            ││ metadata    │       │ status       │
     │            │└──────────────┘       └──────────────┘
     │            │
     ├────────────┼───────────────────────────────┐
     │            │                               │
     ▼            ▼                               ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐
│  vitals  │ │ medica-  │ │  health  │ │  user_devices  │
│          │ │  tions   │ │  alerts  │ │                │
│ id (PK)  │ │          │ │          │ │ id (PK)        │
│ user_id  │ │ id (PK)  │ │ id (PK)  │ │ user_id (FK)   │
│ metric   │ │ user_id  │ │ user_id  │ │ device_type    │
│ value    │ │ name     │ │ type     │ │ platform       │
│ recorded │ │ dosage   │ │ severity │ │ push_token     │
│ source   │ │ schedule │ │ resolved │ │ is_active      │
└──────────┘ └──────────┘ └──────────┘ └────────────────┘
     │                                        │
     │                                        ▼
     │                               ┌────────────────┐
     │                               │device_timeseries│
     │                               │                │
     │                               │ device_id (FK) │
     │                               │ metric_type    │
     │                               │ value          │
     │                               │ timestamp      │
     │                               └────────────────┘
     │
     ├──────────────────────────────────┐
     │                                  │
     ▼                                  ▼
┌──────────────────┐          ┌──────────────────┐
│ user_health_goals│          │ patient_memories │
│                  │          │                  │
│ id (PK)         │          │ id (PK)          │
│ user_id (FK)    │          │ user_id          │
│ goal_type       │          │ content          │
│ target_value    │          │ category         │
│ current_value   │          │ importance       │
│ status          │          │ tsvector (FTS)   │
└────────┬─────────┘          └──────────────────┘
         │
         ▼
┌──────────────────┐
│ health_goal_     │
│ milestones       │
│ + progress       │
└──────────────────┘
```

### Core Tables (27+ tables)

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `users` | User accounts | Auth, profile, avatar |
| `vitals` | Health measurements | RLS enabled, timestamped |
| `medications` | Drug tracking | Schedule (JSONB), interactions |
| `chat_sessions` | Conversations | Archive support |
| `chat_messages` | Messages | Metadata (JSONB) |
| `health_alerts` | Health warnings | Severity levels |
| `user_devices` | IoT devices | Push notification tokens |
| `device_timeseries` | Sensor data | Time-series metrics |
| `patient_memories` | AI memory | Full-text search (GIN index) |
| `user_health_goals` | Goal tracking | Milestones + progress |
| `appointments` | Doctor visits | 30+ columns, insurance |
| `providers` | Doctors | Seeded with 5 providers |
| `insurance_info` | Insurance | Policy data |
| `user_consents` | GDPR/HIPAA | Consent tracking |
| `prediction_history` | ML predictions | Heart risk results |
| `drug_interactions` | Drug pairs | Severity, mechanism |
| `user_preferences` | Settings | Composite PK |
| `session_archives` | Archived chats | SHA-256 checksum |
| `emergency_contacts` | Emergency info | Relationship, phone |

### Security Features

- **Row-Level Security (RLS)** on `vitals` and `medications` — users can only see their own data
- **`updated_at` trigger** — automatic timestamp on all table updates
- **GIN indexes** on `patient_memories` tsvector for fast full-text search

---

## Migration History

```
┌────────────────────────────────────────────────────────────────┐
│                    ALEMBIC MIGRATION CHAIN                      │
│                                                                │
│  7997d76f03d0 ──▶ Baseline                                     │
│       │           (vitals indexes)                              │
│       ▼                                                        │
│  20260110 ──────▶ SQLite → PostgreSQL Migration                 │
│       │           (drug_interactions, user_preferences,         │
│       │            chat_history tables)                         │
│       ▼                                                        │
│  20260119 ──────▶ Session Archives                              │
│       │           (HIPAA-compliant long-term storage)           │
│       ▼                                                        │
│  20260124 ──────▶ pgvector Tables                               │
│       │           (vector_medical_knowledge,                    │
│       │            vector_drug_interactions,                    │
│       │            vector_symptoms_conditions,                  │
│       │            vector_user_memories — 384-dim HNSW)         │
│       ▼                                                        │
│  20260221 ──────▶ Appointment Tables                            │
│       │           (providers, availability, appointments,       │
│       │            insurance_info + 5 seed doctors)             │
│       ▼                                                        │
│  20260221 ──────▶ Missing Tables                                │
│       │           (consents, calendar, push_devices,            │
│       │            content_verifications, predictions)          │
│       ▼                                                        │
│  20260222 ──────▶ Performance Indexes                           │
│                   (9 composite indexes on hot-path queries)    │
│                                                                │
│  + rag_memory_optimization.sql (raw SQL)                       │
│    HNSW recreation, GIN on JSONB, materialized views,          │
│    hash-partitioned tables, pg_cron scheduling                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Connection Management

### PostgreSQL (asyncpg)

```
┌──────────────────────────────────────────┐
│         Connection Pool (asyncpg)        │
│                                          │
│  Min connections:  2                     │
│  Max connections: 10 (lifespan)          │
│                   30 (PostgresDatabase)  │
│                                          │
│  Features:                               │
│  • %s → $1 placeholder conversion        │
│  • Auto-reconnection                     │
│  • Pool health monitoring                │
│  • Graceful shutdown on app exit         │
│                                          │
│  Schema auto-applied on first start      │
│  (core_postgresql_schema.sql)            │
└──────────────────────────────────────────┘
```

### Redis (redis.asyncio)

```
┌──────────────────────────────────────────┐
│           Redis 7 Connection             │
│                                          │
│  Pool: Shared connection pool            │
│  Max memory: 256MB (LRU eviction)        │
│  Persistence: AOF (append-only file)     │
│                                          │
│  Used for:                               │
│  ├── Rate limiting (sorted sets)         │
│  ├── ARQ job queue                       │
│  ├── Circuit breaker state               │
│  ├── Session cache                       │
│  ├── LangGraph checkpoints              │
│  ├── Token blocklist                     │
│  └── L2 cache tier (LZ4 compressed)     │
└──────────────────────────────────────────┘
```

---

## Query Optimization

```
┌──────────────────────────────────────────────────────────┐
│                QUERY OPTIMIZATION                        │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  TieredCache                                       │   │
│  │                                                   │   │
│  │  L1: In-memory LRU ──▶ <10ms response             │   │
│  │  L2: Redis + LZ4   ──▶ ~1-5ms response            │   │
│  │                                                   │   │
│  │  Target: p95 < 10ms for cached queries            │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  BatchInsertManager                                │   │
│  │                                                   │   │
│  │  Batches individual INSERTs into multi-row         │   │
│  │  statements for bulk data loading                  │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  Prepared Statements                               │   │
│  │                                                   │   │
│  │  Pre-compiled SQL for frequent queries             │   │
│  └───────────────────────────────────────────────────┘   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  Performance Indexes (9 composites)                │   │
│  │                                                   │   │
│  │  vitals       (user_id, metric, recorded_at DESC) │   │
│  │  appointments (user_id, status, date)             │   │
│  │  chat_sessions (user_id, updated_at DESC)         │   │
│  │  health_alerts (user_id, resolved = false)        │   │
│  │  notifications (partial: WHERE pending)           │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Query Timeout Tiers

| Tier | Timeout | Use Case |
|------|---------|----------|
| Fast | 5s | Simple lookups |
| Normal | 30s | Standard queries |
| Slow | 60s | Complex joins |
| Batch | 300s | Data migrations |

---

## Caching Strategy

```
┌────────────────────────────────────────────────────────────┐
│                   3-TIER CACHE ARCHITECTURE                 │
│                                                            │
│  Request ──▶ ┌─────────────────────────────────────────┐   │
│              │  L1: In-Memory LRU                      │   │
│              │  Capacity: 1,000 - 10,000 entries       │   │
│              │  TTL: 60 seconds                        │   │
│              │  Latency: < 0.1ms                       │   │
│              └────────────┬────────────────────────────┘   │
│                    MISS   │                                │
│                           ▼                                │
│              ┌─────────────────────────────────────────┐   │
│              │  L2: Redis                              │   │
│              │  Compression: LZ4                       │   │
│              │  TTL: 300 seconds                       │   │
│              │  Latency: 1 - 5ms                       │   │
│              │  Connection pool: 10                    │   │
│              └────────────┬────────────────────────────┘   │
│                    MISS   │                                │
│                           ▼                                │
│              ┌─────────────────────────────────────────┐   │
│              │  L3: PostgreSQL Database                 │   │
│              │  Permanent storage                      │   │
│              │  Latency: 5 - 50ms                      │   │
│              └─────────────────────────────────────────┘   │
│                                                            │
│  CacheStatistics tracks: hit rate, avg latency,            │
│  eviction count per tier                                   │
└────────────────────────────────────────────────────────────┘
```

---

## Vector Storage

### ChromaDB Collections

```
┌──────────────────────────────────────────────────────────┐
│                     CHROMADB                              │
│                                                          │
│  Collection: medical_text_768                            │
│  ├── Embedding model: MedCPT-Query-Encoder               │
│  ├── Dimensions: 768                                     │
│  ├── Documents: 125,000+ medical texts                   │
│  └── Sources: StatPearls, textbooks, guidelines          │
│                                                          │
│  Collection: medical_images_1152                         │
│  ├── Embedding model: SigLIP                             │
│  ├── Dimensions: 1,152                                   │
│  └── Sources: Medical imaging, diagrams                  │
└──────────────────────────────────────────────────────────┘
```

### pgvector Tables

```
┌──────────────────────────────────────────────────────────┐
│                     PGVECTOR                             │
│                                                          │
│  vector_medical_knowledge                                │
│  ├── Dimensions: 384 (all-MiniLM-L6-v2)                 │
│  ├── Index: HNSW (cosine similarity)                     │
│  └── GIN index on metadata JSONB                         │
│                                                          │
│  vector_drug_interactions                                │
│  ├── Same structure                                      │
│  └── Drug pair vectors                                   │
│                                                          │
│  vector_symptoms_conditions                              │
│  ├── Same structure                                      │
│  └── Symptom/condition vectors                           │
│                                                          │
│  vector_user_memories                                    │
│  ├── Same structure                                      │
│  ├── Hash-partitioned (4 partitions)                     │
│  └── Per-user memory vectors                             │
│                                                          │
│  Fallback: TEXT column if pgvector extension unavailable  │
└──────────────────────────────────────────────────────────┘
```

---

## Database Monitoring

```
┌──────────────────────────────────────────────────────────┐
│                DATABASE HEALTH SYSTEM                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  SlowQueryLogger                                   │  │
│  │  Threshold: 100ms                                  │  │
│  │  Logs: query text, duration, caller context        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  QueryPerformanceMonitor                           │  │
│  │  Tracks: p50, p95, p99 latency percentiles         │  │
│  │  Per-query-type statistics                         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  ConnectionPoolMonitor                             │  │
│  │  Active/idle/total connections                     │  │
│  │  Pool utilization percentage                       │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  DatabaseHealthChecker                             │  │
│  │  Periodic checks: PostgreSQL + Redis               │  │
│  │  Graceful degradation on failure                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Materialized Views (auto-refreshed)               │  │
│  │  • mv_vector_collection_stats                      │  │
│  │  • mv_user_memory_stats                            │  │
│  │  • Hourly refresh via pg_cron                      │  │
│  │  • Daily 3AM maintenance (ANALYZE)                 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## ORM Layer

HeartGuard uses **SQLAlchemy 2.0** ORM for structured data access:

### Key ORM Models

| Model | Table | Key Relationships |
|-------|-------|-------------------|
| `User` | `users` | Has many: vitals, medications, devices, sessions |
| `Device` | `user_devices` | Belongs to User; has many timeseries |
| `PatientRecord` | `patient_records` | Belongs to User |
| `Vital` | `vitals` | Belongs to User |
| `ChatSession` | `chat_sessions` | Has many messages; belongs to User |
| `ChatMessage` | `chat_messages` | Belongs to session |
| `Medication` | `medications` | Belongs to User |
| `Provider` | `providers` | Has many appointments |
| `AppointmentRecord` | `appointments` | Belongs to User + Provider |
| `InsuranceInfo` | `insurance_info` | Belongs to User |
| `UserConsent` | `user_consents` | Belongs to User |
| `PredictionHistory` | `prediction_history` | Belongs to User |

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `core_postgresql_schema.sql` | 597 | Full schema (27+ tables, RLS, triggers) |
| `core/database/models.py` | 640 | SQLAlchemy ORM models |
| `core/database/postgres_db.py` | 1,560 | asyncpg connector + CRUD |
| `core/database/query_optimizer.py` | 1,125 | TieredCache, batch inserts, prepared stmts |
| `core/database/query_monitor.py` | 638 | Slow query logging, pool monitoring |
| `core/database/apply_optimizations.py` | 498 | DB optimization scripts |
| `core/database/storage_interface.py` | 305 | Feedback storage ABC |
| `core/database/postgres_feedback_storage.py` | 332 | Feedback PostgreSQL impl |
| `alembic/versions/7997d76f03d0_final_baseline.py` | — | Baseline migration |
| `alembic/versions/20260110_*.py` | — | SQLite → PostgreSQL migration |
| `alembic/versions/20260119_*.py` | — | Session archives |
| `alembic/versions/20260124_*.py` | — | pgvector tables |
| `alembic/versions/20260221_*.py` | — | Appointments + missing tables |
| `alembic/versions/20260222_*.py` | — | Performance indexes |
| `alembic/versions/rag_memory_optimization.sql` | 480 | RAG/memory optimization |

---

*This document describes the Database Architecture of HeartGuard AI v2.1.0*
