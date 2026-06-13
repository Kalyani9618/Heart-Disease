# RAG Architecture Report

## HeartGuard AI — Retrieval-Augmented Generation Pipeline

> **Version:** 2.1.0  
> **Last Updated:** February 2026  
> **Author:** HeartGuard AI Team

---

## Table of Contents

1. [Overview](#overview)
2. [RAG Pipeline Architecture](#rag-pipeline-architecture)
3. [Embedding Layer](#embedding-layer)
4. [Vector Stores](#vector-stores)
5. [Retrieval Strategies](#retrieval-strategies)
6. [NLP Pipeline](#nlp-pipeline)
7. [Trust & Verification](#trust--verification)
8. [Knowledge Graph RAG](#knowledge-graph-rag)
9. [Multimodal RAG](#multimodal-rag)
10. [Memory-RAG Bridge](#memory-rag-bridge)
11. [Colab Integration](#colab-integration)
12. [File Reference](#file-reference)

---

## Overview

**RAG (Retrieval-Augmented Generation)** is how HeartGuard AI grounds its responses in real medical knowledge instead of relying solely on the LLM's training data. When a user asks a medical question, the system:

1. **Retrieves** relevant medical documents from a vector database
2. **Augments** the prompt with those documents as context
3. **Generates** a response using MedGemma with citations

This ensures responses are **factual, up-to-date, and verifiable**.

---

## RAG Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE (End-to-End)                     │
│                                                                  │
│   User Query                                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐                                                │
│  │    QUERY     │  Expand medical synonyms                       │
│  │  PROCESSOR   │  Add related terms                             │
│  │              │  Intent classification                         │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                                │
│  │   INTENT     │  Classify query type ──▶ Select strategy       │
│  │   ROUTER     │                                                │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ├────────────────────────────┐                            │
│         ▼                            ▼                            │
│  ┌──────────────┐          ┌──────────────┐                      │
│  │   EMBED      │          │   HyDE       │  Generate hypothetical│
│  │   (MedCPT)   │          │  Retriever   │  doc, then embed     │
│  │  768-dim     │          │              │  (zero-shot)         │
│  └──────┬───────┘          └──────┬───────┘                      │
│         │                         │                              │
│         └────────────┬────────────┘                              │
│                      │                                           │
│                      ▼                                           │
│  ┌──────────────────────────────────┐                            │
│  │          RETRIEVE                │                            │
│  │                                  │                            │
│  │  ┌──────────┐  ┌──────────────┐  │                            │
│  │  │ ChromaDB │  │   pgvector   │  │                            │
│  │  │ (primary)│  │  (fallback)  │  │                            │
│  │  └──────────┘  └──────────────┘  │                            │
│  │                                  │                            │
│  │  Tiered: Tier1 (StatPearls)      │                            │
│  │          Tier2 (PubMed)          │                            │
│  └──────────────┬───────────────────┘                            │
│                 │                                                │
│                 ▼                                                │
│  ┌──────────────────────────────────┐                            │
│  │          RERANK                  │                            │
│  │                                  │                            │
│  │  CrossEncoderReranker            │                            │
│  │  (MS-MARCO model)               │                            │
│  │  • Bidirectional scoring         │                            │
│  │  • MMR diversity                 │                            │
│  │  • Batch OOM safety              │                            │
│  └──────────────┬───────────────────┘                            │
│                 │                                                │
│                 ▼                                                │
│  ┌──────────────────────────────────┐                            │
│  │       SELF-RAG / CRAG           │                            │
│  │     (Trust Verification)        │                            │
│  │                                  │                            │
│  │  Relevance scoring               │                            │
│  │  Hallucination grading           │                            │
│  │  Web fallback if needed          │                            │
│  └──────────────┬───────────────────┘                            │
│                 │                                                │
│                 ▼                                                │
│  ┌──────────────────────────────────┐                            │
│  │         GENERATE                 │                            │
│  │                                  │                            │
│  │  MedGemma + Retrieved Context    │                            │
│  │  Citation formatting             │                            │
│  │  Safety guardrails               │                            │
│  └──────────────────────────────────┘                            │
│                 │                                                │
│                 ▼                                                │
│            Response with Citations                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Embedding Layer

### Embedding Models

```
┌──────────────────────────────────────────────────────────┐
│               EMBEDDING SERVICE                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  TEXT: MedCPT-Query-Encoder                       │    │
│  │  • Dimensions: 768                                │    │
│  │  • Specialized for medical text                   │    │
│  │  • Query vs Article encoder (asymmetric)          │    │
│  │  • MD5 cache for repeated queries                 │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  IMAGE: SigLIP                                    │    │
│  │  • Dimensions: 1,152                              │    │
│  │  • Medical image embeddings                       │    │
│  │  • Separate collection in ChromaDB                │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Sources:                                                │
│  ├── Remote Colab via ngrok (default)                    │
│  ├── Local GPU (if available)                            │
│  └── CPU fallback                                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  RERANKING: MS-MARCO Cross-Encoder                │    │
│  │  • Bidirectional query-document scoring           │    │
│  │  • MMR (Maximal Marginal Relevance) for diversity │    │
│  │  • Batch processing with OOM protection           │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Vector Stores

### Dual-Store Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  HYBRID STORE                            │
│                                                          │
│  ┌─────────────────────────┐  ┌────────────────────────┐ │
│  │      ChromaDB           │  │      pgvector          │ │
│  │    (Primary Store)      │  │   (Fallback Store)     │ │
│  │                         │  │                        │ │
│  │  medical_text_768       │  │  vector_medical_       │ │
│  │  • 125K+ medical docs   │  │    knowledge           │ │
│  │  • MedCPT embeddings    │  │  • 384-dim             │ │
│  │                         │  │  • HNSW indexes        │ │
│  │  medical_images_1152    │  │  • GIN on metadata     │ │
│  │  • Medical images       │  │                        │ │
│  │  • SigLIP embeddings    │  │  vector_drug_          │ │
│  │                         │  │    interactions         │ │
│  │  Similarity: Cosine     │  │  vector_symptoms_      │ │
│  │                         │  │    conditions           │ │
│  │                         │  │  vector_user_           │ │
│  │                         │  │    memories             │ │
│  └─────────────────────────┘  └────────────────────────┘ │
│                                                          │
│  HybridStore: Tries ChromaDB first, falls back to        │
│  pgvector if ChromaDB is unavailable                     │
└──────────────────────────────────────────────────────────┘
```

---

## Retrieval Strategies

HeartGuard uses **4 retrieval strategies**, selected by the Intent Router:

```
┌──────────────────────────────────────────────────────────┐
│              RETRIEVAL STRATEGIES                         │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  1. TIERED RETRIEVAL                              │    │
│  │                                                  │    │
│  │  Tier 1: StatPearls, textbooks, clinical guides  │    │
│  │          (highest quality, always tried first)    │    │
│  │                                                  │    │
│  │  Tier 2: PubMed research papers                  │    │
│  │          (when Tier 1 is insufficient)            │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  2. HyDE (Hypothetical Document Embeddings)       │    │
│  │                                                  │    │
│  │  Query ──▶ LLM generates hypothetical answer     │    │
│  │         ──▶ Embed the hypothetical answer         │    │
│  │         ──▶ Search with that embedding            │    │
│  │                                                  │    │
│  │  Best for: zero-shot retrieval,                  │    │
│  │  queries with no exact keyword match             │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  3. RAPTOR (Tree-Organized Retrieval)             │    │
│  │                                                  │    │
│  │  Documents organized in a tree hierarchy:        │    │
│  │  Level 0: Raw text chunks                        │    │
│  │  Level 1: Section summaries                      │    │
│  │  Level 2: Document summaries                     │    │
│  │  Level 3: Topic summaries                        │    │
│  │                                                  │    │
│  │  Best for: broad questions needing overview      │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  4. INTENT ROUTER                                 │    │
│  │                                                  │    │
│  │  Classifies query intent → picks best strategy:  │    │
│  │  • Factual question → Tiered Retrieval           │    │
│  │  • Complex question → HyDE                       │    │
│  │  • Overview request → RAPTOR                     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## NLP Pipeline

Medical text requires specialized NLP processing:

```
┌──────────────────────────────────────────────────────────┐
│               MEDICAL NLP PIPELINE                       │
│                                                          │
│  Input Text                                              │
│       │                                                  │
│       ▼                                                  │
│  ┌─────────────────┐                                     │
│  │  Medical         │  Handles: "mg", "q.d.", "b.i.d.", │
│  │  Sentencizer     │  "Dr.", "pt." without splitting    │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │  Medical         │  Custom tokenizer for medical      │
│  │  Tokenizer       │  abbreviations (IV, IM, PO, etc.)  │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │  Entity Ruler    │  Pattern-based NER:                │
│  │  (spaCy)         │  • Drug names                      │
│  │                  │  • Medical conditions               │
│  │                  │  • Anatomy terms                    │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │  Negation        │  NegEx algorithm:                  │
│  │  Detector        │  "no chest pain" → pain NEGATED    │
│  │                  │  "denies fever" → fever NEGATED     │
│  │                  │  Scope-aware detection              │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  Structured medical entities with                        │
│  negation status and context                             │
└──────────────────────────────────────────────────────────┘
```

---

## Trust & Verification

### Self-RAG (Self-Correcting Retrieval)

```
┌──────────────────────────────────────────────────────────┐
│                    SELF-RAG                               │
│                                                          │
│  Query ──▶ Retrieve Documents                            │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────┐                             │
│  │  RELEVANCE SCORING      │  Are these docs relevant?   │
│  │  Score each doc 0-1     │                             │
│  └────────────┬────────────┘                             │
│               │                                          │
│               ▼                                          │
│  ┌─────────────────────────┐                             │
│  │  GENERATE RESPONSE      │  LLM generates answer       │
│  └────────────┬────────────┘                             │
│               │                                          │
│               ▼                                          │
│  ┌─────────────────────────┐                             │
│  │  HALLUCINATION CHECK    │  Is answer grounded in docs?│
│  │  (HallucinationGrader)  │  MD5-cached results (500)   │
│  └────────────┬────────────┘                             │
│               │                                          │
│         ┌─────┴─────┐                                    │
│         │           │                                    │
│     Grounded    Not Grounded                             │
│         │           │                                    │
│         ▼           ▼                                    │
│     Return      Retry with                               │
│     Answer      better docs                              │
└──────────────────────────────────────────────────────────┘
```

### CRAG (Corrective RAG)

```
┌──────────────────────────────────────────────────────────┐
│                      CRAG                                │
│           (Corrective Retrieval-Augmented Generation)     │
│                                                          │
│  Query ──▶ RAG Retrieval                                 │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────┐                             │
│  │  CONFIDENCE CHECK       │                             │
│  │  Is retrieval quality   │                             │
│  │  good enough?           │                             │
│  └────────────┬────────────┘                             │
│               │                                          │
│         ┌─────┴─────┐                                    │
│         │           │                                    │
│     HIGH          LOW                                    │
│  Confidence    Confidence                                │
│         │           │                                    │
│         ▼           ▼                                    │
│   Use RAG       Web Search                               │
│   Results       Fallback                                 │
│         │       (DuckDuckGo,                             │
│         │        medical sites)                          │
│         │           │                                    │
│         └───────────┘                                    │
│               │                                          │
│               ▼                                          │
│         Generate answer                                  │
│         with source tracking                             │
│         (rag vs web vs llm)                              │
└──────────────────────────────────────────────────────────┘
```

---

## Knowledge Graph RAG

```
┌──────────────────────────────────────────────────────────┐
│              GRAPH RAG ENGINE                            │
│                                                          │
│  Query ──▶ Entity Extraction (spaCy + regex)             │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────┐                             │
│  │  GRAPH STORE (NetworkX) │                             │
│  │                         │                             │
│  │  Entities:              │                             │
│  │  [Drug A] ──interacts── [Drug B]                      │
│  │  [Drug]   ──treats──── [Condition]                    │
│  │  [Drug]   ──causes──── [Side Effect]                  │
│  │                         │                             │
│  │  Community Detection:   │                             │
│  │  Louvain clustering     │                             │
│  │  on entity co-occurrence│                             │
│  └────────────┬────────────┘                             │
│               │                                          │
│               ▼                                          │
│  Graph-enhanced retrieval:                               │
│  Query expanded with related entities                    │
│  from the same community cluster                        │
└──────────────────────────────────────────────────────────┘
```

---

## Multimodal RAG

```
┌──────────────────────────────────────────────────────────┐
│            MULTIMODAL RAG ENGINE                         │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐                    │
│  │  Text Query  │    │ Image Upload │                    │
│  └──────┬───────┘    └──────┬───────┘                    │
│         │                   │                            │
│         ▼                   ▼                            │
│  ┌──────────────┐    ┌──────────────┐                    │
│  │  MedCPT      │    │  SigLIP      │                    │
│  │  768-dim     │    │  1,152-dim   │                    │
│  └──────┬───────┘    └──────┬───────┘                    │
│         │                   │                            │
│         ▼                   ▼                            │
│  ┌──────────────┐    ┌──────────────┐                    │
│  │ Text Search  │    │ Image Search │                    │
│  │ (ChromaDB)   │    │ (ChromaDB)   │                    │
│  └──────┬───────┘    └──────┬───────┘                    │
│         │                   │                            │
│         └─────────┬─────────┘                            │
│                   │                                      │
│                   ▼                                      │
│         Combined Context                                 │
│         (text + image results)                           │
│                   │                                      │
│                   ▼                                      │
│         ┌──────────────┐                                 │
│         │  DOCUMENT    │  MinerU + Docling               │
│         │  PARSER      │  PDF/Office parsing             │
│         │              │  Image extraction               │
│         └──────────────┘                                 │
└──────────────────────────────────────────────────────────┘
```

### Document Processing

| Parser | Format | Features |
|--------|--------|----------|
| MinerU | PDF | Text + layout + images |
| Docling | PDF/Office | Structured extraction |
| LibreOffice | Office → PDF | Format conversion |
| ImageProcessor | JPEG/PNG | Resize, format, preprocess |

---

## Memory-RAG Bridge

```
┌──────────────────────────────────────────────────────────┐
│              RAG-MEMORY BRIDGE                           │
│                                                          │
│  ┌─────────────┐         ┌──────────────┐               │
│  │   Memori    │◄───────▶│  RAG Pipeline │               │
│  │  (Patient   │         │              │               │
│  │   Memory)   │         │  Injects:    │               │
│  │             │         │  • Past meds  │               │
│  │  Stores:    │         │  • Allergies  │               │
│  │  • Facts    │         │  • History   │               │
│  │  • History  │         │  • Prefs     │               │
│  │  • Prefs    │         │              │               │
│  └─────────────┘         └──────────────┘               │
│                                                          │
│  ContextManager:                                         │
│  • Conversation context window                           │
│  • Relevance scoring for context selection               │
│  • Balances recency vs importance                        │
└──────────────────────────────────────────────────────────┘
```

---

## Colab Integration

For GPU-constrained setups, embeddings run on Google Colab via ngrok tunnel. This is the **default embedding mode** (`USE_REMOTE_EMBEDDINGS=true`).

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│              REMOTE EMBEDDING ARCHITECTURE                   │
│                                                             │
│  ┌──────────────────────────────┐                           │
│  │  Google Colab (Free GPU)     │                           │
│  │                              │                           │
│  │  colab_server.py (Flask)     │                           │
│  │                              │                           │
│  │  Models (lazy-loaded):       │                           │
│  │  ├── MedCPT (768-dim text)   │  One model on GPU         │
│  │  ├── SigLIP (1152-dim image) │  at a time to fit         │
│  │  └── MedGemma (generation)   │  in 15GB VRAM             │
│  │                              │                           │
│  │  Endpoints:                  │                           │
│  │  POST /embed_text            │                           │
│  │  POST /embed_image           │                           │
│  │  POST /generate              │                           │
│  │  GET  /health                │                           │
│  └──────────────┬───────────────┘                           │
│                 │                                           │
│           ngrok tunnel                                      │
│         (COLAB_API_URL env var)                             │
│                 │                                           │
│                 ▼                                           │
│  ┌──────────────────────────────┐                           │
│  │  Local HeartGuard Server     │                           │
│  │                              │                           │
│  │  RemoteEmbeddingService      │  rag/embedding/remote.py  │
│  │  (singleton, with L1 cache)  │                           │
│  │         │                    │                           │
│  │         ▼                    │                           │
│  │  RemoteColabEmbeddings       │  colab/remote_embeddings  │
│  │  (LangChain-compatible)      │  .py                      │
│  │                              │                           │
│  │  Features:                   │                           │
│  │  ├── Drop-in LangChain embed │                           │
│  │  ├── Auto-retry (3 attempts) │                           │
│  │  ├── 5000-entry local cache  │                           │
│  │  ├── Works with ChromaDB     │                           │
│  │  └── Text + Image support    │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

```env
# Enable remote embeddings (default: true)
USE_REMOTE_EMBEDDINGS=true

# Colab ngrok URL (set when Colab server starts)
COLAB_API_URL=https://your-ngrok-url.ngrok.io
```

### Colab Files

| File | Purpose |
|------|---------|
| `colab/colab_server.py` | Flask server hosting MedCPT + SigLIP + MedGemma |
| `colab/remote_embeddings.py` | LangChain-compatible RemoteColabEmbeddings class |
| `colab/config.py` | Colab configuration |
| `colab/data_ingestion.py` | Document ingestion for Colab |
| `colab/embedding_colab.py` | Colab-side embedding utilities |
| `colab/rag_pipeline.py` | RAG pipeline for Colab |
| `colab/test_remote_embeddings.py` | Connection test script |
| `colab/REMOTE_EMBEDDINGS_GUIDE.md` | Detailed integration guide |
```

---

## RAG Engine Factory

All RAG engines are created through a centralized factory:

```
RAGEngineFactory.create_all_engines()
    │
    ├──▶ BaseRAGEngine (standard text RAG)
    ├──▶ GraphRAGEngine (knowledge graph enhanced)
    ├──▶ MultimodalRAGEngine (text + image)
    └──▶ HeartDiseaseRAG (specialized for cardiac)
```

---

## File Reference

| File | Purpose |
|------|---------|
| **Top-Level** | |
| `rag/__init__.py` | Package exports |
| `rag/rag_engines.py` | RAG engine factory (creates all engine variants) |
| **Embedding** | |
| `rag/embedding/__init__.py` | Factory for RemoteEmbeddingService |
| `rag/embedding/remote.py` | Remote Colab embedding service (singleton, MedCPT + SigLIP) |
| `rag/embedding/base.py` | Base embedding interface |
| **Knowledge Graph** | |
| `rag/knowledge_graph/graph_rag.py` | Graph-enhanced retrieval engine |
| `rag/knowledge_graph/interaction_checker.py` | Drug interaction graph checking |
| `rag/knowledge_graph/medical_ontology.py` | Medical ontology management |
| `rag/knowledge_graph/phonetic_matcher.py` | Phonetic matching for medical terms |
| **Memory** | |
| `rag/memory/memori_integration.py` | Memori ↔ RAG bridge |
| `rag/memory/memori_interfaces.py` | Memory interface contracts |
| **Multimodal** | |
| `rag/multimodal/query.py` | Multimodal RAG query engine (text + image) |
| `rag/multimodal/processors.py` | Medical image preprocessing |
| `rag/multimodal/parser.py` | PDF/Office document parsing (MinerU + Docling) |
| `rag/multimodal/batch_parser.py` | Batch document parsing |
| `rag/multimodal/batch.py` | Batch processing utilities |
| `rag/multimodal/config.py` | Multimodal configuration |
| `rag/multimodal/prompts.py` | Multimodal prompt templates |
| `rag/multimodal/utils.py` | Multimodal utilities |
| `rag/multimodal/zero_shot_classifier.py` | Zero-shot image classification |
| **NLP** | |
| `rag/nlp/factory.py` | NLP pipeline factory / orchestrator |
| `rag/nlp/tokenizer.py` | Medical abbreviation tokenizer |
| `rag/nlp/medical_annotator.py` | Pattern-based medical NER |
| `rag/nlp/negation_detector.py` | NegEx negation detection |
| `rag/nlp/medical_sentencizer.py` | Medical sentence splitting |
| `rag/nlp/symptom_checker.py` | NLP-based symptom checking |
| **Pipeline** | |
| `rag/pipeline/self_rag_medical.py` | Self-correcting retrieval (Self-RAG) |
| `rag/pipeline/crag_fallback.py` | Corrective RAG with web fallback (CRAG) |
| `rag/pipeline/query_optimizer.py` | Query expansion + optimization |
| **Store** | |
| `rag/store/chromadb_store.py` | ChromaDB vector store wrapper |
| `rag/store/vector_store.py` | General vector store (pgvector-compatible) |
| `rag/store/feedback_store.py` | User feedback storage |
| **Trust** | |
| `rag/trust/source_validator.py` | Source credibility validation |
| `rag/trust/conflict_detector.py` | Information conflict detection |
| `rag/trust/explainability.py` | Retrieval explainability |
| **Retrieval** | |
| `rag/retrieval/tiered_retriever.py` | Tier 1/2 source routing |
| `rag/retrieval/fusion_retriever.py` | Reciprocal rank fusion retrieval |
| `rag/retrieval/raptor_retrieval.py` | Tree-organized retrieval (RAPTOR) |
| `rag/retrieval/reranker.py` | MS-MARCO cross-encoder reranking + MMR |
| `rag/retrieval/context_assembler.py` | Context window assembly |
| `rag/retrieval/explainable_retrieval.py` | Explainable retrieval with reasoning |
| `rag/retrieval/token_budget.py` | Token budget management |
| `rag/retrieval/unified_compressor.py` | Document compression |
| `rag/retrieval/models.py` | Retrieval data models |

---

*This document describes the RAG Architecture of HeartGuard AI v2.1.0*
