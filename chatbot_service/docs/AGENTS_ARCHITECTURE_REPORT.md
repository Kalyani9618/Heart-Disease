# Agents Architecture Report

## HeartGuard AI — Agent System

> **Version:** 2.1.0  
> **Last Updated:** February 2026  
> **Author:** HeartGuard AI Team

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Agent Components](#agent-components)
4. [LangGraph Orchestrator](#langgraph-orchestrator)
5. [Specialist Agents](#specialist-agents)
6. [Thinking & Reasoning](#thinking--reasoning)
7. [Clinical Reasoning](#clinical-reasoning)
8. [Deep Research Agent](#deep-research-agent)
9. [Data Flow](#data-flow)
10. [File Reference](#file-reference)

---

## Overview

The Agent System is the **brain** of HeartGuard AI. It receives user queries, decides which specialist should handle them, and returns medically-grounded responses. The system uses **LangGraph** (a state-machine framework) to wire multiple AI agents into a coordinated workflow.

**Key Idea:** Instead of one big AI model doing everything, we have a team of specialist agents — each expert in one area — managed by a supervisor.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH ORCHESTRATOR                             │
│                                                                     │
│  ┌─────────────┐    ┌───────────────────────────────────────────┐   │
│  │   ROUTER    │    │           SUPERVISOR (MedGemma)            │   │
│  │ (Semantic   │───▶│  Decides which agent handles the query     │   │
│  │  RouterV2)  │    │  or synthesizes the final answer            │   │
│  └─────────────┘    └──────────────────┬────────────────────────┘   │
│                                        │                            │
│       ┌────────────────────────────────┼────────────────────┐       │
│       │                │               │              │     │       │
│       ▼                ▼               ▼              ▼     ▼       │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐  │       │
│  │ Medical  │   │   Drug   │   │  Heart   │   │Thinking │  │       │
│  │ Analyst  │   │  Expert  │   │ Analyst  │   │  Agent  │  │       │
│  │ (RAG)    │   │(Checker) │   │(Predict) │   │ (CoT)   │  │       │
│  └─────────┘   └──────────┘   └──────────┘   └─────────┘  │       │
│       │                │               │              │     │       │
│       ▼                ▼               ▼              ▼     ▼       │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐  │       │
│  │Researcher│   │  Data    │   │ Clinical │   │  FHIR   │  │       │
│  │(Deep     │   │ Analyst  │   │Reasoning │   │  Agent  │  │       │
│  │Research) │   │  (SQL)   │   │(DDx/ESI) │   │ (EHR)   │  │       │
│  └─────────┘   └──────────┘   └──────────┘   └─────────┘  │       │
│       │                │               │              │     │       │
│       └────────────────┴───────────────┴──────────────┘     │       │
│                                │                            │       │
│                      ┌─────────────────┐                    │       │
│                      │ Profile Manager │                    │       │
│                      └─────────────────┘                    │       │
│                                                                     │
│  All workers report back ──▶ SUPERVISOR ──▶ FINISH                  │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│               POST-PROCESSING PIPELINE                              │
│                                                                     │
│   PII Scrub ──▶ Save to PostgreSQL ──▶ Record to Memori            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Components

### How Routing Works

There are **two routing paths**:

```
                    User Query
                        │
                        ▼
                ┌───────────────┐
                │ SemanticRouter │
                │    (Fast)      │
                └───────┬───────┘
                        │
              ┌─────────┴──────────┐
              │                    │
        High Confidence      Low Confidence
         (> threshold)        (ambiguous)
              │                    │
              ▼                    ▼
        ┌──────────┐       ┌─────────────┐
        │ Direct to│       │  Supervisor  │
        │ Worker   │       │  LLM decides │
        └──────────┘       └─────────────┘
```

**Fast Path (SemanticRouter):** Uses embedding similarity to classify intent instantly:
- `VITALS` → Data Analyst
- `DRUG_INTERACTION` → Drug Expert  
- `MEDICAL_QA` → Medical Analyst
- `TRIAGE` → Clinical Reasoning

**Slow Path (Supervisor):** When intent is unclear, the MedGemma LLM reads the query and picks the best worker. The supervisor can also chain multiple workers.

---

## LangGraph Orchestrator

**File:** `agents/langgraph_orchestrator.py` (1,409 lines)

This is the **central brain** that wires everything together.

### State Machine

```
┌──────────────────────────────────────────┐
│              AgentState                  │
│                                          │
│  messages     : list[BaseMessage]        │
│  user_id      : str                      │
│  next         : str  (next worker)       │
│  final_response: str                     │
│  intent       : str                      │
│  confidence   : float                    │
│  citations    : list[Citation]           │
│  source       : str  (rag/web/llm)       │
│  thinking     : str                      │
│  web_search   : list                     │
│  deep_search  : list                     │
│  file_ids     : list                     │
└──────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Crash Recovery** | Redis checkpointing — if the app crashes, it resumes from the last saved state |
| **Parallel Workers** | Independent workers can run simultaneously with timeout protection |
| **Max Steps** | Supervisor loop limited to 8 steps to prevent infinite loops |
| **PII Scrubbing** | All responses are scrubbed for personal health info before returning |
| **Tracing** | Every step is recorded via `AgentTracer` for debugging |

---

## Specialist Agents

### 1. Medical Analyst

```
Query ──▶ MedicalSelfRAG ──▶ Retrieve from ChromaDB
                │                      │
                │               ┌──────┴──────┐
                │               │  Relevant?   │
                │               └──────┬──────┘
                │                Yes   │   No
                │                 │    │    │
                │                 ▼    │    ▼
                │           Use RAG   │  CRAGFallback
                │           Results   │  (web search)
                │                 │    │    │
                │                 └────┴────┘
                │                      │
                ▼                      ▼
          Generate Answer with Citations
```

- Uses **Self-RAG** (self-correcting retrieval) for medical Q&A
- Falls back to **CRAG** (Corrective RAG with web search) when retrieval confidence is low
- Sources: ChromaDB medical guidelines (125K+ documents)

### 2. Drug Expert

```
Query ──▶ DrugInteractionChecker ──▶ PostgreSQL + JSON fallback
                                            │
                                            ▼
                                   Drug Interaction Data
                                   (severity, mechanism,
                                    recommendation)
```

- Uses **PostgreSQL** (primary) with **JSON file fallback** for drug-drug interactions
- Returns severity level, mechanism, and clinical recommendations

### 3. Heart Analyst

```
Patient Data ──▶ HeartDiseasePredictor
                        │
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
        RETRIEVE     MEMORY     AUGMENT
        (ChromaDB)  (Memori)   (Prompt)
            │           │           │
            └───────────┼───────────┘
                        │
                        ▼
                    GENERATE
                    (MedGemma)
                        │
                        ▼
                    VALIDATE
                 (Hallucination
                    Grading)
                        │
                        ▼
                HeartRiskResult
                  - risk_level
                  - confidence
                  - contributing_factors
                  - citations
```

- 5-step pipeline: Retrieve → Memory → Augment → Generate → Validate
- Returns risk level (Low/Moderate/High), contributing factors, and citations

### 4. Data Analyst

- Executes **SQL queries** against PostgreSQL for vitals and patient data
- Example: "Show my blood pressure from last week"

### 5. Researcher

```
Query ──▶ MedicalContentSearcher ──▶ Papers, News, Images, Videos
                │                         (PubMed, WHO, FDA, CDC)
                ▼
        ReasoningResearcher ──▶ Deep Analysis
                │
                ▼
         Structured Report
```

- Two-phase: broad medical search → deep reasoning analysis
- Sources: PubMed, WHO, FDA, CDC, NIH

### 6. FHIR Agent

- Queries **FHIR R4** endpoints for Electronic Health Records (EHR)
- Retrieves patient demographics, observations, medications

### 7. Profile Manager

- Manages user profiles and preferences
- Accesses stored patient information

---

## Thinking & Reasoning

### ThinkingAgent (Chain-of-Thought)

**File:** `agents/components/thinking.py`

Inspired by **DeepSeek-R1**, this agent reasons step-by-step before acting.

```
Query ──▶ ThinkingAgent
              │
              ▼
         ┌─────────┐
         │ <think>  │  ◄── Explicit reasoning
         │ block    │      (analysis, planning,
         │          │       evaluation, reflection)
         └────┬────┘
              │
              ▼
         ┌──────────┐
         │<tool_call>│  ◄── Execute available tools
         │ block     │      (search, calculate, etc.)
         └────┬─────┘
              │
              ▼
         ┌─────────┐
         │ <answer> │  ◄── Final response
         │ block    │
         └─────────┘
```

- **Max 3 rounds** of thinking before forced answer
- **Confidence-based early exit:** If confidence > 0.85, stops early
- Dynamic tool execution (supports any callable tool)

### MedicalPlanner

**File:** `agents/components/medical_planner.py`

Generates step-by-step medical plans before executing:
1. Creates initial plan from the question
2. Updates plan based on new context from past steps

### PlanningMixin

**File:** `agents/components/planning.py`

Multi-step task planner that generates 3-7 steps with:
- Step dependencies (step 3 depends on step 1)
- Agent assignment per step
- Auto-replanning when context changes

---

## Clinical Reasoning

### Differential Diagnosis Engine

**File:** `agents/components/differential_diagnosis.py` (583 lines)

```
Patient Presentation
        │
        ▼
┌──────────────────┐
│ 1. Categorize    │  ◄── chest pain, SOB, headache, abdominal
│    Complaint     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. Identify      │  ◄── Per-category red flags
│    Red Flags     │      (e.g., "tearing pain" → dissection)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Initial       │  ◄── SYMPTOM_CONDITION_MAP lookup
│    Differentials │      (7 conditions for chest pain, etc.)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Apply Context │  ◄── Age/sex/symptom multipliers
│    (Bayesian)    │      (age>50 + ACS → ×1.5)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. LLM Reasoning │  ◄── MedGemma probability refinement
│    Adjustment    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. Rank & Output │  ◄── Ranked differentials + ICD-10 codes
│                  │      + cannot-miss diagnoses
└──────────────────┘
```

**Probability Levels:**
- `HIGHLY_LIKELY` (>70%)
- `LIKELY` (40-70%)
- `POSSIBLE` (15-40%)
- `UNLIKELY` (5-15%)
- `RULED_OUT` (<5%)

### Triage System (ESI)

**File:** `agents/components/triage_system.py`

Emergency Severity Index (ESI 1-5) triage assessment:

```
┌─────────────────────────────────────────┐
│          ESI Triage Algorithm            │
│                                         │
│  Step 1: Life-saving needed?            │
│          YES ──▶ ESI-1 (Immediate)      │
│                                         │
│  Step 2: High risk situation?           │
│          YES ──▶ ESI-2 (Emergent)       │
│                                         │
│  Step 3: How many resources needed?     │
│          ≥2  ──▶ ESI-3 (Urgent)         │
│          1   ──▶ ESI-4 (Less Urgent)    │
│          0   ──▶ ESI-5 (Non-Urgent)     │
└─────────────────────────────────────────┘
```

| ESI Level | Category | Example |
|-----------|----------|---------|
| ESI-1 | Immediate | Cardiac arrest, unresponsive |
| ESI-2 | Emergent | Chest pain, stroke symptoms |
| ESI-3 | Urgent | Abdominal pain needing workup |
| ESI-4 | Less Urgent | Sprained ankle |
| ESI-5 | Non-Urgent | Cold symptoms |

---

## Deep Research Agent

**Directory:** `agents/deep_research_agent/`

Three research strategies for in-depth medical questions:

```
┌──────────────────────────────────────────────────┐
│              RESEARCH MODES                       │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  LINEAR  │  │   COT    │  │     AOT      │   │
│  │          │  │ (Chain   │  │ (Atom-of-    │   │
│  │ Search → │  │  of      │  │  Thought)    │   │
│  │ Crawl →  │  │ Thought) │  │              │   │
│  │ Report   │  │          │  │ Decompose →  │   │
│  │          │  │ Reason → │  │ Execute →    │   │
│  │          │  │ Search → │  │ Replan →     │   │
│  │          │  │ Analyze  │  │ Synthesize   │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
│                                                  │
│  Search Sources:                                 │
│  • DuckDuckGo (web search)                       │
│  • PubMed (medical papers)                       │
│  • WHO, FDA, CDC, NIH (verified sources)         │
│                                                  │
│  Crawling:                                       │
│  • crawl4ai (headless browser)                   │
│  • LLM-based content extraction                  │
│  • Screenshot capture                            │
│  • Max 3 concurrent crawlers (~300MB each)       │
└──────────────────────────────────────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Search Tool | `search_tool.py` | DuckDuckGo + PubMed + Medical search |
| Crawler Tool | `crawler_tool.py` | Web page crawling with LLM extraction |
| Reasoning Researcher | `reasoning_researcher.py` | Chain-of-Thought research (max 4 rounds) |
| Atomic Researcher | `atomic_researcher.py` | Decompose query → atomic tasks → synthesize |
| Reporter | `reporter.py` | Synthesize insights into markdown report |

---

## Data Flow

Complete flow from user query to response:

```
1. ENTRY
   User sends query ──▶ LangGraphOrchestrator.execute(query, user_id)

2. ROUTING
   SemanticRouterV2 classifies intent via embeddings
   ├── High confidence ──▶ Direct to specialist worker
   └── Low confidence  ──▶ Supervisor LLM decides

3. WORKER EXECUTION
   Specialist agent processes the query
   └── Produces: content + citations + confidence + source type

4. SUPERVISION LOOP (max 8 steps)
   Supervisor reviews worker output
   ├── Need more info ──▶ Delegate to another worker
   └── Ready ──▶ Synthesize final answer (FINISH)

5. POST-PROCESSING
   ├── PII scrubbing (fail-secure: blocks if scrubbing fails)
   ├── Chat saved to PostgreSQL
   ├── Conversation recorded to Memori (long-term memory)
   └── Agent execution metrics recorded via AgentTracer

6. RESPONSE
   Returns: response, intent, confidence, citations,
            pii_scrubbed flag, thread_id, metadata
```

---

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `agents/__init__.py` | 90 | Package exports |
| `agents/langgraph_orchestrator.py` | 1,710 | Central orchestrator (LangGraph StateGraph) |
| `agents/heart_predictor.py` | 427 | Heart disease risk prediction |
| `agents/evaluation.py` | 511 | LLM-as-judge response evaluator |
| `agents/components/thinking.py` | ~470 | Chain-of-Thought reasoning engine |
| `agents/components/vision.py` | ~330 | Medical image analysis |
| `agents/components/planning.py` | ~330 | Multi-step task planning |
| `agents/components/managed.py` | ~370 | Sub-agent registry & delegation |
| `agents/components/medical_planner.py` | ~90 | Medical-domain planner |
| `agents/components/differential_diagnosis.py` | 583 | Bayesian differential diagnosis |
| `agents/components/triage_system.py` | ~340 | ESI triage system |
| `agents/components/workflow_automation.py` | ~260 | Finding routing to specialists |
| `agents/utils/visualization.py` | 358 | Mermaid diagram generator |
| `agents/deep_research_agent/models.py` | 16 | Research data models |
| `agents/deep_research_agent/search_tool.py` | ~190 | Web + medical search adapters |
| `agents/deep_research_agent/crawler_tool.py` | ~300 | Web crawling + LLM extraction |
| `agents/deep_research_agent/deep_research.py` | ~180 | Research mode dispatcher |
| `agents/deep_research_agent/reasoning_researcher.py` | 502 | Chain-of-Thought research |
| `agents/deep_research_agent/atomic_researcher.py` | ~380 | Atom-of-Thought research |
| `agents/deep_research_agent/reporter.py` | ~80 | Report synthesizer |

---

## External Services Used

| Service | Purpose |
|---------|---------|
| **MedGemma-4B-IT** (port 8090) | LLM for reasoning, routing, generation (local via llama.cpp) |
| **ChromaDB** | Vector store for medical documents |
| **PostgreSQL** | Patient data, chat history |
| **Redis** | Crash recovery checkpoints |
| **FHIR Endpoints** | Electronic Health Records |
| **DuckDuckGo** | Web search |
| **PubMed/WHO/CDC/NIH** | Medical research sources |
| **crawl4ai** | Headless browser crawling |
| **Memori** | Long-term patient memory |

---

*This document describes the Agent Architecture of HeartGuard AI v2.1.0*
