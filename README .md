# FitnessCoach AI — Multi-Agent Fitness Coaching System

> **inmind.academy — Generative AI Track, Spring 2026**
> A multi-agent system that provides personalized fitness coaching through intelligent routing, RAG-powered nutrition guidance, and deterministic progress analysis.

---

## What It Does

FitnessCoach AI is a conversational fitness assistant that routes user queries to specialized AI agents:

- **Exercise Agent** — Retrieves exercises from a structured database and generates detailed coaching responses with form tips, sets/reps recommendations, and progression advice.
- **Nutrition Agent (RAG)** — Uses Retrieval-Augmented Generation over a curated nutrition knowledge base to answer diet, macro, and supplement questions with cited sources.
- **Program Builder Agent** — Constructs personalized multi-week workout programs by matching templates to user goals, experience level, and available equipment.
- **Progress Agent** — Analyzes workout logs using the Epley 1RM formula, detects plateaus, tracks volume trends, and delivers coaching narratives.
- **MCP Server** — Provides deterministic body metrics calculations (TDEE, macros, goal timeline) via a Model Context Protocol server.

---

## Who It Is For

- Fitness enthusiasts who want personalized guidance without a personal trainer
- Developers exploring multi-agent LLM system design
- Students and researchers studying RAG pipelines, agent routing, and evaluation methodology

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User / Chat UI                           │
│                    (http://localhost:8000)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /chat
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT SYSTEM A  :8000                        │
│                  (LangGraph Supervisor)                         │
│                                                                 │
│   ┌─────────────┐   Input Guardrails (blocked patterns)         │
│   │  Supervisor │   Output Guardrails (sanitize + safety notes) │
│   │   Router    │   MAX_ITERATIONS = 5 (loop cap)               │
│   └──────┬──────┘   Request timeouts on all HTTP calls          │
│          │                                                      │
│    ┌─────┴──────────────────────────┐                           │
│    ▼          ▼          ▼          ▼                           │
│ Exercise  Program    Progress    Nutrition ──────────────────┐  │
│  Agent    Builder     Agent       Client                     │  │
│ (JSON     (Template   (Epley      (HTTP)                     │  │
│  filter)   matching)   1RM)                                  │  │
└──────────────────────────────────────────┬───────────────────┘  │
                                           │                      │
          ┌────────────────────────────────┘                      │
          ▼                                                       │
┌─────────────────────────┐    ┌──────────────────────────────┐   │
│   AGENT SYSTEM B :8001  │    │     MCP SERVER  :8002        │   │
│   (LlamaIndex RAG)      │    │   (FastMCP Tools)            │   │
│                         │    │                              │   │
│  Qdrant Vector DB       │    │  • calculate_tdee            │   │
│  all-MiniLM-L6-v2       │    │  • calculate_macros          │   │
│  22 nutrition chunks    │    │  • estimate_goal_timeline    │   │
│  qwen2.5:1.5b@Ollama    │    │                              │   │
└─────────────────────────┘    └──────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                               │
│                                                                 │
│  Ollama :11434          Redis :6379         Qdrant :6333        │
│  qwen2.5:1.5b           LangGraph           nutrition_guides    │
│  GPU-accelerated        checkpointer        collection          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | LangGraph (Agent A) |
| RAG Framework | LlamaIndex (Agent B) |
| LLM | qwen2.5:1.5b via Ollama |
| Vector Database | Qdrant |
| Embedding Model | all-MiniLM-L6-v2 |
| Session Persistence | Redis (LangGraph checkpointer) |
| MCP Tools | FastMCP |
| API Framework | FastAPI |
| Containerization | Docker Compose |

---

## Quick Start

### Prerequisites

- Docker Desktop with GPU support (NVIDIA)
- Git

### Run with one command

```bash
git clone <your-repo-url>
cd Final_Project_Christina
docker-compose up
```

That's it. Docker Compose will:
1. Pull all required images (Ollama, Qdrant, Redis)
2. Build all services (Agent A, Agent B, MCP Server)
3. Download `qwen2.5:1.5b` model automatically (~986MB, first run only)
4. Ingest nutrition data into Qdrant automatically
5. Start all services

> **First run note:** Allow 3-5 minutes for the Ollama model to download. Subsequent starts are instant since model and data are persisted in Docker volumes.

### Access the system

| Service | URL |
|---------|-----|
| Chat UI | Open `chat_ui.html` in your browser |
| Agent A API | http://localhost:8000 |
| Agent B API | http://localhost:8001 |
| Swagger UI | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### Environment Variables

Copy `.env.example` to `.env` and adjust if needed:

```bash
cp .env.example .env
```

Default values work out of the box with Docker Compose.

---

## API Usage

### Chat endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "give me 3 chest exercises for a beginner", "session_id": "user1"}'
```

### Streaming endpoint

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "how much protein should I eat to build muscle", "session_id": "user1"}'
```

### Example queries by route

```bash
# Exercise agent
{"message": "give me beginner dumbbell chest exercises"}

# Nutrition agent (RAG)
{"message": "how much protein should I eat to build muscle"}

# Program builder
{"message": "build me a 4 week beginner strength program 3 days a week"}

# Progress analysis
{"message": "analyze my progress: bench press 80kg 8 reps 3 sets on 2026-03-01, bench press 82kg 9 reps 3 sets on 2026-03-08, bench press 84kg 9 reps 3 sets on 2026-03-15"}

# MCP body metrics
{"message": "calculate my TDEE", "mcp_tdee": {"age": 25, "sex": "female", "weight_kg": 65, "height_cm": 165, "activity_level": "moderate"}}
```

---

## Project Structure

```
Final_Project_Christina/
├── agent_system_a/          # LangGraph supervisor + specialist agents
│   ├── agents/              # exercise, program builder agents
│   ├── app/                 # FastAPI app, graph, supervisor, guardrails
│   ├── llm/                 # Ollama client
│   ├── tools/               # exercise tools, program tools, progress tool, MCP client
│   └── Dockerfile
├── agent_system_b/          # LlamaIndex RAG nutrition agent
│   ├── app/                 # FastAPI app, LlamaIndex agent
│   └── Dockerfile
├── mcp_server/              # FastMCP body metrics server
│   ├── app/                 # calculators, server
│   └── Dockerfile
├── data/                    # exercises.json, workout_programs.json, nutrition_guides.json
├── evaluation/              # test set, retrieval metrics, RAGAS eval scripts
├── chat_ui.html             # Browser-based chat interface
├── ingest_nutrition.py      # One-time Qdrant ingestion script
├── ollama-init.sh           # Auto-pulls qwen2.5:1.5b on startup
├── docker-compose.yml
├── .env.example
├── README.md
└── EVALUATION.md
```

---

## Technical Decisions & Justifications

### Why LangGraph for Agent A?
LangGraph provides a stateful graph with checkpointing — essential for multi-agent workflows where we need to track which specialists have run, prevent infinite loops (`MAX_ITERATIONS = 5`), and persist state across requests via Redis. A simple function-calling loop would lack these production-grade guarantees.

### Why LlamaIndex for Agent B?
Agent B is a dedicated RAG service. LlamaIndex is purpose-built for retrieval workflows — it handles document indexing, embedding, Qdrant integration, and query synthesis in a clean abstraction. Using CrewAI (our original choice) would have added 1.5GB of dependencies for a task LlamaIndex handles in ~300MB.

### Why separate Agent A and Agent B?
This is the architectural point of the project. Agent B is not a function inside Agent A — it is a fully independent FastAPI service on its own port with its own model client, vector DB connection, and RAG pipeline. Agent A calls it over HTTP, meaning they can be scaled, deployed, and updated independently.

### Why qwen2.5:1.5b?
Our deployment targets a consumer GPU (NVIDIA RTX 4050, 6GB VRAM). The 7B model (4.7GB) left insufficient headroom for inference overhead and caused OOM errors. The 1.5B model (986MB) fits comfortably and runs at acceptable latency (~5-8s per response). This is a deliberate hardware-constrained decision, documented as a known limitation.

### Why Qdrant for vector storage?
Qdrant provides persistent storage via Docker volumes, metadata filtering (`source_type=nutrition`), and a clean Python client. The `qdrant_data` volume ensures nutrition chunks survive container restarts without re-ingestion.

### Why Redis for session management?
Redis serves as a LangGraph checkpointer — it persists the agent state (which specialists completed, partial results) per `session_id`. This means the graph can resume if interrupted and prevents re-running completed agents. It falls back gracefully to `MemorySaver` if Redis is unavailable.

### Supervisor Routing Design
The supervisor uses a two-layer routing strategy: fast keyword matching first (no LLM call needed for obvious queries), with LLM fallback for ambiguous cases. This reduces latency for common queries while maintaining flexibility for complex routing decisions.

---
## Architecture Evolution

The system went through a significant architectural redesign during development.

### Original Architecture
- **Agent B** was built with **CrewAI** as the RAG framework
- Used `qwen2.5:7b` as the primary LLM (4.7GB model)
- Agent B handled both nutrition RAG and crew-based task delegation

### Problems Encountered
- **OOM errors** — `qwen2.5:7b` (4.7GB) exceeded available VRAM (6GB RTX 4050) leaving no headroom for inference overhead
- **CrewAI overhead** — CrewAI added ~1.5GB of dependencies for a single-agent RAG task, causing slow container builds and startup times
- **Complexity mismatch** — CrewAI's multi-crew abstractions were overkill for a focused nutrition RAG pipeline

### Final Architecture Decisions
- Replaced CrewAI with **LlamaIndex** — purpose-built for RAG, ~300MB footprint, cleaner Qdrant integration
- Downgraded LLM to **qwen2.5:1.5b** (986MB) — fits comfortably in 6GB VRAM with room for inference
- Kept Agent B as a **fully independent FastAPI service** — preserving the microservice architecture principle


## Known Limitations

1. **Model size constraints** — `qwen2.5:1.5b` is small and occasionally misroutes ambiguous queries (e.g., "deadlift mistakes" may route to nutrition instead of exercise). A larger model (7B+) would improve routing accuracy significantly.

2. **No cross-session memory** — Redis persists LangGraph state per session, but the LLM does not have access to conversation history across separate API calls. Each request is effectively stateless from the LLM's perspective.

3. **Progress tool requires structured input** — The progress analysis tool uses deterministic math (Epley 1RM formula) and requires workout logs in a specific format (`exercise weightkg reps sets on YYYY-MM-DD`). Free-form natural language logs may not parse correctly.

4. **CPU-only embedding** — The `all-MiniLM-L6-v2` embedding model in Agent B runs on CPU. For high-throughput deployments, GPU-accelerated embeddings would reduce ingestion and retrieval latency.

5. **Exercise retrieval quality** — The exercise agent uses direct JSON filtering rather than semantic search. Queries with ambiguous muscle group names or non-standard terminology may return suboptimal results.

6. **GPU dependency** — The system requires an NVIDIA GPU for Ollama. CPU-only mode is possible but response latency increases to 30-60s per query, making it impractical for interactive use.
