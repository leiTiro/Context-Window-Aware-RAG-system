
# Context-Window-Aware RAG (Option 3 Submission)

## Overview
This project demonstrates a **Context-Window-Aware Retrieval-Augmented Generation (RAG)** system.
Instead of sending all available information to a language model, the system deliberately assembles
a bounded context window using predefined sections, strict token budgets, and explicit overflow strategies.

The focus of this submission is **context economics, prioritisation, and safe failure handling**.

---

## High-Level System Flow
1. User submits a query via a CLI
2. Relevant documents are retrieved using vector similarity
3. Short-term conversational memory is collected
4. A Context Builder assembles the final prompt using fixed budgets
5. Overflows are detected and handled deterministically
6. The final context and a token usage report are displayed

---

## Core Components

### Vector Retriever (`retriever.py`)
- Loads a small local corpus of text files
- Uses **TF-IDF vectorisation** and **cosine similarity**
- Ranks documents by semantic similarity
- Intentionally retrieves more content than can fit for overflow demos

### Memory Store (`memory.py`)
- Stores short, user-specific facts
- Fixed budget: **55 tokens**
- Oldest items dropped first when budget is exceeded
- Represents conversational state, not knowledge

### Context Builder (`context_builder.py`)
The core decision-making component.

| Section | Budget | Overflow Strategy |
|------|------|------|
| Instructions | 255 | Truncate |
| Goal | 1500 | Preserve intent |
| Memory | 55 | Drop oldest |
| Retrieval | 550 | Drop lowest relevance |

Each section enforces hard limits and records overflow events.

### CLI Application (`app.py`)
- Orchestrates retrieval, memory, and context assembly
- Prints final context
- Displays token usage and overflow report

---

## Context Assembly Order
1. Instructions
2. Goal
3. Memory
4. Retrieval

This preserves instruction hierarchy and prevents lower-priority data from overriding system rules.

---

## No-Overflow Simulation
When documents and memory fit within budgets:
- No overflow is triggered
- All context is included
- Token report confirms usage below limits

Example:
```
INSTRUCTIONS: 13 / 255
GOAL: 9 / 1500
MEMORY: 26 / 55
RETRIEVAL: 305 / 550
```

---

## Overflow Simulation

### Retrieval Overflow
For demonstration, the retrieval budget is intentionally exceeded.

Behavior:
- Chunks ranked by relevance
- Lowest-relevance chunks dropped
- Overflow reported explicitly

Example:
```
RETRIEVAL overflowed: dropped 2 chunks
RETRIEVAL: 198 / 200 (overflowed)
```

### Memory Overflow
When memory exceeds 55 tokens:
- Oldest entries are dropped
- Recent context preserved

---

## Why Memory and Retrieval Differ
- Memory is **temporal and user-specific**
- Retrieval is **factual and relevance-based**
Each requires a different overflow strategy.

---

## Why This Design Matters
- Prevents silent truncation
- Makes trade-offs explicit
- Enables predictable agent behavior
- Scales to more complex agent systems

---

## How to Run
```bash
pip install -r requirements.txt
python app.py
```
## 0VERFLOW SIMULATIONS
---
Alter the Preset Token Budgets on Context_Builder to simulate Overflow and no Overflow scenerios 
by decreasing or increasing the budget

def __init__(self):
        self.budgets = {
            "instructions": 255,
            "goal": 1500,
            "memory": 55,
            #"retrieval": 550,
            "retrieval": 200,   # TEMPORARILY reduced from 550
        }
---

## What This Demonstrates
- Context-window economics
- Priority-aware prompt assembly
- Deterministic overflow handling
- Real vector retrieval
- Agent-ready system design
