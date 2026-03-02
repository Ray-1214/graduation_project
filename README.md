# Self-Evolving Skill Graph for LLM Agents

> **Skill-Level Evolution Without Parameter Updates: A Formal Framework for Structured Memory in LLM Agents**

[![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Overview

This repository contains the **paper** and **reference implementation** for a formal framework that enables LLM-based agents to accumulate, refine, and reorganize reusable skills — **without updating model parameters**. Instead of fine-tuning, the agent evolves a structured skill graph through memory-based transformations: node insertion, metadata refinement, edge reweighting, and subgraph contraction.

**Key idea:** Skills are explicit, composable policy units organized in a weighted directed graph with utility-driven memory partitioning. A formally specified evolution operator Φ governs how the graph transforms after each episode.

### What This Paper Proves (Analytically)

| Result | Statement |
|--------|-----------|
| **Bounded Growth** (Lemma 1) | Graph size is upper-bounded by capacity K at all times |
| **Depth Reduction** (Lemma 2) | Recursive macro-skill composition yields quasi-exponential planning depth reduction |
| **Entropy Convergence** (Proposition 1) | Structural entropy converges under stationary task distributions |
| **Utility Concentration** (Proposition 2) | Utility mass concentrates into a core–periphery topology |
| **Solvability Preservation** (Proposition 3) | The evolution operator never degrades task solvability |
| **Structural Equilibrium** (Theorem 1) | A unique structural equilibrium exists and is reached in finite time |
| **Partition Convergence** (Corollary 1) | Memory tier assignments converge |

> **Note:** This is a theoretical systems paper. All results are derived analytically from the formal model under explicitly stated assumptions. The implementation in this repo is the reference codebase being developed to validate the framework experimentally.

---

## Research Context

- **Authors:** Bo-Zhang Huang, Ting-Li Chung, Chao-Tung Yang
- **Affiliation:** Department of Computer Science, Tunghai University, Taiwan
- **Supervisor:** Prof. Chao-Tung Yang
- **Paper status:** Manuscript complete, implementation in progress
- **Target venue:** arXiv (cs.AI)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Main Agent                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   Task    │  │Memory-Guided │  │    Reflective     │  │
│  │Decomposer│→ │   Planner    │→ │    Evaluator      │  │
│  └──────────┘  └──────┬───────┘  └────────┬──────────┘  │
│                       │                    │             │
│              ┌────────▼────────┐  ┌────────▼──────────┐  │
│              │  Skill Graph    │  │ Evolution         │  │
│              │  G_t = (Σ,E,W) │←─│ Operator Φ        │  │
│              └────────┬───────┘  └───────────────────┘  │
│                       │                                  │
│         ┌─────────────┼─────────────┐                    │
│         ▼             ▼             ▼                    │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│   │ M_active │ │ M_cold   │ │M_archive │                │
│   │(working) │ │(standby) │ │(dormant) │                │
│   └──────────┘ └──────────┘ └──────────┘                │
│       Three-tier memory with hysteresis                  │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── paper/                    # LaTeX source + compiled PDF (planned)
├── agents/                   # Main agent orchestrator + evaluator
│   ├── main_agent.py
│   └── evaluator_agent.py
├── reasoning/                # Reasoning strategies
│   ├── cot.py                #   Chain of Thought
│   ├── tot.py                #   Tree of Thoughts
│   ├── react.py              #   ReAct (with tool dispatch)
│   └── reflexion.py          #   Reflexion (self-reflection)
├── memory/                   # Memory subsystems
│   ├── episodic_log.py       #   Episodic trace logging (Phase 0 ✅)
│   ├── vector_store.py       #   FAISS-backed vector store
│   ├── short_term.py
│   └── long_term.py
├── skill_graph/              # Core skill graph data structures (Phase 1 ✅)
│   ├── skill_node.py         #   SkillNode: σ = (π, β, I, μ)
│   ├── skill_graph.py        #   SkillGraph: G_t = (Σ_t, E_t, W_t)
│   └── memory_partition.py   #   Three-tier partition with hysteresis
├── skills/                   # Executable tool skills
│   ├── registry.py           #   BaseSkill interface + SkillRegistry
│   ├── web_search.py         #   DuckDuckGo search (Phase 1.5 ✅)
│   ├── admin_query.py        #   Human-in-the-loop queries (Phase 1.5 ✅)
│   ├── calculator.py
│   └── file_ops.py
├── rag/                      # Retrieval-Augmented Generation
│   ├── knowledge_store.py    #   Semantic knowledge cache (Phase 1.5 ✅)
│   ├── retriever.py
│   └── indexer.py
├── core/                     # Shared infrastructure
│   ├── llm_interface.py      #   Local LLM via llama-cpp-python
│   ├── prompt_builder.py     #   Template engine for all strategies
│   └── config.py
├── tests/                    # Acceptance tests (pytest)
│   ├── test_skill_graph.py   #   18 tests — Phase 1 (all pass ✅)
│   └── test_knowledge.py     #   15 tests — Phase 1.5 (all pass ✅)
├── experiments/              # Experiment runners
└── requirements.txt
```

---

## Implementation Progress

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Trace format (TraceStep, EpisodicTrace) | ✅ Complete |
| **Phase 1** | Skill Graph data structures (SkillNode, SkillGraph, MemoryPartition) | ✅ Complete (18/18 tests pass) |
| **Phase 1.5** | Active knowledge acquisition (WebSearch, AdminQuery, KnowledgeStore) | ✅ Complete (15/15 tests pass) |
| **Phase 2** | Evolution Operator Φ (SkillAbstractor, macro-skill contraction, SkillDocumentGenerator) | 🔧 In progress |
| **Phase 3** | Agent integration (SkillRetriever, prompt injection, Skill Document updates) | ⏳ Planned |
| **Phase 4** | Metrics & experiments (5 evaluation criteria from the paper) | ⏳ Planned |
| **Phase 5** | Visualization (FastAPI backend + React/D3.js frontend) | ⏳ Planned |

### Formal Definitions Implemented

| Definition | Paper Section | Implementation |
|-----------|---------------|----------------|
| Def. 2: Skill σ = (π, β, I, μ) | §3 | `skill_graph/skill_node.py` |
| Def. 3: Skill Graph G_t = (Σ, E, W) | §3 | `skill_graph/skill_graph.py` |
| Def. 4: Utility U(σ) = αr + βf − γc | §3 | `SkillNode.compute_utility()` |
| Def. 5: Memory Partition (active/cold/archive) | §3 | `skill_graph/memory_partition.py` |
| Def. 9: Structural Entropy H(G_t) | §3 | `SkillGraph.compute_entropy()` |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Ray-1214/graduation_project.git
cd graduation_project

# Install dependencies
pip install -r requirements.txt
pip install pytest ddgs  # for testing + web search

# Run Phase 1 tests (Skill Graph)
python -m pytest tests/test_skill_graph.py -v

# Run Phase 1.5 tests (Knowledge Acquisition)
python -m pytest tests/test_knowledge.py -v
```

### Requirements

- Python 3.10+
- Local LLM: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with Mistral-7B-Instruct GGUF
- No OpenAI API required — all inference runs locally

---

## Formal Framework Summary

**Core question:** Can an LLM agent accumulate reusable procedural knowledge across episodes through structured memory alone, without parameter updates?

**Approach:** Define skills as composable units σ = (π, β, I, μ) in a weighted directed graph. The evolution operator Φ transforms the graph after each episode through:

1. **Utility evaluation** — decay + reinforcement for used skills
2. **Skill insertion** — MDL-based abstraction from reasoning traces
3. **Subgraph contraction** — merge frequent skill co-occurrences into macro-skills
4. **Memory tier update** — hysteresis-based partition reassignment

**Key properties proved:**
- Graph size bounded by capacity K (Lemma 1)
- Planning depth reduces quasi-exponentially with macro-skill depth (Lemma 2)
- Structural entropy converges to H* under stationary tasks (Proposition 1)
- Unique structural equilibrium exists (Theorem 1)

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{huang2026skillevolution,
  title={Skill-Level Evolution Without Parameter Updates: A Formal Framework for Structured Memory in LLM Agents},
  author={Huang, Bo-Zhang and Chung, Ting-Li and Yang, Chao-Tung},
  year={2026},
  note={Manuscript under preparation. Code: \url{https://github.com/Ray-1214/graduation_project}}
}
```

---

## arXiv Endorsement Request

We are preparing to submit this paper to **arXiv cs.AI**. As new submitters, we need an endorsement from someone who has published in the cs.AI category.

**If you have cs.AI submission privileges and find this work to be a legitimate research contribution, we would greatly appreciate your endorsement.**

The paper presents a purely theoretical framework (no empirical claims) with:
- 9 formal definitions
- 5 explicitly stated assumptions
- 2 lemmas, 4 propositions, 1 theorem, 1 corollary (all with proofs)
- A reference implementation with 33 passing acceptance tests

To endorse, please visit [arXiv endorsement page](https://arxiv.org/auth/endorse) or contact us:
- 📧 Bo-Zhang Huang: [bozhanghuang.ac@gmail.com](mailto:bozhanghuang.ac@gmail.com)
- 📧 Ting-Li Chung: [tinglichung.ac@gmail.com](mailto:tinglichung.ac@gmail.com)

We are happy to provide the full manuscript PDF upon request.

---

## License

MIT

---

*This project is part of an undergraduate thesis at the Department of Computer Science, Tunghai University, supervised by Prof. Chao-Tung Yang.*
