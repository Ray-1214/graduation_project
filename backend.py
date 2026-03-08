"""
FastAPI backend — transparent API for the Self-Evolving Skill Graph agent.

Endpoints:
  POST /api/run              — Execute a single task
  GET  /api/graph            — Current skill graph JSON
  GET  /api/metrics          — All historical metrics
  GET  /api/graph/history/{id} — Graph snapshot at episode N
  GET  /api/episode/{id}     — Full EpisodeDetail for one episode
  GET  /api/episodes         — Summary list of all episodes
  WS   /ws/live              — Real-time event stream
  POST /api/admin/respond    — Admin responds to agent query
  POST /api/admin/skip       — Admin skips agent query
  GET  /api/admin/pending    — Pending admin queries
  GET  /api/knowledge        — KnowledgeStore entries
  GET  /api/history          — Command history
  GET  /api/status           — Agent readiness check

Launch:
  uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════
#  Pydantic models (request / response)
# ═══════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    task_description: str
    mode: str = "auto"  # auto | learn | execute
    expected_answer: str = ""


class AdminRespondRequest(BaseModel):
    query_id: str
    response: str


class AdminSkipRequest(BaseModel):
    query_id: str


# ═══════════════════════════════════════════════════════════════════
#  Application singleton + state
# ═══════════════════════════════════════════════════════════════════

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Self-Evolving Skill Graph — API",
    version="1.0.0",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ─────────────────────────────────────────────────

# Agent singleton (lazy init)
_agent = None
_agent_error: Optional[str] = None

# Metrics tracker
_tracker = None

# Episode history: list of EpisodeDetail dicts
_episode_history: List[Dict[str, Any]] = []

# Graph snapshots per episode: {episode_id: snapshot_dict}
_graph_snapshots: Dict[int, Dict[str, Any]] = {}

# Command history (from /api/run calls)
_command_history: List[Dict[str, Any]] = []

# WebSocket clients
_ws_clients: Set[WebSocket] = set()

# Thread pool for blocking agent calls
_executor = ThreadPoolExecutor(max_workers=2)


# ═══════════════════════════════════════════════════════════════════
#  Lazy agent initialisation
# ═══════════════════════════════════════════════════════════════════

def _get_agent():
    """Lazily create MainAgent + MetricsTracker singleton."""
    global _agent, _agent_error, _tracker
    if _agent is not None:
        return _agent
    if _agent_error is not None:
        return None
    try:
        from agents.main_agent import MainAgent
        from skill_graph.metrics import MetricsTracker

        _agent = MainAgent()
        _tracker = MetricsTracker()

        # Switch AdminQuery to async mode so agent doesn't block on input()
        admin_skill = _agent.skills.get("admin_query")
        if admin_skill is not None:
            admin_skill.mode = "async"
            # Wire callback for WebSocket notifications
            admin_skill._on_query_posted = _on_admin_query_posted

        logger.info("MainAgent initialised successfully")
        return _agent
    except Exception as exc:
        _agent_error = str(exc)
        logger.error("Failed to initialise MainAgent: %s", exc)
        return None


def _get_tracker():
    """Return MetricsTracker (requires agent to be initialised)."""
    _get_agent()  # ensure init
    return _tracker


def _get_admin_skill():
    """Return the AdminQuery skill instance."""
    agent = _get_agent()
    if agent is None:
        return None
    return agent.skills.get("admin_query")


# ═══════════════════════════════════════════════════════════════════
#  WebSocket broadcast helpers
# ═══════════════════════════════════════════════════════════════════

async def _broadcast(message: dict):
    """Send a JSON message to all connected WebSocket clients."""
    dead: List[WebSocket] = []
    data = json.dumps(message, ensure_ascii=False, default=str)
    for ws in _ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


def _broadcast_sync(message: dict):
    """Fire-and-forget broadcast from a sync context (executor thread)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast(message), loop)
        else:
            loop.run_until_complete(_broadcast(message))
    except Exception:
        pass


def _on_admin_query_posted(query_id: str, question: str):
    """Callback from AdminQuery._execute_async — notify WS clients."""
    _broadcast_sync({
        "type": "admin_query",
        "query_id": query_id,
        "question": question,
    })
    _broadcast_sync({
        "type": "agent_waiting",
        "reason": "admin_query",
        "query_id": query_id,
    })


# ═══════════════════════════════════════════════════════════════════
#  EpisodeDetail builder
# ═══════════════════════════════════════════════════════════════════

def _build_episode_detail(
    episode_id: int,
    task_desc: str,
    expected: str,
    result,  # AgentResult
    metrics_rec: Dict[str, Any],
    graph_snapshot: Dict[str, Any],
    duration_ms: int,
) -> Dict[str, Any]:
    """Build the full EpisodeDetail dict from AgentResult."""
    episode_raw = result.episode or {}
    steps = episode_raw.get("steps", [])

    # ── Parse steps into phases ──────────────────────────────
    phase_1 = _extract_phase(steps, "thought", until="action")
    phase_2_steps = _extract_execution_steps(steps)
    phase_3 = _extract_verification(result)
    phase_4 = _extract_reflexion(result)

    detail = {
        "episode_id": episode_id,
        "task": {
            "id": f"t-{episode_id:03d}",
            "tier": 0,
            "description": task_desc,
            "expected_answer": expected,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_total_ms": duration_ms,

        "context_assembly": {
            "slots": {},
            "total_tokens": 0,
            "budget": 4096,
            "overflow_handled": False,
        },

        "compound_reasoning": {
            "strategy_selected": result.strategy or "unknown",
            "strategy_reason": "",

            "phase_1_thinking": phase_1,
            "phase_2_execution": {
                "steps": phase_2_steps,
                "total_llm_calls": sum(
                    1 for s in phase_2_steps
                    if s.get("type") in ("thought", "final_answer")
                ),
                "total_tool_calls": sum(
                    1 for s in phase_2_steps
                    if s.get("type") == "action"
                ),
                "duration_ms": duration_ms // 2,
            },
            "phase_3_verification": phase_3,
            "phase_4_reflexion": phase_4,
        },

        "evolution": {
            "events": [],
            "graph_ops": {
                "insertions": 0,
                "contractions": 0,
                "updates": 0,
            },
            "graph_after": {
                "sigma_size": graph_snapshot.get("num_skills", 0),
                "entropy": graph_snapshot.get("structural_entropy", 0),
            },
        },

        "metrics": metrics_rec,

        "answer": result.answer,
        "correct": _check_correct(result.answer, expected),
    }

    # Populate evolution events from log string
    evo_log = result.evolution_log or ""
    if "inserted" in evo_log.lower():
        detail["evolution"]["graph_ops"]["insertions"] = 1
    if "contract" in evo_log.lower():
        detail["evolution"]["graph_ops"]["contractions"] = 1

    return detail


def _extract_phase(steps, step_type, until=None):
    """Extract first phase of a particular step_type from episode steps."""
    content_parts = []
    for s in steps:
        st = s.get("step_type", s.get("type", ""))
        if until and st == until:
            break
        if st == step_type:
            content_parts.append(s.get("content", ""))
    return {
        "parsed_result": {
            "content": "\n".join(content_parts),
        },
        "duration_ms": 0,
    }


def _extract_execution_steps(steps):
    """Convert raw episode steps into numbered execution step list."""
    result = []
    for i, s in enumerate(steps):
        st = s.get("step_type", s.get("type", ""))
        entry = {
            "step": i + 1,
            "type": st,
            "content": s.get("content", ""),
        }
        meta = s.get("metadata", {})
        if st == "action" and meta:
            entry["tool"] = meta.get("tool", "")
            entry["tool_input"] = meta.get("input", "")
            entry["tool_output"] = meta.get("output", "")
        result.append(entry)
    return result


def _extract_verification(result):
    """Build phase 3 verification block."""
    block = {
        "hallucination_guard": {
            "overall_score": result.hallucination_score or 0.0,
            "hallucination_detected": (
                result.hallucination_score is not None
                and result.hallucination_score < 0.5
            ),
        },
        "verdict": result.verification_verdict or "not_run",
        "duration_ms": 0,
    }
    return block


def _extract_reflexion(result):
    """Build phase 4 reflexion block."""
    return {
        "reflection_text": result.reflection or "",
        "reflexion_entries": [],
        "duration_ms": 0,
    }


def _check_correct(answer: str, expected: str) -> bool:
    """Simple accuracy check."""
    if not expected:
        return False
    a = answer.strip().lower()
    e = expected.strip().lower()
    return e in a or a in e


# ═══════════════════════════════════════════════════════════════════
#  1) POST /api/run
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/run")
async def api_run(req: RunRequest):
    """Execute a task and return the full EpisodeDetail."""
    agent = _get_agent()
    if agent is None:
        return {"error": f"Agent not ready: {_agent_error}"}

    tracker = _get_tracker()
    episode_id = len(_episode_history)

    # Determine strategy from mode
    strategy = None  # auto
    use_rag = False
    if req.mode == "learn":
        use_rag = True
    elif req.mode == "execute":
        strategy = None  # still auto but skip reflection
    # auto = None

    # Broadcast start
    await _broadcast({
        "type": "phase_start",
        "phase": "running",
        "episode_id": episode_id,
    })

    # Record command history
    cmd_entry = {
        "command": req.task_description,
        "mode": req.mode,
        "episode_id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    _command_history.append(cmd_entry)

    # Run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    start_ms = time.time()

    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: agent.run(
                task=req.task_description,
                strategy=strategy,
                use_rag=use_rag,
                do_reflect=(req.mode != "execute"),
            ),
        )
    except Exception as exc:
        cmd_entry["status"] = "error"
        await _broadcast({
            "type": "episode_complete",
            "episode_id": episode_id,
            "error": str(exc),
        })
        return {"error": str(exc)}

    duration_ms = int((time.time() - start_ms) * 1000)

    # Record metrics
    graph = agent.skill_graph
    episode_raw = result.episode or {}
    raw_steps = len(episode_raw.get("steps", []))
    compressed_steps = max(1, raw_steps)

    evo_log = agent.evolution._last_log if hasattr(
        agent.evolution, "_last_log"
    ) else None
    contraction_ops = len(evo_log.contracted) if evo_log else 0
    total_graph_ops = (
        len(evo_log.inserted) + len(evo_log.rejected_skills) +
        len(evo_log.contracted) + len(evo_log.tier_changes)
        if evo_log else 0
    )

    metrics_rec = tracker.record(
        episode_id=episode_id,
        graph=graph,
        raw_trace_lengths=[raw_steps] if raw_steps > 0 else [1],
        compressed_trace_lengths=[compressed_steps] if compressed_steps > 0 else [1],
        contraction_ops=contraction_ops,
        total_graph_ops=total_graph_ops,
    )

    # Graph snapshot
    snapshot = graph.snapshot(agent.memory_partition)
    _graph_snapshots[episode_id] = snapshot

    # Build EpisodeDetail
    detail = _build_episode_detail(
        episode_id=episode_id,
        task_desc=req.task_description,
        expected=req.expected_answer,
        result=result,
        metrics_rec=metrics_rec,
        graph_snapshot=snapshot,
        duration_ms=duration_ms,
    )
    _episode_history.append(detail)
    cmd_entry["status"] = "completed"

    # Broadcast completion
    await _broadcast({
        "type": "episode_complete",
        "episode_id": episode_id,
        "metrics": metrics_rec,
    })

    return detail


# ═══════════════════════════════════════════════════════════════════
#  2) GET /api/graph
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/graph")
async def api_graph():
    """Return the current skill graph as JSON."""
    agent = _get_agent()
    if agent is None:
        return {"error": f"Agent not ready: {_agent_error}"}

    graph = agent.skill_graph
    snap = graph.snapshot(agent.memory_partition)

    return {
        "nodes": snap.get("nodes", []),
        "edges": snap.get("edges", []),
        "entropy": snap.get("structural_entropy", 0),
        "capacity": snap.get("capacity", 0),
        "sigma_size": snap.get("num_skills", 0),
    }


# ═══════════════════════════════════════════════════════════════════
#  3) GET /api/metrics
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/metrics")
async def api_metrics():
    """Return all historical metrics with iteration summaries."""
    tracker = _get_tracker()
    if tracker is None:
        return {"history": [], "iterations": []}

    history = tracker.get_history()
    iterations = tracker.get_iteration_summary(episodes_per_iteration=10)
    return {
        "history": history,
        "iterations": iterations,
    }


# ═══════════════════════════════════════════════════════════════════
#  4) GET /api/graph/history/{episode_id}
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/graph/history/{episode_id}")
async def api_graph_history(episode_id: int):
    """Return the graph snapshot at a specific episode."""
    if episode_id in _graph_snapshots:
        return _graph_snapshots[episode_id]
    return {"error": f"No snapshot for episode {episode_id}"}


# ═══════════════════════════════════════════════════════════════════
#  5) GET /api/episode/{episode_id}
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/episode/{episode_id}")
async def api_episode(episode_id: int):
    """Return the full EpisodeDetail for one episode."""
    if 0 <= episode_id < len(_episode_history):
        return _episode_history[episode_id]
    return {"error": f"Episode {episode_id} not found"}


# ═══════════════════════════════════════════════════════════════════
#  6) GET /api/episodes
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/episodes")
async def api_episodes():
    """Return summary list of all episodes."""
    summaries = []
    for ep in _episode_history:
        summaries.append({
            "episode_id": ep["episode_id"],
            "task": ep["task"]["description"],
            "strategy": ep["compound_reasoning"]["strategy_selected"],
            "correct": ep.get("correct", False),
            "duration_ms": ep.get("duration_total_ms", 0),
            "answer": ep.get("answer", ""),
            "timestamp": ep.get("timestamp", ""),
        })
    return summaries


# ═══════════════════════════════════════════════════════════════════
#  7) WebSocket /ws/live
# ═══════════════════════════════════════════════════════════════════

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Real-time event stream."""
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        while True:
            # Keep alive — client can send pings; we just listen
            data = await websocket.receive_text()
            # Echo-back for ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d remain)",
                     len(_ws_clients))


# ═══════════════════════════════════════════════════════════════════
#  8) POST /api/admin/respond
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/admin/respond")
async def api_admin_respond(req: AdminRespondRequest):
    """Admin responds to a pending agent query."""
    skill = _get_admin_skill()
    if skill is None:
        return {"error": "AdminQuery skill not available"}

    ok = skill.set_response(req.query_id, req.response)
    if not ok:
        return {"error": f"Query {req.query_id} not found or already resolved"}

    await _broadcast({
        "type": "admin_response_received",
        "query_id": req.query_id,
    })
    return {"status": "ok", "query_id": req.query_id}


# ═══════════════════════════════════════════════════════════════════
#  9) POST /api/admin/skip
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/admin/skip")
async def api_admin_skip(req: AdminSkipRequest):
    """Admin skips a pending agent query."""
    skill = _get_admin_skill()
    if skill is None:
        return {"error": "AdminQuery skill not available"}

    ok = skill.skip_query(req.query_id)
    if not ok:
        return {"error": f"Query {req.query_id} not found or already resolved"}

    await _broadcast({
        "type": "admin_response_received",
        "query_id": req.query_id,
        "skipped": True,
    })
    return {"status": "skipped", "query_id": req.query_id}


# ═══════════════════════════════════════════════════════════════════
#  10) GET /api/admin/pending
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/admin/pending")
async def api_admin_pending():
    """Return list of pending admin queries."""
    skill = _get_admin_skill()
    if skill is None:
        return []

    pending = []
    for qid, entry in skill.pending_queries.items():
        pending.append({
            "query_id": qid,
            "question": entry["question"],
            "timestamp": datetime.fromtimestamp(
                entry["timestamp"], tz=timezone.utc
            ).isoformat(),
            "context": entry.get("context", ""),
        })
    return pending


# ═══════════════════════════════════════════════════════════════════
#  11) GET /api/knowledge
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/knowledge")
async def api_knowledge(source: Optional[str] = None):
    """Return all knowledge entries, optionally filtered by source."""
    agent = _get_agent()
    if agent is None:
        return []

    store = agent.knowledge_store
    entries = []
    for entry in store._entries.values():
        d = entry.to_dict()
        if source and d.get("source") != source:
            continue
        entries.append(d)
    return entries


# ═══════════════════════════════════════════════════════════════════
#  12) GET /api/history
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/history")
async def api_history():
    """Return command history from /api/run calls."""
    return _command_history


# ═══════════════════════════════════════════════════════════════════
#  GET /api/status — readiness check
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status():
    """Check agent readiness."""
    agent = _get_agent()
    return {
        "ready": agent is not None,
        "error": _agent_error,
        "episodes_completed": len(_episode_history),
        "graph_size": len(agent.skill_graph) if agent else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  POST /api/graph/save — persist SkillGraph + MemoryPartition
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/graph/save")
async def api_save_graph():
    """Manually trigger SkillGraph persistence."""
    agent = _get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        agent.save_state()
        return {
            "status": "saved",
            "skills": len(agent.skill_graph),
            "edges": agent.skill_graph._graph.number_of_edges(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════
#  GET /api/graph/load — reload from disk (development use)
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/graph/load")
async def api_reload_graph():
    """Reload SkillGraph and MemoryPartition from disk."""
    agent = _get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        from skill_graph.skill_graph import SkillGraph
        from skill_graph.memory_partition import MemoryPartition

        agent.skill_graph = SkillGraph.load(agent._graph_path)
        agent.memory_partition = MemoryPartition.load(agent._partition_path)
        return {
            "status": "reloaded",
            "skills": len(agent.skill_graph),
            "edges": agent.skill_graph._graph.number_of_edges(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════
#  Static file mount (production)
# ═══════════════════════════════════════════════════════════════════

_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount(
        "/",
        StaticFiles(directory=str(_frontend_dist), html=True),
        name="frontend",
    )
    logger.info("Mounted frontend from %s", _frontend_dist)
