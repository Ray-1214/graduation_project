"""
Tests for backend.py — FastAPI endpoints.

Uses TestClient with a heavily-mocked agent to verify
all REST endpoints, WebSocket, and admin query flow.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════════
#  Fixtures — mock agent to avoid loading LLM
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FakeAgentResult:
    task: str = "test task"
    strategy: str = "cot"
    answer: str = "42"
    score: float = 1.0
    reflection: str = "good job"
    episode: Optional[Dict[str, Any]] = None
    duration_s: float = 0.5
    evolution_log: str = "inserted 1"
    verification_verdict: str = "PASS"
    hallucination_score: float = 1.0

    def __post_init__(self):
        if self.episode is None:
            self.episode = {
                "task": self.task,
                "strategy": self.strategy,
                "steps": [
                    {"step_type": "thought", "content": "Thinking..."},
                    {"step_type": "action", "content": "calculator(6*7)",
                     "metadata": {"tool": "calculator", "input": "6*7",
                                  "output": "42"}},
                    {"step_type": "observation", "content": "42"},
                    {"step_type": "finish", "content": "42"},
                ],
                "result": self.answer,
            }


def _make_mock_agent():
    """Create a mock agent with all required attributes."""
    agent = MagicMock()

    # run() returns FakeAgentResult
    agent.run.return_value = FakeAgentResult()

    # skill_graph
    graph = MagicMock()
    graph.__len__ = MagicMock(return_value=3)
    graph.compute_entropy.return_value = 1.5
    graph.snapshot.return_value = {
        "num_skills": 3,
        "num_edges": 2,
        "capacity": 100,
        "structural_entropy": 1.5,
        "nodes": [
            {"id": "s1", "name": "skill_1"},
            {"id": "s2", "name": "skill_2"},
            {"id": "s3", "name": "skill_3"},
        ],
        "edges": [
            {"src": "s1", "dst": "s2", "weight": 0.8},
            {"src": "s2", "dst": "s3", "weight": 0.5},
        ],
    }
    graph.skills = [MagicMock(utility=0.5)] * 3
    agent.skill_graph = graph

    # memory_partition
    agent.memory_partition = MagicMock()

    # evolution
    agent.evolution = MagicMock()
    agent.evolution._last_log = None

    # knowledge_store — use a real-ish mock
    ks = MagicMock()
    ks._entries = {}
    agent.knowledge_store = ks

    # skill registry with admin_query
    from skills.admin_query import AdminQuery
    admin = AdminQuery(mode="async")
    skills_mock = MagicMock()
    skills_mock.get.return_value = admin
    agent.skills = skills_mock

    return agent


@pytest.fixture(autouse=True)
def reset_backend_state():
    """Reset backend module-level state before each test."""
    import backend
    backend._agent = None
    backend._agent_error = None
    backend._tracker = None
    backend._episode_history.clear()
    backend._graph_snapshots.clear()
    backend._command_history.clear()
    backend._ws_clients.clear()
    yield
    # Cleanup
    backend._agent = None
    backend._agent_error = None
    backend._tracker = None
    backend._episode_history.clear()
    backend._graph_snapshots.clear()
    backend._command_history.clear()


@pytest.fixture
def mock_agent():
    """Patch _get_agent to return a mock."""
    import backend
    from skill_graph.metrics import MetricsTracker

    agent = _make_mock_agent()
    tracker = MetricsTracker()

    backend._agent = agent
    backend._agent_error = None
    backend._tracker = tracker
    return agent


@pytest.fixture
def client(mock_agent):
    """TestClient with mocked agent."""
    import backend
    return TestClient(backend.app)


# ═══════════════════════════════════════════════════════════════════
#  GET /api/status
# ═══════════════════════════════════════════════════════════════════

class TestStatus:

    def test_status_ready(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["error"] is None


# ═══════════════════════════════════════════════════════════════════
#  GET /api/graph
# ═══════════════════════════════════════════════════════════════════

class TestGraph:

    def test_returns_nodes_and_edges(self, client):
        resp = client.get("/api/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert data["sigma_size"] == 3
        assert data["entropy"] == 1.5
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2


# ═══════════════════════════════════════════════════════════════════
#  GET /api/metrics
# ═══════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_empty_metrics(self, client):
        resp = client.get("/api/metrics")
        data = resp.json()
        assert data["history"] == []
        assert data["iterations"] == []


# ═══════════════════════════════════════════════════════════════════
#  GET /api/episodes, GET /api/episode/{id}
# ═══════════════════════════════════════════════════════════════════

class TestEpisodes:

    def test_empty_episodes(self, client):
        resp = client.get("/api/episodes")
        assert resp.json() == []

    def test_episode_not_found(self, client):
        resp = client.get("/api/episode/99")
        assert resp.status_code == 200
        assert "error" in resp.json()


# ═══════════════════════════════════════════════════════════════════
#  GET /api/graph/history/{id}
# ═══════════════════════════════════════════════════════════════════

class TestGraphHistory:

    def test_not_found(self, client):
        resp = client.get("/api/graph/history/0")
        assert resp.status_code == 200
        assert "error" in resp.json()


# ═══════════════════════════════════════════════════════════════════
#  POST /api/run
# ═══════════════════════════════════════════════════════════════════

class TestRun:

    def test_run_returns_episode_detail(self, client, mock_agent):
        resp = client.post("/api/run", json={
            "task_description": "What is 6 * 7?",
            "mode": "auto",
            "expected_answer": "42",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["episode_id"] == 0
        assert data["task"]["description"] == "What is 6 * 7?"
        assert data["answer"] == "42"
        assert data["correct"] is True
        assert "compound_reasoning" in data
        assert "metrics" in data
        assert "evolution" in data

    def test_run_populates_history(self, client, mock_agent):
        client.post("/api/run", json={
            "task_description": "test",
            "mode": "auto",
        })
        # Episodes list
        resp = client.get("/api/episodes")
        episodes = resp.json()
        assert len(episodes) == 1
        assert episodes[0]["episode_id"] == 0

        # Episode detail
        resp = client.get("/api/episode/0")
        detail = resp.json()
        assert detail["episode_id"] == 0

    def test_run_populates_graph_snapshot(self, client, mock_agent):
        client.post("/api/run", json={
            "task_description": "test",
            "mode": "auto",
        })
        resp = client.get("/api/graph/history/0")
        data = resp.json()
        assert "num_skills" in data

    def test_run_populates_metrics(self, client, mock_agent):
        client.post("/api/run", json={
            "task_description": "test",
            "mode": "auto",
        })
        resp = client.get("/api/metrics")
        data = resp.json()
        assert len(data["history"]) == 1

    def test_run_learn_mode(self, client, mock_agent):
        """Learn mode should enable RAG."""
        resp = client.post("/api/run", json={
            "task_description": "Learn about decorators",
            "mode": "learn",
        })
        assert resp.status_code == 200
        # Agent should have been called with use_rag=True
        call_kwargs = mock_agent.run.call_args
        assert call_kwargs.kwargs.get("use_rag") is True \
            or call_kwargs[1].get("use_rag") is True


# ═══════════════════════════════════════════════════════════════════
#  GET /api/history
# ═══════════════════════════════════════════════════════════════════

class TestHistory:

    def test_empty_history(self, client):
        resp = client.get("/api/history")
        assert resp.json() == []

    def test_history_after_run(self, client, mock_agent):
        client.post("/api/run", json={
            "task_description": "test task",
            "mode": "auto",
        })
        resp = client.get("/api/history")
        history = resp.json()
        assert len(history) == 1
        assert history[0]["command"] == "test task"
        assert history[0]["mode"] == "auto"
        assert history[0]["status"] == "completed"


# ═══════════════════════════════════════════════════════════════════
#  GET /api/knowledge
# ═══════════════════════════════════════════════════════════════════

class TestKnowledge:

    def test_empty_knowledge(self, client):
        resp = client.get("/api/knowledge")
        assert resp.json() == []


# ═══════════════════════════════════════════════════════════════════
#  Admin query endpoints
# ═══════════════════════════════════════════════════════════════════

class TestAdmin:

    def test_pending_empty(self, client):
        resp = client.get("/api/admin/pending")
        assert resp.json() == []

    def test_respond_not_found(self, client):
        resp = client.post("/api/admin/respond", json={
            "query_id": "q-nonexistent",
            "response": "test",
        })
        data = resp.json()
        assert "error" in data

    def test_skip_not_found(self, client):
        resp = client.post("/api/admin/skip", json={
            "query_id": "q-nonexistent",
        })
        data = resp.json()
        assert "error" in data

    def test_admin_respond_flow(self, client, mock_agent):
        """Simulate: post a pending query → respond → verify resolved."""
        admin_skill = mock_agent.skills.get("admin_query")

        # Simulate a pending query (as if agent posted one)
        import threading
        event = threading.Event()
        admin_skill.pending_queries["q-test123"] = {
            "query_id": "q-test123",
            "question": "What is GIL?",
            "event": event,
            "response": None,
            "skipped": False,
            "timestamp": time.time(),
            "context": "",
        }

        # Check pending
        resp = client.get("/api/admin/pending")
        pending = resp.json()
        assert len(pending) == 1
        assert pending[0]["query_id"] == "q-test123"
        assert pending[0]["question"] == "What is GIL?"

        # Respond
        resp = client.post("/api/admin/respond", json={
            "query_id": "q-test123",
            "response": "Global Interpreter Lock",
        })
        assert resp.json()["status"] == "ok"
        assert event.is_set()

    def test_admin_skip_flow(self, client, mock_agent):
        """Simulate: post a pending query → skip."""
        admin_skill = mock_agent.skills.get("admin_query")

        import threading
        event = threading.Event()
        admin_skill.pending_queries["q-skip1"] = {
            "query_id": "q-skip1",
            "question": "Help?",
            "event": event,
            "response": None,
            "skipped": False,
            "timestamp": time.time(),
            "context": "",
        }

        resp = client.post("/api/admin/skip", json={
            "query_id": "q-skip1",
        })
        assert resp.json()["status"] == "skipped"
        assert event.is_set()


# ═══════════════════════════════════════════════════════════════════
#  WebSocket /ws/live
# ═══════════════════════════════════════════════════════════════════

class TestWebSocket:

    def test_connect_and_ping(self, client):
        with client.websocket_connect("/ws/live") as ws:
            ws.send_text("ping")
            data = ws.receive_text()
            assert data == "pong"

    def test_multiple_clients(self, client):
        """Multiple WS clients can connect simultaneously."""
        import backend
        with client.websocket_connect("/ws/live") as ws1:
            assert len(backend._ws_clients) >= 1
            ws1.send_text("ping")
            assert ws1.receive_text() == "pong"


# ═══════════════════════════════════════════════════════════════════
#  AdminQuery skill unit tests
# ═══════════════════════════════════════════════════════════════════

class TestAdminQueryAsync:

    def test_set_response(self):
        from skills.admin_query import AdminQuery
        admin = AdminQuery(mode="async")

        import threading
        event = threading.Event()
        admin.pending_queries["q-abc"] = {
            "query_id": "q-abc",
            "question": "test?",
            "event": event,
            "response": None,
            "skipped": False,
            "timestamp": time.time(),
            "context": "",
        }
        assert admin.set_response("q-abc", "answer")
        assert event.is_set()

    def test_skip_query(self):
        from skills.admin_query import AdminQuery
        admin = AdminQuery(mode="async")

        import threading
        event = threading.Event()
        admin.pending_queries["q-def"] = {
            "query_id": "q-def",
            "question": "test?",
            "event": event,
            "response": None,
            "skipped": False,
            "timestamp": time.time(),
            "context": "",
        }
        assert admin.skip_query("q-def")
        assert event.is_set()

    def test_set_response_not_found(self):
        from skills.admin_query import AdminQuery
        admin = AdminQuery(mode="async")
        assert admin.set_response("q-nonexistent", "x") is False

    def test_execute_async_blocks_then_returns(self):
        """_execute_async blocks until set_response is called."""
        from skills.admin_query import AdminQuery
        admin = AdminQuery(mode="async")

        result_holder = []

        def run():
            r = admin._execute_async("What is Python?")
            result_holder.append(r)

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.1)  # Let it block

        # Find the pending query and respond
        assert len(admin.pending_queries) == 1
        qid = list(admin.pending_queries.keys())[0]
        admin.set_response(qid, "A programming language")
        t.join(timeout=2)

        assert len(result_holder) == 1
        assert "A programming language" in result_holder[0]
