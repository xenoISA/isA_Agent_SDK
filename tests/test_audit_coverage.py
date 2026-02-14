"""
Tests for audit coverage gaps (Issue #24).

Covers 7 critical code paths identified during codebase audit:
1. Concurrent SQLite operations (session_client.py, storage_client.py)
2. ISAAgentOptions validation (options.py)
3. Swarm circular handoff termination (swarm.py)
4. Sync-from-async detection (_query.py)
5. DAG deadlock/cycle handling (dag/scheduler.py)
6. OutputFormat.from_pydantic (options.py)
7. Resume error handling (_query.py)
"""

import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from isa_agent_sdk.clients.base import BackendConfig, ClientMode
from isa_agent_sdk.clients.session_client import LocalSessionBackend
from isa_agent_sdk.options import (
    ISAAgentOptions,
    MCPServerConfig,
    OutputFormat,
    OutputFormatType,
)
from isa_agent_sdk.agents.agent import Agent, AgentRunResult
from isa_agent_sdk.agents.swarm_types import (
    HandoffAction,
    SwarmAgent,
    SwarmState,
)
from isa_agent_sdk.agents.swarm import SwarmOrchestrator
from isa_agent_sdk._messages import AgentMessage
from isa_agent_sdk.dag.models import DAGTask, DAGTaskStatus, DAGState
from isa_agent_sdk.dag.scheduler import DAGScheduler


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_swarm.py)
# ---------------------------------------------------------------------------

def _make_runner(response_text: str):
    """Create a mock runner that yields a single result message."""
    async def runner(prompt, *, options=None):
        yield AgentMessage.result(response_text)
    return runner


def _make_agent(name: str, response_text: str) -> Agent:
    """Create an Agent with a mock runner returning fixed text."""
    opts = ISAAgentOptions()
    return Agent(name, opts, runner=_make_runner(response_text))


def _make_stateful_runner(responses: list):
    """Create a runner that returns different responses on successive calls."""
    call_count = {"n": 0}

    async def runner(prompt, *, options=None):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        yield AgentMessage.result(responses[idx])

    return runner


# ===========================================================================
# 1. TestConcurrentSQLiteOperations
# ===========================================================================

class TestConcurrentSQLiteOperations:
    """Test that _db_lock serializes concurrent writes with check_same_thread=False."""

    @pytest.fixture
    async def backend(self, tmp_path):
        config = BackendConfig(
            mode=ClientMode.LOCAL,
            local_storage_path=str(tmp_path),
            local_db_name="test_concurrent.db",
        )
        backend = LocalSessionBackend(config)
        await backend.initialize()
        yield backend
        await backend.close()

    async def test_concurrent_session_creates(self, backend):
        """Launch 10 concurrent create_session() calls, verify all succeed."""
        tasks = [
            backend.create_session(user_id="user1", title=f"Session {i}")
            for i in range(10)
        ]
        sessions = await asyncio.gather(*tasks)

        assert len(sessions) == 10
        # All session IDs must be unique
        ids = [s.id for s in sessions]
        assert len(set(ids)) == 10

    async def test_concurrent_session_updates(self, backend):
        """Create a session then launch 10 concurrent update_session() calls."""
        session = await backend.create_session(user_id="user1", title="Original")

        async def update_title(i):
            session_copy = await backend.get_session(session.id)
            session_copy.title = f"Updated {i}"
            return await backend.update_session(session_copy)

        results = await asyncio.gather(*[update_title(i) for i in range(10)])
        assert all(r is True for r in results)

        # Final session should have a valid title
        final = await backend.get_session(session.id)
        assert final is not None
        assert final.title.startswith("Updated")

    async def test_concurrent_add_messages(self, backend):
        """Create a session, launch 20 concurrent add_message() calls."""
        session = await backend.create_session(user_id="user1", title="Msg test")

        tasks = [
            backend.add_message(session.id, "user", f"Message {i}")
            for i in range(20)
        ]
        messages = await asyncio.gather(*tasks)

        assert len(messages) == 20
        # All message IDs should be unique
        msg_ids = [m.id for m in messages]
        assert len(set(msg_ids)) == 20

        # Verify count matches
        stored = await backend.get_messages(session.id)
        assert len(stored) == 20

    async def test_concurrent_mixed_operations(self, backend):
        """Mix of create/update/delete/add_message operations in parallel."""
        # Pre-create a session to update/delete/add messages to
        session = await backend.create_session(user_id="user1", title="Mixed")

        async def create_op():
            return await backend.create_session(user_id="user1", title="New")

        async def update_op():
            s = await backend.get_session(session.id)
            if s:
                s.title = "Updated"
                await backend.update_session(s)

        async def add_msg_op():
            await backend.add_message(session.id, "user", "hello")

        async def delete_op():
            # Create then delete a new session
            s = await backend.create_session(user_id="user1", title="ToDelete")
            await backend.delete_session(s.id)

        ops = (
            [create_op() for _ in range(3)]
            + [update_op() for _ in range(3)]
            + [add_msg_op() for _ in range(3)]
            + [delete_op() for _ in range(3)]
        )
        # No OperationalError should be raised
        await asyncio.gather(*ops)


# ===========================================================================
# 2. TestISAAgentOptionsValidation
# ===========================================================================

class TestISAAgentOptionsValidation:
    """Test ISAAgentOptions and MCPServerConfig validation."""

    def test_max_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            ISAAgentOptions(max_iterations=0)

    def test_max_iterations_negative_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            ISAAgentOptions(max_iterations=-5)

    def test_confidence_threshold_out_of_range_low(self):
        with pytest.raises(ValueError, match="failsafe_confidence_threshold"):
            ISAAgentOptions(failsafe_confidence_threshold=-0.1)

    def test_confidence_threshold_out_of_range_high(self):
        with pytest.raises(ValueError, match="failsafe_confidence_threshold"):
            ISAAgentOptions(failsafe_confidence_threshold=1.1)

    def test_confidence_threshold_boundaries(self):
        # Should not raise
        opts_low = ISAAgentOptions(failsafe_confidence_threshold=0.0)
        assert opts_low.failsafe_confidence_threshold == 0.0

        opts_high = ISAAgentOptions(failsafe_confidence_threshold=1.0)
        assert opts_high.failsafe_confidence_threshold == 1.0

    def test_summarization_threshold_zero_raises(self):
        with pytest.raises(ValueError, match="summarization_threshold must be positive"):
            ISAAgentOptions(summarization_threshold=0)

    def test_max_parallel_tasks_zero_raises(self):
        with pytest.raises(ValueError, match="max_parallel_tasks must be positive"):
            ISAAgentOptions(max_parallel_tasks=0)

    def test_max_task_retries_negative_raises(self):
        with pytest.raises(ValueError, match="max_task_retries must be non-negative"):
            ISAAgentOptions(max_task_retries=-1)

    def test_task_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="task_timeout must be positive"):
            ISAAgentOptions(task_timeout=0)

    def test_mcp_url_invalid_protocol(self):
        with pytest.raises(ValueError, match="url must start with"):
            MCPServerConfig(url="ftp://example.com")

    def test_mcp_url_valid_protocols(self):
        for proto in ("http://x.com", "https://x.com", "ws://x.com", "wss://x.com"):
            cfg = MCPServerConfig(url=proto)
            assert cfg.url == proto

    def test_mcp_missing_command_and_url(self):
        with pytest.raises(ValueError, match="requires either"):
            MCPServerConfig()

    def test_invalid_enum_string(self):
        with pytest.raises(ValueError):
            ISAAgentOptions(execution_mode="invalid")

    def test_valid_defaults(self):
        opts = ISAAgentOptions()
        assert opts.max_iterations == 50
        assert opts.failsafe_confidence_threshold == 0.7
        assert opts.max_parallel_tasks == 3


# ===========================================================================
# 3. TestSwarmCircularHandoff
# ===========================================================================

class TestSwarmCircularHandoff:
    """Test that circular handoffs are terminated by max_handoffs."""

    async def test_circular_handoff_ab_ab(self):
        """Agent A always hands off to B, B always hands off to A."""
        a = Agent("a", ISAAgentOptions(), runner=_make_runner("[HANDOFF: b] your turn"))
        b = Agent("b", ISAAgentOptions(), runner=_make_runner("[HANDOFF: a] your turn"))

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="Agent A"),
                SwarmAgent(agent=b, description="Agent B"),
            ],
            entry_agent="a",
            max_handoffs=4,
        )
        result = await swarm.run("ping pong")

        # Should terminate and return a result
        assert result is not None
        assert result.text is not None
        # Trace should show A->B->A->B pattern
        trace_pairs = [(h["from"], h["to"]) for h in result.handoff_trace]
        assert ("a", "b") in trace_pairs
        assert ("b", "a") in trace_pairs

    async def test_circular_handoff_three_agents(self):
        """A->B->C->A cycle. Verify max_handoffs stops it."""
        a = Agent("a", ISAAgentOptions(), runner=_make_runner("[HANDOFF: b] go"))
        b = Agent("b", ISAAgentOptions(), runner=_make_runner("[HANDOFF: c] go"))
        c = Agent("c", ISAAgentOptions(), runner=_make_runner("[HANDOFF: a] go"))

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="A"),
                SwarmAgent(agent=b, description="B"),
                SwarmAgent(agent=c, description="C"),
            ],
            entry_agent="a",
            max_handoffs=5,
        )
        result = await swarm.run("cycle")
        assert result is not None
        # All three agents should appear in the trace
        agents_in_trace = {h["from"] for h in result.handoff_trace}
        assert agents_in_trace == {"a", "b", "c"}

    async def test_max_handoffs_exact_limit(self):
        """Set max_handoffs=2, verify exactly 3 agent runs (loop runs max_handoffs+1)."""
        call_count = {"n": 0}

        async def counting_runner(prompt, *, options=None):
            call_count["n"] += 1
            yield AgentMessage.result("[HANDOFF: b] go" if call_count["n"] % 2 == 1 else "[HANDOFF: a] go")

        a = Agent("a", ISAAgentOptions(), runner=counting_runner)
        b = Agent("b", ISAAgentOptions(), runner=counting_runner)

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="A"),
                SwarmAgent(agent=b, description="B"),
            ],
            entry_agent="a",
            max_handoffs=2,
        )
        result = await swarm.run("test")
        # max_handoffs=2 → loop runs 3 times → 3 agent runs, 3 handoffs recorded
        assert call_count["n"] == 3
        assert len(result.handoff_trace) == 3

    async def test_handoff_trace_records_all(self):
        """Verify SwarmRunResult.handoff_trace records every handoff in the cycle."""
        responses_a = [
            "[HANDOFF: b] first handoff",
            "[HANDOFF: b] second handoff",
            "Done [COMPLETE]",
        ]
        responses_b = [
            "[HANDOFF: a] back to a",
            "[HANDOFF: a] back again",
        ]
        a = Agent("a", ISAAgentOptions(), runner=_make_stateful_runner(responses_a))
        b = Agent("b", ISAAgentOptions(), runner=_make_stateful_runner(responses_b))

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="A"),
                SwarmAgent(agent=b, description="B"),
            ],
            entry_agent="a",
            max_handoffs=10,
        )
        result = await swarm.run("test trace")

        # Expected trace: a->b, b->a, a->b, b->a, a completes
        assert len(result.handoff_trace) == 4
        assert result.handoff_trace[0]["from"] == "a"
        assert result.handoff_trace[0]["to"] == "b"
        assert result.handoff_trace[1]["from"] == "b"
        assert result.handoff_trace[1]["to"] == "a"
        assert result.handoff_trace[2]["from"] == "a"
        assert result.handoff_trace[2]["to"] == "b"
        assert result.handoff_trace[3]["from"] == "b"
        assert result.handoff_trace[3]["to"] == "a"
        assert result.final_agent == "a"


# ===========================================================================
# 4. TestQuerySyncFromAsyncContext
# ===========================================================================

class TestQuerySyncFromAsyncContext:
    """Test that sync wrappers raise when called from async context."""

    async def test_query_sync_from_async_raises(self):
        from isa_agent_sdk._query import query_sync

        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            # query_sync is a generator; must call next() to enter the body
            gen = query_sync("test")
            next(gen)

    async def test_ask_sync_from_async_raises(self):
        from isa_agent_sdk._query import ask_sync

        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            ask_sync("test")

    async def test_resume_sync_from_async_raises(self):
        from isa_agent_sdk._query import resume_sync

        with pytest.raises(RuntimeError, match="cannot be called from an async context"):
            gen = resume_sync("session_123")
            next(gen)

    async def test_query_sync_error_message_suggests_async(self):
        from isa_agent_sdk._query import query_sync

        with pytest.raises(RuntimeError) as exc_info:
            gen = query_sync("test")
            next(gen)
        assert "Use query() (async) instead" in str(exc_info.value)


# ===========================================================================
# 5. TestDAGDeadlockHandling
# ===========================================================================

class TestDAGDeadlockHandling:
    """Test DAG cycle detection and failure cascading."""

    def test_self_dependency_detected(self):
        dag = DAGScheduler.build_dag([
            {"id": "a", "title": "A", "depends_on": ["a"]},
        ])
        errors = DAGScheduler.validate(dag)
        assert any("depends on itself" in e for e in errors)

    def test_two_node_cycle_detected(self):
        dag = DAGScheduler.build_dag([
            {"id": "a", "title": "A", "depends_on": ["b"]},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ])
        errors = DAGScheduler.validate(dag)
        assert any("Cycle detected" in e for e in errors)

    def test_three_node_cycle_detected(self):
        dag = DAGScheduler.build_dag([
            {"id": "a", "title": "A", "depends_on": ["c"]},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["b"]},
        ])
        errors = DAGScheduler.validate(dag)
        assert any("Cycle detected" in e for e in errors)

    def test_compute_wavefronts_cycle_raises(self):
        dag = DAGScheduler.build_dag([
            {"id": "a", "title": "A", "depends_on": ["b"]},
            {"id": "b", "title": "B", "depends_on": ["a"]},
        ])
        with pytest.raises(ValueError, match="Cycle detected"):
            DAGScheduler.compute_wavefronts(dag)

    def test_failure_cascade_marks_dependents_skipped(self):
        """Fail task A, verify B (depends on A) and C (depends on B) are SKIPPED."""
        dag = DAGScheduler.build_dag([
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["b"]},
        ])
        DAGScheduler.mark_task_failed(dag, "a", "test failure")

        assert dag.tasks["a"].status == DAGTaskStatus.FAILED
        assert dag.tasks["b"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["c"].status == DAGTaskStatus.SKIPPED

    def test_failure_cascade_does_not_affect_independent(self):
        """Fail task A, verify task D (no deps) remains PENDING."""
        dag = DAGScheduler.build_dag([
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "d", "title": "D"},  # independent
        ])
        DAGScheduler.mark_task_failed(dag, "a", "test failure")

        assert dag.tasks["a"].status == DAGTaskStatus.FAILED
        assert dag.tasks["b"].status == DAGTaskStatus.SKIPPED
        assert dag.tasks["d"].status == DAGTaskStatus.PENDING

    def test_mixed_dag_with_failure(self):
        """Diamond DAG (A->B,C->D), fail B, verify D is SKIPPED but C completes."""
        dag = DAGScheduler.build_dag([
            {"id": "a", "title": "A"},
            {"id": "b", "title": "B", "depends_on": ["a"]},
            {"id": "c", "title": "C", "depends_on": ["a"]},
            {"id": "d", "title": "D", "depends_on": ["b", "c"]},
        ])
        # Complete A first, then fail B
        DAGScheduler.mark_task_completed(dag, "a", "done")
        DAGScheduler.mark_task_failed(dag, "b", "test failure")

        assert dag.tasks["a"].status == DAGTaskStatus.COMPLETED
        assert dag.tasks["b"].status == DAGTaskStatus.FAILED
        # C is independent of B, should remain PENDING (ready to run)
        assert dag.tasks["c"].status == DAGTaskStatus.PENDING
        # D depends on B (which failed), so D should be SKIPPED
        assert dag.tasks["d"].status == DAGTaskStatus.SKIPPED


# ===========================================================================
# 6. TestOutputFormatFromPydantic
# ===========================================================================

class TestOutputFormatFromPydantic:
    """Test OutputFormat.from_pydantic() schema extraction."""

    def test_from_pydantic_basic_model(self):
        from pydantic import BaseModel

        class Simple(BaseModel):
            name: str
            age: int
            active: bool = True

        fmt = OutputFormat.from_pydantic(Simple)
        assert fmt.type == OutputFormatType.JSON_SCHEMA
        assert fmt.schema is not None
        assert "properties" in fmt.schema
        assert "name" in fmt.schema["properties"]
        assert "age" in fmt.schema["properties"]

    def test_from_pydantic_nested_model(self):
        from pydantic import BaseModel

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        fmt = OutputFormat.from_pydantic(Person)
        assert fmt.schema is not None
        props = fmt.schema.get("properties", {})
        assert "address" in props

    def test_from_pydantic_non_model_raises(self):
        class NotAModel:
            pass

        with pytest.raises(TypeError, match="does not appear to be a Pydantic model"):
            OutputFormat.from_pydantic(NotAModel)

    def test_from_pydantic_strict_flag(self):
        from pydantic import BaseModel

        class M(BaseModel):
            x: int

        fmt = OutputFormat.from_pydantic(M, strict=False)
        assert fmt.strict is False

    def test_from_pydantic_schema_has_properties(self):
        from pydantic import BaseModel

        class M(BaseModel):
            x: int
            y: str

        fmt = OutputFormat.from_pydantic(M)
        assert "properties" in fmt.schema
        assert "required" in fmt.schema

    def test_from_pydantic_type_is_json_schema(self):
        from pydantic import BaseModel

        class M(BaseModel):
            x: int

        fmt = OutputFormat.from_pydantic(M)
        assert fmt.type == OutputFormatType.JSON_SCHEMA


# ===========================================================================
# 7. TestResumeWithStateCorruption
# ===========================================================================

class TestResumeWithStateCorruption:
    """Test resume() error handling paths."""

    def _patch_resume_modules(self, mods_dict):
        """Helper: patch sys.modules and return (originals, cleanup_fn)."""
        import sys
        originals = {}
        for mod_name, mock_mod in mods_dict.items():
            originals[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_mod

        def cleanup():
            for mod_name in mods_dict:
                if originals[mod_name] is not None:
                    sys.modules[mod_name] = originals[mod_name]
                else:
                    sys.modules.pop(mod_name, None)

        return cleanup

    async def test_resume_import_error_yields_error(self):
        """Patch imports to raise ImportError, verify error message."""
        from isa_agent_sdk._query import resume

        with patch.dict(
            "sys.modules",
            {"isa_agent_sdk.graphs.smart_agent_graph": None},
        ):
            messages = []
            async for msg in resume("session_test"):
                messages.append(msg)

            error_msgs = [m for m in messages if m.is_error]
            assert len(error_msgs) > 0
            assert any("not available" in (m.content or "").lower()
                      or "import" in (m.content or "").lower()
                      for m in error_msgs)

    async def test_resume_generic_exception_yields_error(self):
        """Mock graph build to raise RuntimeError, verify error is yielded."""
        from isa_agent_sdk._query import resume

        broken_mod = MagicMock()
        broken_mod.SmartAgentGraphBuilder.return_value.build_graph.side_effect = (
            RuntimeError("corruption detected")
        )
        mock_context_mod = MagicMock()
        mock_langgraph = MagicMock()

        cleanup = self._patch_resume_modules({
            "isa_agent_sdk.graphs.smart_agent_graph": broken_mod,
            "isa_agent_sdk.graphs.utils.context_init": mock_context_mod,
            "langgraph.types": mock_langgraph,
        })

        try:
            messages = []
            async for msg in resume("session_test"):
                messages.append(msg)

            error_msgs = [m for m in messages if m.is_error]
            assert len(error_msgs) > 0
            assert any("corruption detected" in (m.content or "") for m in error_msgs)
        finally:
            cleanup()

    async def test_resume_nonexistent_session_yields_error(self):
        """Mock graph to raise during build, verify error is yielded."""
        from isa_agent_sdk._query import resume

        broken_mod = MagicMock()
        broken_mod.SmartAgentGraphBuilder.return_value.build_graph.side_effect = (
            RuntimeError("session not found")
        )
        mock_context_mod = MagicMock()
        mock_langgraph = MagicMock()

        cleanup = self._patch_resume_modules({
            "isa_agent_sdk.graphs.smart_agent_graph": broken_mod,
            "isa_agent_sdk.graphs.utils.context_init": mock_context_mod,
            "langgraph.types": mock_langgraph,
        })

        try:
            messages = []
            async for msg in resume("nonexistent_session"):
                messages.append(msg)

            error_msgs = [m for m in messages if m.is_error]
            assert len(error_msgs) > 0
        finally:
            cleanup()

    async def test_resume_with_value_creates_command(self):
        """Patch graph, verify Command(resume=value) is passed."""
        from isa_agent_sdk._query import resume

        mock_graph = MagicMock()

        async def mock_astream_events(input_val, config=None, version=None):
            return
            yield  # make it an async generator

        mock_graph.astream_events = mock_astream_events

        broken_mod = MagicMock()
        broken_mod.SmartAgentGraphBuilder.return_value.build_graph.return_value = mock_graph

        mock_context_mod = MagicMock()

        async def mock_prepare(*args, **kwargs):
            return {"available_tools": [], "default_prompts": {}}

        mock_context_mod.prepare_runtime_context = mock_prepare

        mock_command_cls = MagicMock()
        mock_langgraph_types = MagicMock()
        mock_langgraph_types.Command = mock_command_cls

        cleanup = self._patch_resume_modules({
            "isa_agent_sdk.graphs.smart_agent_graph": broken_mod,
            "isa_agent_sdk.graphs.utils.context_init": mock_context_mod,
            "langgraph.types": mock_langgraph_types,
        })

        try:
            messages = []
            async for msg in resume("session_1", resume_value={"authorized": True}):
                messages.append(msg)

            # Verify Command was called with resume=value
            mock_command_cls.assert_called_once_with(resume={"authorized": True})
        finally:
            cleanup()

    async def test_resume_without_value_passes_none(self):
        """Patch graph, verify initial_input=None when no resume_value."""
        from isa_agent_sdk._query import resume

        captured_input = {}

        mock_graph = MagicMock()

        async def mock_astream_events(input_val, config=None, version=None):
            captured_input["input"] = input_val
            return
            yield

        mock_graph.astream_events = mock_astream_events

        broken_mod = MagicMock()
        broken_mod.SmartAgentGraphBuilder.return_value.build_graph.return_value = mock_graph

        mock_context_mod = MagicMock()

        async def mock_prepare(*args, **kwargs):
            return {"available_tools": [], "default_prompts": {}}

        mock_context_mod.prepare_runtime_context = mock_prepare

        mock_langgraph_types = MagicMock()

        cleanup = self._patch_resume_modules({
            "isa_agent_sdk.graphs.smart_agent_graph": broken_mod,
            "isa_agent_sdk.graphs.utils.context_init": mock_context_mod,
            "langgraph.types": mock_langgraph_types,
        })

        try:
            messages = []
            async for msg in resume("session_2"):
                messages.append(msg)

            # When no resume_value, input should be None
            assert captured_input.get("input") is None
        finally:
            cleanup()
