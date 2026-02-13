"""
Scenario 3: Multi-Agent Runner tests.

Tests the entry_agent override, DAG vs swarm routing, and orchestrator construction.
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from isa_agent_sdk.services.multi_agent_runner import _build_orchestrator
from isa_agent_sdk.services.agent_config_store import (
    AgentConfigStore,
    MultiAgentSpecRecord,
    MultiAgentRoutingType,
    MultiAgentExecutionMode,
    AgentVisibility,
)
from isa_agent_sdk.services.multi_agent_runner import MultiAgentRunner


# ---------------------------------------------------------------------------
# _build_orchestrator tests
# ---------------------------------------------------------------------------

class TestBuildOrchestrator:
    def test_entry_agent_override_wins(self):
        routing_spec = {
            "agents": [
                {"name": "researcher", "system_prompt": "Research."},
                {"name": "writer", "system_prompt": "Write."},
            ],
            "entry_agent": "researcher",  # This should be overridden
        }
        orchestrator, _, _, _ = _build_orchestrator(
            routing_spec, "user-1",
            entry_agent_override="writer",
        )
        assert orchestrator.entry_agent == "writer"

    def test_falls_back_to_routing_spec_entry_agent_id(self):
        routing_spec = {
            "agents": [
                {"name": "a", "system_prompt": "A."},
                {"name": "b", "system_prompt": "B."},
            ],
            "entry_agent_id": "b",
        }
        orchestrator, _, _, _ = _build_orchestrator(routing_spec, "user-1")
        assert orchestrator.entry_agent == "b"

    def test_falls_back_to_first_agent(self):
        routing_spec = {
            "agents": [
                {"name": "first", "system_prompt": "First."},
                {"name": "second", "system_prompt": "Second."},
            ],
        }
        orchestrator, _, _, _ = _build_orchestrator(routing_spec, "user-1")
        assert orchestrator.entry_agent == "first"

    def test_missing_agents_raises(self):
        with pytest.raises(ValueError, match="missing agents"):
            _build_orchestrator({"agents": []}, "user-1")

    def test_routing_type_dag(self):
        routing_spec = {
            "agents": [{"name": "a", "system_prompt": "A."}],
            "routing_type": "dag",
            "tasks": [{"id": "t1", "agent": "a"}],
        }
        _, tasks, routing_type, _ = _build_orchestrator(routing_spec, "user-1")
        assert routing_type == MultiAgentRoutingType.DAG
        assert len(tasks) == 1

    def test_execution_mode_parallel(self):
        routing_spec = {
            "agents": [{"name": "a", "system_prompt": "A."}],
            "execution_mode": "parallel",
        }
        _, _, _, execution_mode = _build_orchestrator(routing_spec, "user-1")
        assert execution_mode == MultiAgentExecutionMode.PARALLEL

    def test_invalid_routing_type_defaults_to_dag(self):
        routing_spec = {
            "agents": [{"name": "a", "system_prompt": "A."}],
            "routing_type": "invalid_type",
        }
        _, _, routing_type, _ = _build_orchestrator(routing_spec, "user-1")
        assert routing_type == MultiAgentRoutingType.DAG


# ---------------------------------------------------------------------------
# MultiAgentRunner integration tests (with mock store)
# ---------------------------------------------------------------------------

def _make_store_with_spec(spec: MultiAgentSpecRecord) -> AgentConfigStore:
    store = object.__new__(AgentConfigStore)
    store._fallback_to_memory = True
    store._memory_configs = {}
    store._memory_multi_specs = {spec.id: spec}
    return store


class TestMultiAgentRunnerNotFound:
    @pytest.mark.asyncio
    async def test_run_nonexistent_spec_raises(self):
        store = object.__new__(AgentConfigStore)
        store._fallback_to_memory = True
        store._memory_configs = {}
        store._memory_multi_specs = {}
        runner = MultiAgentRunner(store)

        with pytest.raises(ValueError, match="not found"):
            await runner.run(
                multi_agent_id="nonexistent",
                prompt="test",
                user_id="user-1",
            )

    @pytest.mark.asyncio
    async def test_stream_nonexistent_spec_raises(self):
        store = object.__new__(AgentConfigStore)
        store._fallback_to_memory = True
        store._memory_configs = {}
        store._memory_multi_specs = {}
        runner = MultiAgentRunner(store)

        with pytest.raises(ValueError, match="not found"):
            async for _ in runner.stream(
                multi_agent_id="nonexistent",
                prompt="test",
                user_id="user-1",
            ):
                pass
