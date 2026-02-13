"""
Scenario 3: Generated Agent (Vibe) + Multi-Agent tests.

Tests create_generated flow, entry_agent inference, and multi-agent spec creation.
"""

import pytest
from unittest.mock import AsyncMock

from isa_agent_sdk.services.agent_config_store import (
    AgentConfigStore,
    AgentVisibility,
)
from isa_agent_sdk.services.agent_creation import AgentCreationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> AgentConfigStore:
    store = object.__new__(AgentConfigStore)
    store._fallback_to_memory = True
    store._memory_configs = {}
    store._memory_multi_specs = {}
    return store


def _make_service(store=None) -> AgentCreationService:
    if store is None:
        store = _make_store()
    svc = AgentCreationService(store)
    store.new_config_id = AsyncMock(side_effect=lambda: f"cfg-{len(store._memory_configs) + 1}")
    store.new_multi_id = AsyncMock(side_effect=lambda: f"multi-{len(store._memory_multi_specs) + 1}")
    return svc


# ---------------------------------------------------------------------------
# Single agent (no multi-agent spec)
# ---------------------------------------------------------------------------

class TestCreateGeneratedSingle:
    @pytest.mark.asyncio
    async def test_single_agent_no_multi_spec(self):
        svc = _make_service()
        payload = {
            "name": "Simple Agent",
            "model_id": "gpt-5-nano",
            "system_prompt": "You help.",
        }
        record, multi_spec = await svc.create_generated(
            owner_id="user-1",
            generated_payload=payload,
        )
        assert record.name == "Simple Agent"
        assert record.source == "generated"
        assert multi_spec is None
        assert "multi_agent_id" not in record.metadata


# ---------------------------------------------------------------------------
# Multi-agent spec creation
# ---------------------------------------------------------------------------

class TestCreateGeneratedMultiAgent:
    @pytest.mark.asyncio
    async def test_creates_multi_spec_from_payload(self):
        svc = _make_service()
        payload = {
            "name": "Research Team",
            "model_id": "gpt-5-nano",
            "multi_agent": {
                "name": "Research Pipeline",
                "agents": [
                    {"name": "researcher", "system_prompt": "Research."},
                    {"name": "writer", "system_prompt": "Write."},
                ],
                "tasks": [
                    {"id": "research", "agent": "researcher"},
                    {"id": "write", "agent": "writer", "depends_on": ["research"]},
                ],
            },
        }
        record, multi_spec = await svc.create_generated(
            owner_id="user-1",
            generated_payload=payload,
        )
        assert multi_spec is not None
        assert multi_spec.name == "Research Pipeline"
        assert record.metadata["multi_agent_id"] == multi_spec.id

    @pytest.mark.asyncio
    async def test_infers_entry_agent_from_first_agent(self):
        svc = _make_service()
        payload = {
            "name": "Team",
            "model_id": "gpt-5-nano",
            "multi_agent": {
                "agents": [
                    {"name": "researcher", "system_prompt": "Research."},
                    {"name": "writer", "system_prompt": "Write."},
                ],
            },
        }
        _, multi_spec = await svc.create_generated(
            owner_id="user-1",
            generated_payload=payload,
        )
        assert multi_spec is not None
        assert multi_spec.entry_agent_id == "researcher"

    @pytest.mark.asyncio
    async def test_explicit_entry_agent_id(self):
        svc = _make_service()
        payload = {
            "name": "Team",
            "model_id": "gpt-5-nano",
            "multi_agent": {
                "entry_agent_id": "writer",
                "agents": [
                    {"name": "researcher", "system_prompt": "Research."},
                    {"name": "writer", "system_prompt": "Write."},
                ],
            },
        }
        _, multi_spec = await svc.create_generated(
            owner_id="user-1",
            generated_payload=payload,
        )
        assert multi_spec is not None
        assert multi_spec.entry_agent_id == "writer"

    @pytest.mark.asyncio
    async def test_no_fallback_to_literal_entry(self):
        """When agents list is empty and no entry_agent key, should raise ValueError."""
        svc = _make_service()
        payload = {
            "name": "Bad Team",
            "model_id": "gpt-5-nano",
            "multi_agent": {
                "agents": [],
                "tasks": [{"id": "t1"}],
            },
        }
        with pytest.raises(ValueError, match="Cannot infer entry_agent_id"):
            await svc.create_generated(
                owner_id="user-1",
                generated_payload=payload,
            )

    @pytest.mark.asyncio
    async def test_entry_agent_from_id_field(self):
        """Agents with 'id' instead of 'name' should work."""
        svc = _make_service()
        payload = {
            "name": "Team",
            "model_id": "gpt-5-nano",
            "multi_agent": {
                "agents": [
                    {"id": "agent-a", "system_prompt": "A."},
                ],
            },
        }
        _, multi_spec = await svc.create_generated(
            owner_id="user-1",
            generated_payload=payload,
        )
        assert multi_spec is not None
        assert multi_spec.entry_agent_id == "agent-a"

    @pytest.mark.asyncio
    async def test_agents_and_tasks_at_top_level(self):
        """When agents/tasks are at top level (not in multi_agent key)."""
        svc = _make_service()
        payload = {
            "name": "Team",
            "model_id": "gpt-5-nano",
            "agents": [
                {"name": "planner", "system_prompt": "Plan."},
            ],
            "tasks": [
                {"id": "plan", "agent": "planner"},
            ],
        }
        _, multi_spec = await svc.create_generated(
            owner_id="user-1",
            generated_payload=payload,
        )
        assert multi_spec is not None
        assert multi_spec.entry_agent_id == "planner"
