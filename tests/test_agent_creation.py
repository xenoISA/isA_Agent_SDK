"""
Tests for AgentCreationService — all 4 creation paths.

Scenario 1: Basic Agent
Scenario 2: Template Agent
Scenario 4: Local Dev Agent
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from isa_agent_sdk.services.agent_config_store import (
    AgentConfigStore,
    AgentConfigRecord,
    AgentVisibility,
    MultiAgentSpecRecord,
)
from isa_agent_sdk.services.agent_creation import (
    AgentCreationService,
    _build_options,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> AgentConfigStore:
    """Create an AgentConfigStore in memory-fallback mode."""
    store = object.__new__(AgentConfigStore)
    store._fallback_to_memory = True
    store._memory_configs = {}
    store._memory_multi_specs = {}
    return store


def _make_service(store=None) -> AgentCreationService:
    if store is None:
        store = _make_store()
    svc = AgentCreationService(store)
    # Mock ID generators to return predictable IDs
    store.new_config_id = AsyncMock(side_effect=lambda: f"cfg-{len(store._memory_configs) + 1}")
    store.new_multi_id = AsyncMock(side_effect=lambda: f"multi-{len(store._memory_multi_specs) + 1}")
    return svc


# ---------------------------------------------------------------------------
# Scenario 1: Basic Agent
# ---------------------------------------------------------------------------

class TestCreateBasic:
    @pytest.mark.asyncio
    async def test_creates_config_with_correct_fields(self):
        svc = _make_service()
        record = await svc.create_basic(
            owner_id="user-1",
            name="My Agent",
            description="A helpful assistant",
            model_id="claude-sonnet-4-5-20250929",
            system_prompt="You are helpful.",
            tools=["web_search"],
            skills=["coding"],
            mode="REACTIVE",
            env="cloud_pool",
        )
        assert record.id == "cfg-1"
        assert record.owner_id == "user-1"
        assert record.name == "My Agent"
        assert record.description == "A helpful assistant"
        assert record.source == "basic"
        assert record.visibility == AgentVisibility.PRIVATE
        assert record.options["model"] == "claude-sonnet-4-5-20250929"
        assert record.options["system_prompt"] == "You are helpful."
        assert record.options["allowed_tools"] == ["web_search"]
        assert record.options["skills"] == ["coding"]
        assert record.options["execution_mode"] == "reactive"

    @pytest.mark.asyncio
    async def test_creates_config_with_org_visibility(self):
        svc = _make_service()
        record = await svc.create_basic(
            owner_id="user-1",
            name="Shared Agent",
            model_id="gpt-5-nano",
            visibility=AgentVisibility.ORG,
            org_id="org-1",
        )
        assert record.visibility == AgentVisibility.ORG
        assert record.org_id == "org-1"

    @pytest.mark.asyncio
    async def test_private_visibility_clears_org_id(self):
        svc = _make_service()
        record = await svc.create_basic(
            owner_id="user-1",
            name="Private Agent",
            model_id="gpt-5-nano",
            visibility=AgentVisibility.PRIVATE,
            org_id="org-1",  # Should be cleared for private
        )
        assert record.visibility == AgentVisibility.PRIVATE
        assert record.org_id is None

    @pytest.mark.asyncio
    async def test_graph_type_flows_into_options(self):
        svc = _make_service()
        record = await svc.create_basic(
            owner_id="user-1",
            name="Coding Agent",
            model_id="gpt-5-nano",
            graph_type="coding",
        )
        assert record.options["graph_type"] == "coding"

    @pytest.mark.asyncio
    async def test_config_is_persisted_in_store(self):
        store = _make_store()
        svc = _make_service(store)
        record = await svc.create_basic(
            owner_id="user-1",
            name="Test",
            model_id="gpt-5-nano",
        )
        # Verify it's in the store
        fetched = await store.get_config(record.id)
        assert fetched is not None
        assert fetched.name == "Test"


class TestBuildOptions:
    def test_normalizes_mode_to_lowercase(self):
        opts = _build_options(
            model_id="gpt-5-nano",
            system_prompt="hi",
            tools=["web_search"],
            skills=[],
            mode="REACTIVE",
            env="cloud_pool",
        )
        assert opts["execution_mode"] == "reactive"

    def test_defaults_tools_to_empty_list(self):
        opts = _build_options(
            model_id="gpt-5-nano",
            system_prompt="",
            tools=None,
            skills=None,
            mode="collaborative",
            env="cloud_pool",
        )
        assert opts["allowed_tools"] == []
        assert opts["skills"] == []

    def test_includes_graph_type(self):
        opts = _build_options(
            model_id="gpt-5-nano",
            system_prompt="",
            tools=[],
            skills=[],
            mode="reactive",
            env="cloud_pool",
        )
        assert opts["graph_type"] == "smart_agent"


# ---------------------------------------------------------------------------
# Scenario 2: Template Agent
# ---------------------------------------------------------------------------

class TestCreateFromTemplate:
    @pytest.mark.asyncio
    async def test_returns_tuple_of_record_and_none(self):
        """create_from_template returns (AgentConfigRecord, None) when no multi_agent."""
        svc = _make_service()
        template = {"id": "tpl-1", "name": "Simple", "system_prompt": "Hi."}
        result = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano"},
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        record, multi_spec = result
        assert isinstance(record, AgentConfigRecord)
        assert multi_spec is None

    @pytest.mark.asyncio
    async def test_merges_template_and_overrides(self):
        svc = _make_service()
        template = {
            "id": "tpl-qa",
            "name": "QA Agent",
            "description": "Quality assurance",
            "system_prompt": "You are a QA specialist.",
            "tools": ["web_search", "bash_execute"],
            "skills": ["testing"],
            "mode": "COLLABORATIVE",
        }
        overrides = {
            "model_id": "claude-sonnet-4-5-20250929",
            "name": "My QA Agent",  # Override template name
        }
        record, multi_spec = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides=overrides,
        )
        assert record.name == "My QA Agent"
        assert record.source == "template"
        assert record.template_id == "tpl-qa"
        assert record.options["model"] == "claude-sonnet-4-5-20250929"
        assert record.options["system_prompt"] == "You are a QA specialist."
        assert "web_search" in record.options["allowed_tools"]
        assert multi_spec is None

    @pytest.mark.asyncio
    async def test_preserves_template_id(self):
        svc = _make_service()
        template = {"id": "tpl-code", "name": "Coder", "system_prompt": "Code stuff."}
        record, _ = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano"},
        )
        assert record.template_id == "tpl-code"

    @pytest.mark.asyncio
    async def test_overrides_replace_template_values(self):
        svc = _make_service()
        template = {
            "id": "tpl-1",
            "name": "Template Name",
            "system_prompt": "Original prompt",
            "tools": ["old_tool"],
        }
        overrides = {
            "model_id": "gpt-5-nano",
            "system_prompt": "New prompt",
            "tools": ["new_tool"],
        }
        record, _ = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides=overrides,
        )
        assert record.options["system_prompt"] == "New prompt"
        assert record.options["allowed_tools"] == ["new_tool"]

    @pytest.mark.asyncio
    async def test_graph_type_from_template(self):
        """graph_type in the template flows into options."""
        svc = _make_service()
        template = {
            "id": "tpl-research",
            "name": "Research Agent",
            "system_prompt": "Research.",
            "graph_type": "research",
        }
        record, _ = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano"},
        )
        assert record.options["graph_type"] == "research"

    @pytest.mark.asyncio
    async def test_graph_type_override_beats_template(self):
        """Override graph_type takes precedence over the template's."""
        svc = _make_service()
        template = {
            "id": "tpl-1",
            "name": "Agent",
            "system_prompt": "Hi.",
            "graph_type": "conversation",
        }
        record, _ = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano", "graph_type": "coding"},
        )
        assert record.options["graph_type"] == "coding"

    @pytest.mark.asyncio
    async def test_graph_type_defaults_to_smart_agent(self):
        """When neither template nor overrides specify graph_type, defaults to smart_agent."""
        svc = _make_service()
        template = {"id": "tpl-1", "name": "Agent", "system_prompt": "Hi."}
        record, _ = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano"},
        )
        assert record.options["graph_type"] == "smart_agent"

    @pytest.mark.asyncio
    async def test_multi_agent_template_creates_spec(self):
        """Template with multi_agent key returns a MultiAgentSpecRecord."""
        svc = _make_service()
        template = {
            "id": "tpl-team",
            "name": "Team Agent",
            "system_prompt": "Coordinate.",
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
        record, multi_spec = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano"},
        )
        assert multi_spec is not None
        assert isinstance(multi_spec, MultiAgentSpecRecord)
        assert multi_spec.name == "Research Pipeline"
        assert multi_spec.entry_agent_id == "researcher"
        assert record.metadata["multi_agent_id"] == multi_spec.id

    @pytest.mark.asyncio
    async def test_multi_agent_override_suppresses_template_spec(self):
        """When overrides contain multi_agent, template's multi_agent is skipped."""
        svc = _make_service()
        template = {
            "id": "tpl-team",
            "name": "Team Agent",
            "system_prompt": "Coordinate.",
            "multi_agent": {
                "agents": [{"name": "a1", "system_prompt": "A."}],
            },
        }
        record, multi_spec = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano", "multi_agent": True},
        )
        assert multi_spec is None
        assert "multi_agent_id" not in record.metadata

    @pytest.mark.asyncio
    async def test_multi_agent_template_persists_spec_in_store(self):
        """Multi-agent spec created from template is persisted in the store."""
        store = _make_store()
        svc = _make_service(store)
        template = {
            "id": "tpl-team",
            "name": "Team Agent",
            "system_prompt": "Coordinate.",
            "multi_agent": {
                "entry_agent_id": "planner",
                "agents": [{"name": "planner", "system_prompt": "Plan."}],
            },
        }
        _, multi_spec = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano"},
        )
        assert multi_spec is not None
        fetched = await store.get_multi_spec(multi_spec.id)
        assert fetched is not None
        assert fetched.entry_agent_id == "planner"

    @pytest.mark.asyncio
    async def test_visibility_flows_to_multi_spec(self):
        """ORG visibility and org_id propagate to the multi-agent spec."""
        svc = _make_service()
        template = {
            "id": "tpl-team",
            "name": "Org Team",
            "system_prompt": "Team.",
            "multi_agent": {
                "agents": [{"name": "agent-1", "system_prompt": "Do stuff."}],
            },
        }
        record, multi_spec = await svc.create_from_template(
            owner_id="user-1",
            template=template,
            overrides={"model_id": "gpt-5-nano"},
            visibility=AgentVisibility.ORG,
            org_id="org-42",
        )
        assert multi_spec is not None
        assert multi_spec.visibility == AgentVisibility.ORG
        assert multi_spec.org_id == "org-42"
        assert record.visibility == AgentVisibility.ORG
        assert record.org_id == "org-42"


# ---------------------------------------------------------------------------
# Scenario 4: Local Dev Agent
# ---------------------------------------------------------------------------

class TestCreateLocalDev:
    @pytest.mark.asyncio
    async def test_stores_deployment_id_in_metadata(self):
        svc = _make_service()
        record = await svc.create_local_dev(
            owner_id="user-1",
            name="Local Agent",
            deployment_id="deploy-abc",
            model_id="gpt-5-nano",
        )
        assert record.source == "local"
        assert record.metadata["deployment_id"] == "deploy-abc"

    @pytest.mark.asyncio
    async def test_no_deployment_id(self):
        svc = _make_service()
        record = await svc.create_local_dev(
            owner_id="user-1",
            name="Local Agent",
            model_id="gpt-5-nano",
        )
        assert record.source == "local"
        assert "deployment_id" not in record.metadata

    @pytest.mark.asyncio
    async def test_custom_metadata_merged(self):
        svc = _make_service()
        record = await svc.create_local_dev(
            owner_id="user-1",
            name="Local Agent",
            model_id="gpt-5-nano",
            deployment_id="dep-1",
            metadata={"custom_key": "value"},
        )
        assert record.metadata["deployment_id"] == "dep-1"
        assert record.metadata["custom_key"] == "value"
