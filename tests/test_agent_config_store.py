"""
Tests for AgentConfigStore — column whitelist and memory fallback deduplication.
"""

import pytest
from datetime import datetime

from isa_agent_sdk.services.agent_config_store import (
    AgentConfigStore,
    AgentConfigRecord,
    AgentVisibility,
    MultiAgentSpecRecord,
    MultiAgentRoutingType,
    MultiAgentExecutionMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> AgentConfigStore:
    """Create an AgentConfigStore in memory-fallback mode (no Postgres)."""
    store = object.__new__(AgentConfigStore)
    store._fallback_to_memory = True
    store._memory_configs = {}
    store._memory_multi_specs = {}
    return store


def _make_record(
    id: str,
    owner_id: str = "user-1",
    org_id: str | None = None,
    visibility: AgentVisibility = AgentVisibility.PRIVATE,
) -> AgentConfigRecord:
    now = datetime.utcnow()
    return AgentConfigRecord(
        id=id,
        owner_id=owner_id,
        org_id=org_id,
        visibility=visibility,
        name=f"Agent {id}",
        description="test",
        options={"model": "gpt-5-nano"},
        source="basic",
        metadata={},
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Fix 0A: update_config column whitelist
# ---------------------------------------------------------------------------

class TestUpdateConfigWhitelist:
    @pytest.mark.asyncio
    async def test_rejects_invalid_columns(self):
        store = _make_store()
        record = _make_record("cfg-1")
        store._memory_configs["cfg-1"] = record

        with pytest.raises(ValueError, match="Invalid update fields"):
            await store.update_config("cfg-1", "user-1", {"name; DROP TABLE agent_configs; --": "x"})

    @pytest.mark.asyncio
    async def test_rejects_unknown_column(self):
        store = _make_store()
        record = _make_record("cfg-1")
        store._memory_configs["cfg-1"] = record

        with pytest.raises(ValueError, match="Invalid update fields"):
            await store.update_config("cfg-1", "user-1", {"unknown_field": "value"})

    @pytest.mark.asyncio
    async def test_allows_valid_columns(self):
        store = _make_store()
        record = _make_record("cfg-1")
        store._memory_configs["cfg-1"] = record

        result = await store.update_config("cfg-1", "user-1", {"name": "Updated Name"})
        assert result is not None
        assert result.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_allows_multiple_valid_columns(self):
        store = _make_store()
        record = _make_record("cfg-1")
        store._memory_configs["cfg-1"] = record

        result = await store.update_config("cfg-1", "user-1", {
            "name": "New Name",
            "description": "New Description",
        })
        assert result is not None
        assert result.name == "New Name"
        assert result.description == "New Description"

    @pytest.mark.asyncio
    async def test_rejects_mix_valid_and_invalid(self):
        store = _make_store()
        record = _make_record("cfg-1")
        store._memory_configs["cfg-1"] = record

        with pytest.raises(ValueError, match="Invalid update fields"):
            await store.update_config("cfg-1", "user-1", {
                "name": "ok",
                "evil_column": "bad",
            })

    @pytest.mark.asyncio
    async def test_wrong_owner_returns_none(self):
        store = _make_store()
        record = _make_record("cfg-1", owner_id="user-1")
        store._memory_configs["cfg-1"] = record

        result = await store.update_config("cfg-1", "user-2", {"name": "Nope"})
        assert result is None


# ---------------------------------------------------------------------------
# Fix 0B: list_configs memory fallback deduplication
# ---------------------------------------------------------------------------

class TestListConfigsDedup:
    @pytest.mark.asyncio
    async def test_no_duplicates_when_owner_matches_org(self):
        """A config owned by the user with ORG visibility should appear once."""
        store = _make_store()
        record = _make_record(
            "cfg-1",
            owner_id="user-1",
            org_id="org-1",
            visibility=AgentVisibility.ORG,
        )
        store._memory_configs["cfg-1"] = record

        results = await store.list_configs(
            owner_id="user-1",
            org_ids=["org-1"],
            include_org_shared=True,
        )
        # Should appear exactly once, not twice
        ids = [r.id for r in results]
        assert ids.count("cfg-1") == 1

    @pytest.mark.asyncio
    async def test_private_and_org_configs_combined(self):
        """Private + org configs from different owners should all appear."""
        store = _make_store()

        private = _make_record("cfg-private", owner_id="user-1")
        org_shared = _make_record(
            "cfg-org",
            owner_id="user-2",
            org_id="org-1",
            visibility=AgentVisibility.ORG,
        )
        store._memory_configs["cfg-private"] = private
        store._memory_configs["cfg-org"] = org_shared

        results = await store.list_configs(
            owner_id="user-1",
            org_ids=["org-1"],
            include_org_shared=True,
        )
        ids = [r.id for r in results]
        assert "cfg-private" in ids
        assert "cfg-org" in ids
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_no_org_shared_when_disabled(self):
        """With include_org_shared=False, only owner configs returned."""
        store = _make_store()

        own = _make_record("cfg-own", owner_id="user-1")
        org = _make_record(
            "cfg-org",
            owner_id="user-2",
            org_id="org-1",
            visibility=AgentVisibility.ORG,
        )
        store._memory_configs["cfg-own"] = own
        store._memory_configs["cfg-org"] = org

        results = await store.list_configs(
            owner_id="user-1",
            org_ids=["org-1"],
            include_org_shared=False,
        )
        ids = [r.id for r in results]
        assert ids == ["cfg-own"]


# ---------------------------------------------------------------------------
# Fix 0B-ext: list_multi_specs memory fallback deduplication
# ---------------------------------------------------------------------------

def _make_multi_spec(
    id: str,
    owner_id: str = "user-1",
    org_id: str | None = None,
    visibility: AgentVisibility = AgentVisibility.PRIVATE,
) -> MultiAgentSpecRecord:
    now = datetime.utcnow()
    return MultiAgentSpecRecord(
        id=id,
        owner_id=owner_id,
        org_id=org_id,
        visibility=visibility,
        name=f"Spec {id}",
        entry_agent_id="agent-a",
        routing_type=MultiAgentRoutingType.DAG,
        execution_mode=MultiAgentExecutionMode.SEQUENTIAL,
        routing_spec={"agents": []},
        created_at=now,
        updated_at=now,
    )


class TestListMultiSpecsDedup:
    @pytest.mark.asyncio
    async def test_no_duplicates_when_owner_matches_org(self):
        """A multi-spec owned by the user with ORG visibility should appear once."""
        store = _make_store()
        spec = _make_multi_spec(
            "ms-1",
            owner_id="user-1",
            org_id="org-1",
            visibility=AgentVisibility.ORG,
        )
        store._memory_multi_specs["ms-1"] = spec

        results = await store.list_multi_specs(
            owner_id="user-1",
            org_ids=["org-1"],
            include_org_shared=True,
        )
        ids = [r.id for r in results]
        assert ids.count("ms-1") == 1

    @pytest.mark.asyncio
    async def test_private_and_org_specs_combined(self):
        """Private + org specs from different owners should all appear."""
        store = _make_store()

        private = _make_multi_spec("ms-private", owner_id="user-1")
        org_shared = _make_multi_spec(
            "ms-org",
            owner_id="user-2",
            org_id="org-1",
            visibility=AgentVisibility.ORG,
        )
        store._memory_multi_specs["ms-private"] = private
        store._memory_multi_specs["ms-org"] = org_shared

        results = await store.list_multi_specs(
            owner_id="user-1",
            org_ids=["org-1"],
            include_org_shared=True,
        )
        ids = [r.id for r in results]
        assert "ms-private" in ids
        assert "ms-org" in ids
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_no_org_shared_when_disabled(self):
        """With include_org_shared=False, only owner specs returned."""
        store = _make_store()

        own = _make_multi_spec("ms-own", owner_id="user-1")
        org = _make_multi_spec(
            "ms-org",
            owner_id="user-2",
            org_id="org-1",
            visibility=AgentVisibility.ORG,
        )
        store._memory_multi_specs["ms-own"] = own
        store._memory_multi_specs["ms-org"] = org

        results = await store.list_multi_specs(
            owner_id="user-1",
            org_ids=["org-1"],
            include_org_shared=False,
        )
        ids = [r.id for r in results]
        assert ids == ["ms-own"]


# ---------------------------------------------------------------------------
# update_multi_spec column whitelist + memory fallback
# ---------------------------------------------------------------------------

class TestUpdateMultiSpec:
    @pytest.mark.asyncio
    async def test_allows_valid_columns(self):
        store = _make_store()
        spec = _make_multi_spec("ms-1", owner_id="user-1")
        store._memory_multi_specs["ms-1"] = spec

        result = await store.update_multi_spec("ms-1", "user-1", {"name": "Renamed"})
        assert result is not None
        assert result.name == "Renamed"

    @pytest.mark.asyncio
    async def test_rejects_invalid_columns(self):
        store = _make_store()
        spec = _make_multi_spec("ms-1", owner_id="user-1")
        store._memory_multi_specs["ms-1"] = spec

        with pytest.raises(ValueError, match="Invalid update fields"):
            await store.update_multi_spec("ms-1", "user-1", {"owner_id": "hacker"})

    @pytest.mark.asyncio
    async def test_wrong_owner_returns_none(self):
        store = _make_store()
        spec = _make_multi_spec("ms-1", owner_id="user-1")
        store._memory_multi_specs["ms-1"] = spec

        result = await store.update_multi_spec("ms-1", "user-2", {"name": "Nope"})
        assert result is None

    @pytest.mark.asyncio
    async def test_updates_multiple_fields(self):
        store = _make_store()
        spec = _make_multi_spec("ms-1", owner_id="user-1")
        store._memory_multi_specs["ms-1"] = spec

        result = await store.update_multi_spec("ms-1", "user-1", {
            "name": "New Name",
            "description": "New desc",
            "entry_agent_id": "agent-b",
        })
        assert result is not None
        assert result.name == "New Name"
        assert result.description == "New desc"
        assert result.entry_agent_id == "agent-b"
