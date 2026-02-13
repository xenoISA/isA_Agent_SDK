#!/usr/bin/env python3
"""
Agent Config Persistence

Provides async CRUD for agent configs and multi-agent specs using
isa_common.AsyncPostgresClient (native asyncpg).
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from isa_common import AsyncPostgresClient
from isa_agent_sdk.core.config import settings

logger = logging.getLogger(__name__)


class AgentVisibility(str, Enum):
    PRIVATE = "private"
    ORG = "org"


class MultiAgentRoutingType(str, Enum):
    DAG = "dag"
    LLM = "llm"


class MultiAgentExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class AgentConfigRecord(BaseModel):
    id: str
    owner_id: str
    org_id: Optional[str] = None
    visibility: AgentVisibility = AgentVisibility.PRIVATE
    name: str
    description: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    template_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class MultiAgentSpecRecord(BaseModel):
    id: str
    owner_id: str
    org_id: Optional[str] = None
    visibility: AgentVisibility = AgentVisibility.PRIVATE
    name: str
    description: Optional[str] = None
    entry_agent_id: str
    routing_type: MultiAgentRoutingType = MultiAgentRoutingType.DAG
    execution_mode: MultiAgentExecutionMode = MultiAgentExecutionMode.SEQUENTIAL
    routing_spec: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class AgentConfigStore:
    """Persistent store for agent configs + multi-agent specs."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        schema: Optional[str] = None,
    ):
        infra = settings.infrastructure
        self.schema = schema or settings.resources.postgres.schema or "agent"

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.schema):
            raise ValueError(f"Invalid schema name: {self.schema!r}")

        self.db = AsyncPostgresClient(
            host=host or infra.postgres_host,
            port=port or infra.postgres_port,
            database=database or infra.postgres_db,
            username=username or infra.postgres_user,
            password=password or infra.postgres_password,
            user_id="agent-config-service",
        )

        self._fallback_to_memory = False
        self._memory_configs: Dict[str, AgentConfigRecord] = {}
        self._memory_multi_specs: Dict[str, MultiAgentSpecRecord] = {}

    async def setup(self) -> bool:
        """Ensure schema and tables exist."""
        try:
            async with self.db:
                await self.db.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {self.schema}",
                    schema="public",
                )

                await self.db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.agent_configs (
                        id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        org_id TEXT NULL,
                        visibility TEXT NOT NULL DEFAULT 'private'
                            CHECK (visibility IN ('private', 'org')),
                        name TEXT NOT NULL,
                        description TEXT NULL,
                        options JSONB NOT NULL,
                        source TEXT NULL,
                        template_id TEXT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """,
                    schema="public",
                )

                await self.db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_agent_configs_owner ON {self.schema}.agent_configs(owner_id)",
                    schema="public",
                )
                await self.db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_agent_configs_org_vis ON {self.schema}.agent_configs(org_id, visibility)",
                    schema="public",
                )

                await self.db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.multi_agent_specs (
                        id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        org_id TEXT NULL,
                        visibility TEXT NOT NULL DEFAULT 'private'
                            CHECK (visibility IN ('private', 'org')),
                        name TEXT NOT NULL,
                        description TEXT NULL,
                        entry_agent_id TEXT NOT NULL,
                        routing_type TEXT NOT NULL DEFAULT 'dag'
                            CHECK (routing_type IN ('dag', 'llm')),
                        execution_mode TEXT NOT NULL DEFAULT 'sequential'
                            CHECK (execution_mode IN ('sequential', 'parallel')),
                        routing_spec JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """,
                    schema="public",
                )

                await self.db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_multi_specs_owner ON {self.schema}.multi_agent_specs(owner_id)",
                    schema="public",
                )
                await self.db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_multi_specs_org_vis ON {self.schema}.multi_agent_specs(org_id, visibility)",
                    schema="public",
                )

            return True
        except Exception as e:
            logger.warning(f"AgentConfigStore setup failed, using memory fallback: {e}")
            self._fallback_to_memory = True
            return False

    async def close(self) -> None:
        await self.db.close()

    def _new_id(self) -> str:
        return f"agent_{uuid.uuid4().hex[:12]}"

    def _new_multi_id(self) -> str:
        return f"multi_{uuid.uuid4().hex[:12]}"

    # ==========================================
    # Agent Config CRUD
    # ==========================================

    async def create_config(self, record: AgentConfigRecord) -> AgentConfigRecord:
        if self._fallback_to_memory:
            self._memory_configs[record.id] = record
            return record

        sql = f"""
            INSERT INTO {self.schema}.agent_configs (
                id, owner_id, org_id, visibility, name, description, options,
                source, template_id, metadata, created_at, updated_at
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12
            ) RETURNING *
        """
        params = [
            record.id,
            record.owner_id,
            record.org_id,
            record.visibility.value,
            record.name,
            record.description,
            record.options,
            record.source,
            record.template_id,
            record.metadata,
            record.created_at,
            record.updated_at,
        ]

        async with self.db:
            row = await self.db.query_row(sql, params=params, schema=self.schema)
        if not row:
            logger.error("create_config: INSERT returned no row for id=%s — DB write failed", record.id)
            raise RuntimeError(f"Failed to persist agent config {record.id} to database")
        return AgentConfigRecord(**row)

    async def get_config(self, config_id: str) -> Optional[AgentConfigRecord]:
        if self._fallback_to_memory:
            return self._memory_configs.get(config_id)

        async with self.db:
            row = await self.db.query_row(
                f"SELECT * FROM {self.schema}.agent_configs WHERE id = $1",
                params=[config_id],
                schema=self.schema,
            )
        return AgentConfigRecord(**row) if row else None

    async def list_configs(
        self,
        owner_id: Optional[str],
        org_ids: Optional[List[str]] = None,
        include_org_shared: bool = True,
        limit: int = 200,
        offset: int = 0,
    ) -> List[AgentConfigRecord]:
        if self._fallback_to_memory:
            values = list(self._memory_configs.values())
            if owner_id:
                values = [v for v in values if v.owner_id == owner_id]
                if include_org_shared and org_ids:
                    org_configs = [
                        v for v in self._memory_configs.values()
                        if v.visibility == AgentVisibility.ORG and v.org_id in org_ids
                    ]
                    values = values + org_configs
                    # Deduplicate by ID (owner config may also match org query)
                    seen: set[str] = set()
                    deduped: List[AgentConfigRecord] = []
                    for v in values:
                        if v.id not in seen:
                            seen.add(v.id)
                            deduped.append(v)
                    values = deduped
            return values

        if owner_id and include_org_shared and org_ids:
            sql = (
                f"SELECT * FROM {self.schema}.agent_configs "
                "WHERE owner_id = $1 OR (visibility = 'org' AND org_id = ANY($2)) "
                "ORDER BY updated_at DESC LIMIT $3 OFFSET $4"
            )
            params = [owner_id, org_ids, limit, offset]
        elif owner_id:
            sql = (
                f"SELECT * FROM {self.schema}.agent_configs "
                "WHERE owner_id = $1 "
                "ORDER BY updated_at DESC LIMIT $2 OFFSET $3"
            )
            params = [owner_id, limit, offset]
        else:
            sql = (
                f"SELECT * FROM {self.schema}.agent_configs "
                "ORDER BY updated_at DESC LIMIT $1 OFFSET $2"
            )
            params = [limit, offset]

        async with self.db:
            rows = await self.db.query(sql, params=params, schema=self.schema)
        return [AgentConfigRecord(**row) for row in (rows or [])]

    _ALLOWED_UPDATE_COLUMNS = frozenset({
        "name", "description", "visibility", "org_id",
        "source", "template_id", "metadata", "options",
    })

    async def update_config(
        self,
        config_id: str,
        owner_id: str,
        updates: Dict[str, Any],
    ) -> Optional[AgentConfigRecord]:
        invalid = set(updates.keys()) - self._ALLOWED_UPDATE_COLUMNS
        if invalid:
            raise ValueError(f"Invalid update fields: {invalid}")

        if self._fallback_to_memory:
            existing = self._memory_configs.get(config_id)
            if not existing or existing.owner_id != owner_id:
                return None
            data = existing.model_dump()
            data.update(updates)
            data["updated_at"] = datetime.utcnow()
            updated = AgentConfigRecord(**data)
            self._memory_configs[config_id] = updated
            return updated

        if not updates:
            return await self.get_config(config_id)

        set_clauses = []
        params: List[Any] = []
        idx = 1
        for key, value in updates.items():
            set_clauses.append(f"{key} = ${idx}")
            params.append(value)
            idx += 1
        set_clauses.append("updated_at = NOW()")

        sql = (
            f"UPDATE {self.schema}.agent_configs SET {', '.join(set_clauses)} "
            f"WHERE id = ${idx} AND owner_id = ${idx + 1} RETURNING *"
        )
        params.extend([config_id, owner_id])

        async with self.db:
            row = await self.db.query_row(sql, params=params, schema=self.schema)
        return AgentConfigRecord(**row) if row else None

    async def delete_config(self, config_id: str, owner_id: str) -> bool:
        if self._fallback_to_memory:
            existing = self._memory_configs.get(config_id)
            if not existing or existing.owner_id != owner_id:
                return False
            del self._memory_configs[config_id]
            return True

        sql = (
            f"DELETE FROM {self.schema}.agent_configs "
            "WHERE id = $1 AND owner_id = $2"
        )
        async with self.db:
            affected = await self.db.execute(sql, params=[config_id, owner_id], schema=self.schema)
        return bool(affected)

    # ==========================================
    # Multi-Agent Spec CRUD (future use)
    # ==========================================

    async def create_multi_spec(self, record: MultiAgentSpecRecord) -> MultiAgentSpecRecord:
        if self._fallback_to_memory:
            self._memory_multi_specs[record.id] = record
            return record

        sql = f"""
            INSERT INTO {self.schema}.multi_agent_specs (
                id, owner_id, org_id, visibility, name, description, entry_agent_id,
                routing_type, execution_mode, routing_spec, created_at, updated_at
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12
            ) RETURNING *
        """
        params = [
            record.id,
            record.owner_id,
            record.org_id,
            record.visibility.value,
            record.name,
            record.description,
            record.entry_agent_id,
            record.routing_type.value,
            record.execution_mode.value,
            record.routing_spec,
            record.created_at,
            record.updated_at,
        ]
        async with self.db:
            row = await self.db.query_row(sql, params=params, schema=self.schema)
        if not row:
            logger.error("create_multi_spec: INSERT returned no row for id=%s — DB write failed", record.id)
            raise RuntimeError(f"Failed to persist multi-agent spec {record.id} to database")
        return MultiAgentSpecRecord(**row)

    async def get_multi_spec(self, spec_id: str) -> Optional[MultiAgentSpecRecord]:
        if self._fallback_to_memory:
            return self._memory_multi_specs.get(spec_id)

        async with self.db:
            row = await self.db.query_row(
                f"SELECT * FROM {self.schema}.multi_agent_specs WHERE id = $1",
                params=[spec_id],
                schema=self.schema,
            )
        return MultiAgentSpecRecord(**row) if row else None

    async def list_multi_specs(
        self,
        owner_id: Optional[str],
        org_ids: Optional[List[str]] = None,
        include_org_shared: bool = True,
        limit: int = 200,
        offset: int = 0,
    ) -> List[MultiAgentSpecRecord]:
        if self._fallback_to_memory:
            values = list(self._memory_multi_specs.values())
            if owner_id:
                values = [v for v in values if v.owner_id == owner_id]
                if include_org_shared and org_ids:
                    org_specs = [
                        v for v in self._memory_multi_specs.values()
                        if v.visibility == AgentVisibility.ORG and v.org_id in org_ids
                    ]
                    values = values + org_specs
                    # Deduplicate by ID (owner spec may also match org query)
                    seen: set[str] = set()
                    deduped: List[MultiAgentSpecRecord] = []
                    for v in values:
                        if v.id not in seen:
                            seen.add(v.id)
                            deduped.append(v)
                    values = deduped
            return values

        if owner_id and include_org_shared and org_ids:
            sql = (
                f"SELECT * FROM {self.schema}.multi_agent_specs "
                "WHERE owner_id = $1 OR (visibility = 'org' AND org_id = ANY($2)) "
                "ORDER BY updated_at DESC LIMIT $3 OFFSET $4"
            )
            params = [owner_id, org_ids, limit, offset]
        elif owner_id:
            sql = (
                f"SELECT * FROM {self.schema}.multi_agent_specs "
                "WHERE owner_id = $1 "
                "ORDER BY updated_at DESC LIMIT $2 OFFSET $3"
            )
            params = [owner_id, limit, offset]
        else:
            sql = (
                f"SELECT * FROM {self.schema}.multi_agent_specs "
                "ORDER BY updated_at DESC LIMIT $1 OFFSET $2"
            )
            params = [limit, offset]

        async with self.db:
            rows = await self.db.query(sql, params=params, schema=self.schema)
        return [MultiAgentSpecRecord(**row) for row in (rows or [])]

    _ALLOWED_MULTI_SPEC_UPDATE_COLUMNS = frozenset({
        "name", "description", "entry_agent_id", "routing_type",
        "execution_mode", "routing_spec", "visibility", "org_id",
    })

    async def update_multi_spec(
        self,
        spec_id: str,
        owner_id: str,
        updates: Dict[str, Any],
    ) -> Optional[MultiAgentSpecRecord]:
        invalid = set(updates.keys()) - self._ALLOWED_MULTI_SPEC_UPDATE_COLUMNS
        if invalid:
            raise ValueError(f"Invalid update fields: {invalid}")

        if self._fallback_to_memory:
            existing = self._memory_multi_specs.get(spec_id)
            if not existing or existing.owner_id != owner_id:
                return None
            data = existing.model_dump()
            data.update(updates)
            data["updated_at"] = datetime.utcnow()
            updated = MultiAgentSpecRecord(**data)
            self._memory_multi_specs[spec_id] = updated
            return updated

        if not updates:
            return await self.get_multi_spec(spec_id)

        set_clauses = []
        params: List[Any] = []
        idx = 1
        for key, value in updates.items():
            set_clauses.append(f"{key} = ${idx}")
            params.append(value)
            idx += 1
        set_clauses.append("updated_at = NOW()")

        sql = (
            f"UPDATE {self.schema}.multi_agent_specs SET {', '.join(set_clauses)} "
            f"WHERE id = ${idx} AND owner_id = ${idx + 1} RETURNING *"
        )
        params.extend([spec_id, owner_id])

        async with self.db:
            row = await self.db.query_row(sql, params=params, schema=self.schema)
        return MultiAgentSpecRecord(**row) if row else None

    async def delete_multi_spec(self, spec_id: str, owner_id: str) -> bool:
        if self._fallback_to_memory:
            existing = self._memory_multi_specs.get(spec_id)
            if not existing or existing.owner_id != owner_id:
                return False
            del self._memory_multi_specs[spec_id]
            return True

        sql = (
            f"DELETE FROM {self.schema}.multi_agent_specs "
            "WHERE id = $1 AND owner_id = $2"
        )
        async with self.db:
            affected = await self.db.execute(sql, params=[spec_id, owner_id], schema=self.schema)
        return bool(affected)

    async def new_config_id(self) -> str:
        return self._new_id()

    async def new_multi_id(self) -> str:
        return self._new_multi_id()
