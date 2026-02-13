#!/usr/bin/env python3
"""
Agent Creation Service

Implements the four agent creation flows:
1) basic
2) template
3) generated (from isa_vibe payload)
4) local dev (scaffold/deployment flow)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .agent_config_store import (
    AgentConfigStore,
    AgentConfigRecord,
    AgentVisibility,
    MultiAgentSpecRecord,
    MultiAgentRoutingType,
    MultiAgentExecutionMode,
)


class AgentCreationService:
    """Create agent configs + multi-agent specs with consistent normalization."""

    def __init__(self, store: AgentConfigStore):
        self.store = store

    async def create_basic(
        self,
        *,
        owner_id: str,
        name: str,
        description: str = "",
        model_id: str,
        system_prompt: str = "",
        tools: Optional[list] = None,
        skills: Optional[list] = None,
        mode: str = "REACTIVE",
        env: str = "cloud_pool",
        graph_type: str = "smart_agent",
        visibility: AgentVisibility = AgentVisibility.PRIVATE,
        org_id: Optional[str] = None,
        source: str = "basic",
        template_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentConfigRecord:
        options = _build_options(
            model_id=model_id,
            system_prompt=system_prompt,
            tools=tools,
            skills=skills,
            mode=mode,
            env=env,
            graph_type=graph_type,
        )

        now = datetime.utcnow()
        record = AgentConfigRecord(
            id=await self.store.new_config_id(),
            owner_id=owner_id,
            org_id=org_id if visibility == AgentVisibility.ORG else None,
            visibility=visibility,
            name=name,
            description=description,
            options=options,
            source=source,
            template_id=template_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        return await self.store.create_config(record)

    async def create_from_template(
        self,
        *,
        owner_id: str,
        template: Dict[str, Any],
        overrides: Dict[str, Any],
        visibility: AgentVisibility = AgentVisibility.PRIVATE,
        org_id: Optional[str] = None,
        source: str = "template",
    ) -> Tuple[AgentConfigRecord, Optional[MultiAgentSpecRecord]]:
        graph_type = overrides.get("graph_type") or template.get("graph_type", "smart_agent")

        merged = {
            "name": template.get("name"),
            "description": template.get("description", ""),
            "model_id": overrides.get("model_id"),
            "system_prompt": template.get("system_prompt", ""),
            "tools": template.get("tools", []),
            "skills": template.get("skills", []),
            "mode": template.get("mode", "COLLABORATIVE"),
            "env": overrides.get("env", "cloud_pool"),
        }
        merged.update({k: v for k, v in overrides.items() if v is not None})

        metadata = dict(overrides.get("metadata") or {})

        multi_spec: Optional[MultiAgentSpecRecord] = None
        if template.get("multi_agent") and not overrides.get("multi_agent"):
            multi_spec = await self._maybe_create_multi_agent_spec(
                owner_id=owner_id,
                payload=template,
                visibility=visibility,
                org_id=org_id,
            )
            if multi_spec:
                metadata["multi_agent_id"] = multi_spec.id

        record = await self.create_basic(
            owner_id=owner_id,
            name=merged["name"],
            description=merged.get("description", ""),
            model_id=merged.get("model_id") or overrides.get("model_id") or "gpt-5-nano",
            system_prompt=merged.get("system_prompt", ""),
            tools=merged.get("tools"),
            skills=merged.get("skills"),
            mode=merged.get("mode", "COLLABORATIVE"),
            env=merged.get("env", "cloud_pool"),
            graph_type=graph_type,
            visibility=visibility,
            org_id=org_id,
            source=source,
            template_id=template.get("id"),
            metadata=metadata,
        )
        return record, multi_spec

    async def create_generated(
        self,
        *,
        owner_id: str,
        generated_payload: Dict[str, Any],
        visibility: AgentVisibility = AgentVisibility.PRIVATE,
        org_id: Optional[str] = None,
        source: str = "generated",
    ) -> Tuple[AgentConfigRecord, Optional[MultiAgentSpecRecord]]:
        name = generated_payload.get("name") or "Generated Agent"
        description = generated_payload.get("description") or ""
        model_id = generated_payload.get("model_id") or generated_payload.get("model") or "gpt-5-nano"
        system_prompt = generated_payload.get("system_prompt") or ""
        tools = generated_payload.get("tools") or []
        skills = generated_payload.get("skills") or []
        mode = generated_payload.get("mode") or "COLLABORATIVE"
        env = generated_payload.get("env") or "cloud_pool"
        metadata = generated_payload.get("metadata") or {}

        multi_spec = await self._maybe_create_multi_agent_spec(
            owner_id=owner_id,
            payload=generated_payload,
            visibility=visibility,
            org_id=org_id,
        )
        if multi_spec:
            metadata = dict(metadata)
            metadata["multi_agent_id"] = multi_spec.id

        record = await self.create_basic(
            owner_id=owner_id,
            name=name,
            description=description,
            model_id=model_id,
            system_prompt=system_prompt,
            tools=tools,
            skills=skills,
            mode=mode,
            env=env,
            visibility=visibility,
            org_id=org_id,
            source=source,
            metadata=metadata,
        )
        return record, multi_spec

    async def create_local_dev(
        self,
        *,
        owner_id: str,
        name: str,
        description: str = "",
        deployment_id: Optional[str] = None,
        model_id: str = "gpt-5-nano",
        system_prompt: str = "",
        tools: Optional[list] = None,
        skills: Optional[list] = None,
        mode: str = "COLLABORATIVE",
        env: str = "cloud_pool",
        visibility: AgentVisibility = AgentVisibility.PRIVATE,
        org_id: Optional[str] = None,
        source: str = "local",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentConfigRecord:
        merged_metadata = dict(metadata or {})
        if deployment_id:
            merged_metadata["deployment_id"] = deployment_id

        return await self.create_basic(
            owner_id=owner_id,
            name=name,
            description=description,
            model_id=model_id,
            system_prompt=system_prompt,
            tools=tools,
            skills=skills,
            mode=mode,
            env=env,
            visibility=visibility,
            org_id=org_id,
            source=source,
            metadata=merged_metadata,
        )

    async def _maybe_create_multi_agent_spec(
        self,
        *,
        owner_id: str,
        payload: Dict[str, Any],
        visibility: AgentVisibility,
        org_id: Optional[str],
    ) -> Optional[MultiAgentSpecRecord]:
        multi = (
            payload.get("multi_agent")
            or payload.get("multi_agent_spec")
            or payload.get("multi_agent_config")
        )
        if not multi and payload.get("agents") and payload.get("tasks"):
            multi = {
                "agents": payload.get("agents"),
                "tasks": payload.get("tasks"),
            }
        if not multi:
            return None

        routing_spec = multi.get("routing_spec") if isinstance(multi, dict) else None
        if routing_spec is None:
            routing_spec = dict(multi) if isinstance(multi, dict) else {"value": multi}

        entry_agent = None
        if isinstance(multi, dict):
            entry_agent = (
                multi.get("entry_agent_id")
                or multi.get("entry_agent")
                or multi.get("entry")
            )
            if not entry_agent and isinstance(multi.get("agents"), list) and len(multi["agents"]) > 0:
                first = multi["agents"][0]
                if isinstance(first, dict):
                    entry_agent = first.get("id") or first.get("name")

        routing_type_value = (multi.get("routing_type") if isinstance(multi, dict) else None) or "dag"
        execution_mode_value = (multi.get("execution_mode") if isinstance(multi, dict) else None) or "parallel"

        try:
            routing_type = MultiAgentRoutingType(routing_type_value)
        except Exception:
            routing_type = MultiAgentRoutingType.DAG

        try:
            execution_mode = MultiAgentExecutionMode(execution_mode_value)
        except Exception:
            execution_mode = MultiAgentExecutionMode.PARALLEL

        if not entry_agent:
            raise ValueError(
                "Cannot infer entry_agent_id for multi-agent spec: "
                "no entry_agent_id/entry_agent/entry field found and "
                "agents list is empty or missing id/name"
            )

        now = datetime.utcnow()
        record = MultiAgentSpecRecord(
            id=await self.store.new_multi_id(),
            owner_id=owner_id,
            org_id=org_id if visibility == AgentVisibility.ORG else None,
            visibility=visibility,
            name=(multi.get("name") if isinstance(multi, dict) else None) or "multi_agent",
            description=multi.get("description") if isinstance(multi, dict) else None,
            entry_agent_id=entry_agent,
            routing_type=routing_type,
            execution_mode=execution_mode,
            routing_spec=routing_spec or {},
            created_at=now,
            updated_at=now,
        )
        return await self.store.create_multi_spec(record)


def _normalize_mode(mode: str) -> str:
    return mode.lower() if isinstance(mode, str) else "reactive"


def _build_options(
    *,
    model_id: str,
    system_prompt: str,
    tools: Optional[list],
    skills: Optional[list],
    mode: str,
    env: str,
    graph_type: str = "smart_agent",
) -> Dict[str, Any]:
    return {
        "model": model_id,
        "system_prompt": system_prompt,
        "allowed_tools": tools or [],
        "skills": skills or [],
        "execution_mode": _normalize_mode(mode),
        "env": env,
        "graph_type": graph_type,
    }
