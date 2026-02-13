#!/usr/bin/env python3
"""
Multi-Agent Runner

Executes multi-agent specs stored in AgentConfigStore using SwarmOrchestrator.
Routing priority: DAG (explicit) -> LLM handoff (implicit).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from isa_agent_sdk.agents import Agent, SwarmAgent, SwarmOrchestrator
from isa_agent_sdk.options import ISAAgentOptions
from isa_agent_sdk._messages import AgentMessage

from .agent_config_store import (
    AgentConfigStore,
    MultiAgentExecutionMode,
    MultiAgentRoutingType,
)


class MultiAgentRunner:
    """Run multi-agent specs with DAG-first routing."""

    def __init__(self, store: AgentConfigStore):
        self.store = store

    async def run(
        self,
        *,
        multi_agent_id: str,
        prompt: str,
        user_id: str,
    ) -> str:
        multi_spec = await self.store.get_multi_spec(multi_agent_id)

        if not multi_spec:
            raise ValueError("multi_agent spec not found")

        orchestrator, tasks, routing_type, execution_mode = _build_orchestrator(
            multi_spec.routing_spec, user_id,
            entry_agent_override=multi_spec.entry_agent_id,
        )

        if routing_type == MultiAgentRoutingType.DAG and tasks:
            result = await orchestrator.run_dag(
                tasks,
                parallel=(execution_mode == MultiAgentExecutionMode.PARALLEL),
            )
            return result.text

        result = await orchestrator.run(prompt)
        return result.text

    async def stream(
        self,
        *,
        multi_agent_id: str,
        prompt: str,
        user_id: str,
    ) -> AsyncIterator[AgentMessage]:
        multi_spec = await self.store.get_multi_spec(multi_agent_id)
        if not multi_spec:
            raise ValueError("multi_agent spec not found")

        orchestrator, tasks, routing_type, execution_mode = _build_orchestrator(
            multi_spec.routing_spec, user_id,
            entry_agent_override=multi_spec.entry_agent_id,
        )

        if routing_type == MultiAgentRoutingType.DAG and tasks:
            result = await orchestrator.run_dag(
                tasks,
                parallel=(execution_mode == MultiAgentExecutionMode.PARALLEL),
            )
            yield AgentMessage(type="result", content=result.text)
            return

        async for msg in orchestrator.stream(prompt):
            yield msg


def _build_orchestrator(
    routing_spec: Dict[str, Any],
    user_id: str,
    entry_agent_override: Optional[str] = None,
) -> tuple[SwarmOrchestrator, List[Dict[str, Any]], MultiAgentRoutingType, MultiAgentExecutionMode]:
    agents_spec = routing_spec.get("agents") or []
    tasks = routing_spec.get("tasks") or routing_spec.get("dag") or []

    routing_type_value = routing_spec.get("routing_type") or "dag"
    execution_mode_value = routing_spec.get("execution_mode") or "parallel"

    try:
        routing_type = MultiAgentRoutingType(routing_type_value)
    except Exception:
        routing_type = MultiAgentRoutingType.DAG

    try:
        execution_mode = MultiAgentExecutionMode(execution_mode_value)
    except Exception:
        execution_mode = MultiAgentExecutionMode.PARALLEL

    swarm_agents: List[SwarmAgent] = []
    for agent_def in agents_spec:
        if not isinstance(agent_def, dict):
            continue
        name = agent_def.get("name") or agent_def.get("id")
        if not name:
            continue
        opts = ISAAgentOptions(
            system_prompt=agent_def.get("system_prompt", ""),
            allowed_tools=agent_def.get("tools", []),
            model=agent_def.get("model") or agent_def.get("model_id") or "gpt-5-nano",
            skills=agent_def.get("skills", []),
            execution_mode=(agent_def.get("mode") or "reactive").lower(),
            user_id=user_id,
        )
        swarm_agents.append(
            SwarmAgent(
                agent=Agent(name, opts),
                description=agent_def.get("description", name),
            )
        )

    if not swarm_agents:
        raise ValueError("multi_agent spec missing agents")

    entry_agent = (
        entry_agent_override
        or routing_spec.get("entry_agent_id")
        or routing_spec.get("entry_agent")
        or swarm_agents[0].name
    )

    orchestrator = SwarmOrchestrator(
        agents=swarm_agents,
        entry_agent=entry_agent,
        max_handoffs=routing_spec.get("max_handoffs", 10),
    )
    return orchestrator, tasks, routing_type, execution_mode
