#!/usr/bin/env python3
"""
Multi-agent orchestration for isa_agent_sdk.

This module exposes a clean orchestrator that composes multiple Agent
instances with explicit handoff and shared state. Graphs remain internal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, AsyncIterator

from isa_agent_sdk._messages import AgentMessage
from .agent import Agent, AgentRunResult


RouterFn = Callable[[Dict[str, Any], AgentRunResult], Optional[str]]


@dataclass
class MultiAgentResult:
    outputs: Dict[str, AgentRunResult]
    shared_state: Dict[str, Any]
    final_agent: Optional[str] = None


class MultiAgentOrchestrator:
    """
    Orchestrate multiple Agents with handoff + shared state.

    Args:
        agents: Mapping of agent name -> Agent
        entry_agent: Default first agent to run
        router: Optional router function (state, last_result) -> next_agent
        max_steps: Safety cap on number of handoffs
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        entry_agent: Optional[str] = None,
        router: Optional[RouterFn] = None,
        max_steps: int = 6,
    ):
        if not agents:
            raise ValueError("agents mapping cannot be empty")
        self.agents = agents
        self.entry_agent = entry_agent or next(iter(agents.keys()))
        self.router = router
        self.max_steps = max_steps

    async def run(
        self,
        prompt: str,
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> MultiAgentResult:
        """Run the multi-agent workflow and return final outputs."""
        shared_state = state.get("shared_state") if state else None
        if shared_state is None:
            shared_state = {}

        outputs: Dict[str, AgentRunResult] = {}
        next_agent = (state or {}).get("next_agent") or self.entry_agent
        final_agent: Optional[str] = None

        for _ in range(self.max_steps):
            if not next_agent:
                break
            if next_agent.lower() in {"end", "done", "finish", "complete"}:
                break
            if next_agent not in self.agents:
                raise KeyError(f"Unknown agent '{next_agent}'. Available: {list(self.agents.keys())}")

            agent = self.agents[next_agent]
            result = await agent.run(prompt)
            outputs[next_agent] = result
            final_agent = next_agent

            if result.shared_state:
                shared_state.update(result.shared_state)

            if self.router:
                next_agent = self.router(
                    {"shared_state": shared_state, "outputs": outputs, "last_agent": final_agent},
                    result,
                )
            else:
                next_agent = None

        return MultiAgentResult(outputs=outputs, shared_state=shared_state, final_agent=final_agent)

    async def stream(
        self,
        prompt: str,
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentMessage]:
        """
        Stream messages across agents. Yields AgentMessage with metadata
        indicating which agent produced the message.
        """
        shared_state = state.get("shared_state") if state else None
        if shared_state is None:
            shared_state = {}

        outputs: Dict[str, AgentRunResult] = {}
        next_agent = (state or {}).get("next_agent") or self.entry_agent

        for _ in range(self.max_steps):
            if not next_agent:
                break
            if next_agent.lower() in {"end", "done", "finish", "complete"}:
                break
            if next_agent not in self.agents:
                raise KeyError(f"Unknown agent '{next_agent}'. Available: {list(self.agents.keys())}")

            agent = self.agents[next_agent]
            last_result: Optional[AgentRunResult] = None

            async for msg in agent.stream(prompt):
                # Annotate message with agent name
                if msg.metadata is None:
                    msg.metadata = {}
                msg.metadata["agent"] = next_agent
                yield msg

                # Capture last complete message for routing
                if msg.is_complete:
                    last_result = AgentRunResult(text=msg.content or "", messages=[msg])

            if last_result is not None:
                outputs[next_agent] = last_result

            if self.router and last_result is not None:
                next_agent = self.router(
                    {"shared_state": shared_state, "outputs": outputs, "last_agent": next_agent},
                    last_result,
                )
            else:
                next_agent = None
