#!/usr/bin/env python3
"""
Agent wrapper for isa_agent_sdk

Provides a clean "Agent" abstraction that hides the underlying graph
implementation. Uses the existing query() API internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional
import copy

from isa_agent_sdk.options import ISAAgentOptions
from isa_agent_sdk._messages import AgentMessage
from isa_agent_sdk import query


RunnerFn = Callable[[str, ISAAgentOptions], AsyncIterator[AgentMessage]]


@dataclass
class AgentRunResult:
    text: str
    messages: list[AgentMessage]
    structured_output: Optional[Dict[str, Any]] = None
    shared_state: Optional[Dict[str, Any]] = None


class Agent:
    """
    High-level Agent abstraction.

    Args:
        name: Agent name
        options: ISAAgentOptions for this agent
        runner: Optional override for query() (useful for tests)
    """

    def __init__(
        self,
        name: str,
        options: Optional[ISAAgentOptions] = None,
        runner: Optional[RunnerFn] = None,
    ):
        self.name = name
        self.options = options or ISAAgentOptions()
        self._runner = runner or query

    def _clone_options(self, overrides: Optional[Dict[str, Any]] = None) -> ISAAgentOptions:
        overrides = overrides or {}
        try:
            import dataclasses
            return dataclasses.replace(self.options, **overrides)
        except Exception:
            cloned = copy.copy(self.options)
            for key, value in overrides.items():
                setattr(cloned, key, value)
            return cloned

    async def stream(
        self,
        prompt: str,
        *,
        options: Optional[ISAAgentOptions] = None,
        **overrides: Any,
    ) -> AsyncIterator[AgentMessage]:
        """Stream messages from the agent."""
        run_options = options or self._clone_options(overrides)
        async for msg in self._runner(prompt, options=run_options):
            yield msg

    async def run(
        self,
        prompt: str,
        *,
        options: Optional[ISAAgentOptions] = None,
        **overrides: Any,
    ) -> AgentRunResult:
        """Run the agent and collect final output."""
        run_options = options or self._clone_options(overrides)
        messages: list[AgentMessage] = []
        final_text = ""
        structured_output: Optional[Dict[str, Any]] = None

        async for msg in self._runner(prompt, options=run_options):
            messages.append(msg)
            if msg.structured_output:
                structured_output = msg.structured_output
            if msg.is_complete and msg.content:
                final_text = msg.content
            elif msg.is_text and msg.content:
                final_text += msg.content

        return AgentRunResult(
            text=final_text,
            messages=messages,
            structured_output=structured_output,
        )
