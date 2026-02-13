"""
Swarm multi-agent orchestration data models.

Provides data structures for dynamic agent handoff orchestration
where agents decide when to hand off control based on their specialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent, AgentRunResult
    from isa_agent_sdk._messages import AgentMessage


class HandoffAction(str, Enum):
    """Action parsed from an agent's response."""
    CONTINUE = "continue"
    HANDOFF = "handoff"
    COMPLETE = "complete"


@dataclass
class HandoffResult:
    """Result of parsing an agent's response for handoff directives."""
    action: HandoffAction
    target_agent: Optional[str] = None
    reason: Optional[str] = None
    context_for_next: Optional[str] = None


@dataclass
class SwarmAgent:
    """
    Wrapper around an Agent with swarm-specific metadata.

    Args:
        agent: The underlying Agent instance
        description: Human-readable description of this agent's capabilities
        handoff_description: Override description shown to peer agents (defaults to description)
    """
    agent: "Agent"
    description: str = ""
    handoff_description: Optional[str] = None

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def effective_description(self) -> str:
        return self.handoff_description or self.description


@dataclass
class SwarmState:
    """
    Mutable state tracked across the swarm execution loop.

    Tracks the active agent, conversation history, shared state,
    and a trace of all handoffs for debugging/observability.
    """
    active_agent: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    handoff_trace: List[Dict[str, str]] = field(default_factory=list)
    handoff_count: int = 0

    def record_handoff(self, from_agent: str, to_agent: str, reason: str) -> None:
        self.handoff_trace.append({
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "step": str(self.handoff_count),
        })
        self.handoff_count += 1
        self.active_agent = to_agent


@dataclass
class SwarmRunResult:
    """Final result returned by SwarmOrchestrator.run()."""
    text: str
    messages: List["AgentMessage"]
    final_agent: str
    handoff_trace: List[Dict[str, str]]
    shared_state: Dict[str, Any]
    agent_outputs: Dict[str, "AgentRunResult"]
