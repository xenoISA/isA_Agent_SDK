"""
Handoff prompt injection and response parsing for swarm orchestration.

Teaches agents about their peers via system prompt injection and
detects handoff directives in agent responses.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from .swarm_types import HandoffAction, HandoffResult, SwarmAgent

logger = logging.getLogger(__name__)

# Pattern: [HANDOFF: agent_name] optional reason text
_HANDOFF_RE = re.compile(
    r"\[HANDOFF:\s*([^\]]+?)\s*\]\s*(.*)",
    re.DOTALL,
)

# Pattern: [COMPLETE]
_COMPLETE_RE = re.compile(r"\[COMPLETE\]")


def build_handoff_system_prompt_section(
    current_agent_name: str,
    peers: List[SwarmAgent],
    conversation_summary: Optional[str] = None,
) -> str:
    """Build a prompt section that teaches an agent about its peers and handoff syntax.

    Args:
        current_agent_name: Name of the agent receiving this prompt section
        peers: All SwarmAgents in the swarm (current agent will be filtered out)
        conversation_summary: Optional summary of recent conversation turns

    Returns:
        Prompt section string to append to the agent's system prompt
    """
    peer_lines = []
    for peer in peers:
        if peer.name == current_agent_name:
            continue
        peer_lines.append(f"- **{peer.name}**: {peer.effective_description}")

    if not peer_lines:
        return ""

    section = (
        "\n\n---\n"
        "## Agent Collaboration\n\n"
        "You are part of a multi-agent team. The following peer agents are available:\n"
        + "\n".join(peer_lines)
        + "\n\n"
        "### Handoff Protocol\n"
        "When you determine that another agent is better suited to continue, "
        "end your response with:\n"
        "```\n"
        "[HANDOFF: agent_name] reason for handoff\n"
        "```\n\n"
        "When your part of the task is fully complete, end your response with:\n"
        "```\n"
        "[COMPLETE]\n"
        "```\n\n"
        "### Rules\n"
        "- Only hand off when the task genuinely requires another agent's expertise\n"
        "- Never hand off to yourself\n"
        "- Provide useful context before the handoff directive\n"
        "- If no handoff is needed, just complete your response normally\n"
    )

    if conversation_summary:
        section += (
            "\n### Recent Context\n"
            f"{conversation_summary}\n"
        )

    return section


def parse_handoff_from_response(
    response_text: str,
    available_agents: Set[str],
) -> HandoffResult:
    """Parse an agent's response text for handoff directives.

    Scans the last 500 characters for [HANDOFF: agent_name] or [COMPLETE].

    Args:
        response_text: Full response text from the agent
        available_agents: Set of valid agent names for validation

    Returns:
        HandoffResult describing the detected action
    """
    if not response_text:
        return HandoffResult(action=HandoffAction.COMPLETE)

    tail = response_text[-500:]

    # Check for HANDOFF directive
    match = _HANDOFF_RE.search(tail)
    if match:
        target = match.group(1).strip()
        reason = match.group(2).strip() or ""

        if target not in available_agents:
            logger.warning(
                "Agent requested handoff to unknown agent '%s'. "
                "Available: %s. Treating as COMPLETE.",
                target,
                available_agents,
            )
            return HandoffResult(
                action=HandoffAction.COMPLETE,
                context_for_next=response_text,
                reason=f"Unknown target agent: {target}",
            )

        # Text before the directive is context for the next agent
        directive_start = response_text.rfind("[HANDOFF:")
        context = response_text[:directive_start].rstrip() if directive_start > 0 else ""

        return HandoffResult(
            action=HandoffAction.HANDOFF,
            target_agent=target,
            reason=reason,
            context_for_next=context,
        )

    # Check for COMPLETE directive
    if _COMPLETE_RE.search(tail):
        return HandoffResult(action=HandoffAction.COMPLETE)

    # No directive found — treat as complete (safe default)
    return HandoffResult(action=HandoffAction.COMPLETE)


def strip_handoff_directive(text: str) -> str:
    """Remove handoff directives from text for clean user-facing output.

    Strips [HANDOFF: ...] reason and [COMPLETE] from the text.

    Args:
        text: Response text potentially containing directives

    Returns:
        Cleaned text with directives removed
    """
    if not text:
        return text

    # Remove [HANDOFF: agent_name] reason (including rest of line)
    cleaned = re.sub(r"\[HANDOFF:\s*[^\]]+?\s*\].*", "", text, flags=re.DOTALL)
    # Remove [COMPLETE]
    cleaned = _COMPLETE_RE.sub("", cleaned)
    return cleaned.rstrip()
