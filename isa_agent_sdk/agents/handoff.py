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

# Directive patterns that must be stripped from injected content
_DIRECTIVE_STRIP_RE = re.compile(r"\[(?:HANDOFF:[^\]]*|COMPLETE)\]", re.IGNORECASE)


def _sanitize_prompt_text(text: str) -> str:
    """Strip handoff/complete directives from text to prevent prompt injection."""
    return _DIRECTIVE_STRIP_RE.sub("", text).strip()


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
        safe_name = _sanitize_prompt_text(peer.name)
        safe_desc = _sanitize_prompt_text(peer.effective_description)
        peer_lines.append(f"- **{safe_name}**: {safe_desc}")

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

    # Search the last 500 characters for directives to avoid scanning huge responses
    tail = response_text[-500:]

    # Check for HANDOFF directive in tail
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

        # Compute directive position relative to full text
        tail_offset = max(0, len(response_text) - 500)
        directive_start = tail_offset + match.start()
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
    tail_snippet = response_text[-80:] if len(response_text) > 80 else response_text
    logger.info(
        "No handoff directive found in response; defaulting to COMPLETE. "
        "Response tail: %r",
        tail_snippet,
    )
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
