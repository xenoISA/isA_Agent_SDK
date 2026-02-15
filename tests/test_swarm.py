"""
Tests for the Swarm multi-agent orchestration system.
"""

import pytest
from unittest.mock import AsyncMock

from isa_agent_sdk._messages import AgentMessage
from isa_agent_sdk.options import ISAAgentOptions, SystemPromptConfig
from isa_agent_sdk.agents.agent import Agent, AgentRunResult
from isa_agent_sdk.agents.swarm_types import (
    HandoffAction,
    HandoffResult,
    SwarmAgent,
    SwarmRunResult,
    SwarmState,
)
from isa_agent_sdk.agents.handoff import (
    build_handoff_system_prompt_section,
    parse_handoff_from_response,
    strip_handoff_directive,
)
from isa_agent_sdk.agents.swarm import SwarmOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(response_text: str):
    """Create a mock runner that yields a single result message."""
    async def runner(prompt, *, options=None):
        yield AgentMessage.result(response_text)
    return runner


def _make_agent(name: str, response_text: str, system_prompt=None) -> Agent:
    """Create an Agent with a mock runner returning fixed text."""
    opts = ISAAgentOptions(system_prompt=system_prompt)
    return Agent(name, opts, runner=_make_runner(response_text))


def _make_stateful_runner(responses: list):
    """Create a runner that returns different responses on successive calls."""
    call_count = {"n": 0}

    async def runner(prompt, *, options=None):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        yield AgentMessage.result(responses[idx])

    return runner


# ---------------------------------------------------------------------------
# Handoff parsing tests
# ---------------------------------------------------------------------------

class TestHandoffParsing:
    def test_handoff_parsing_valid(self):
        text = "Here is my research.\n\n[HANDOFF: writer] Please write the summary"
        result = parse_handoff_from_response(text, {"writer", "reviewer"})
        assert result.action == HandoffAction.HANDOFF
        assert result.target_agent == "writer"
        assert result.reason == "Please write the summary"

    def test_handoff_parsing_unknown_agent(self):
        text = "Done.\n[HANDOFF: unknown_agent] reason"
        result = parse_handoff_from_response(text, {"writer", "reviewer"})
        assert result.action == HandoffAction.COMPLETE

    def test_handoff_parsing_no_directive(self):
        text = "Here is a normal response with no special directives."
        result = parse_handoff_from_response(text, {"writer"})
        assert result.action == HandoffAction.COMPLETE

    def test_handoff_parsing_complete(self):
        text = "All done with the task.\n[COMPLETE]"
        result = parse_handoff_from_response(text, {"writer"})
        assert result.action == HandoffAction.COMPLETE

    def test_handoff_parsing_in_long_text(self):
        long_prefix = "x" * 1000
        text = long_prefix + "\n[HANDOFF: writer] take over"
        result = parse_handoff_from_response(text, {"writer"})
        assert result.action == HandoffAction.HANDOFF
        assert result.target_agent == "writer"

    def test_handoff_parsing_empty_text(self):
        result = parse_handoff_from_response("", {"writer"})
        assert result.action == HandoffAction.COMPLETE

    def test_handoff_context_for_next(self):
        text = "Research findings:\n- Point A\n- Point B\n\n[HANDOFF: writer] summarize"
        result = parse_handoff_from_response(text, {"writer"})
        assert result.action == HandoffAction.HANDOFF
        assert "Point A" in result.context_for_next
        assert "Point B" in result.context_for_next


# ---------------------------------------------------------------------------
# Prompt injection tests
# ---------------------------------------------------------------------------

class TestPromptInjection:
    def test_prompt_injection_builds_section(self):
        peers = [
            SwarmAgent(agent=_make_agent("researcher", ""), description="Research expert"),
            SwarmAgent(agent=_make_agent("writer", ""), description="Technical writer"),
        ]
        section = build_handoff_system_prompt_section("researcher", peers)
        assert "writer" in section
        assert "Technical writer" in section
        assert "[HANDOFF:" in section
        assert "[COMPLETE]" in section
        # Should not list self
        assert "**researcher**" not in section

    def test_prompt_injection_with_summary(self):
        peers = [
            SwarmAgent(agent=_make_agent("writer", ""), description="Writer"),
        ]
        section = build_handoff_system_prompt_section(
            "researcher", peers, conversation_summary="Previous turn info",
        )
        assert "Previous turn info" in section

    def test_prompt_injection_single_agent(self):
        """No peers → empty section."""
        peers = [
            SwarmAgent(agent=_make_agent("only_agent", ""), description="Solo"),
        ]
        section = build_handoff_system_prompt_section("only_agent", peers)
        assert section == ""


# ---------------------------------------------------------------------------
# strip_handoff_directive tests
# ---------------------------------------------------------------------------

class TestStripHandoffDirective:
    def test_strip_handoff(self):
        text = "Here is my output.\n\n[HANDOFF: writer] please continue"
        cleaned = strip_handoff_directive(text)
        assert "[HANDOFF" not in cleaned
        assert "please continue" not in cleaned
        assert "Here is my output." in cleaned

    def test_strip_complete(self):
        text = "Final answer.\n[COMPLETE]"
        cleaned = strip_handoff_directive(text)
        assert "[COMPLETE]" not in cleaned
        assert "Final answer." in cleaned

    def test_strip_no_directive(self):
        text = "Normal text without directives."
        assert strip_handoff_directive(text) == text

    def test_strip_empty(self):
        assert strip_handoff_directive("") == ""


# ---------------------------------------------------------------------------
# SwarmState tests
# ---------------------------------------------------------------------------

class TestSwarmState:
    def test_record_handoff(self):
        state = SwarmState(active_agent="a")
        state.record_handoff("a", "b", "needs writing")
        assert state.active_agent == "b"
        assert state.handoff_count == 1
        assert state.handoff_trace[0]["from"] == "a"
        assert state.handoff_trace[0]["to"] == "b"
        assert state.handoff_trace[0]["reason"] == "needs writing"


# ---------------------------------------------------------------------------
# SwarmOrchestrator constructor tests
# ---------------------------------------------------------------------------

class TestSwarmConstructor:
    def test_list_constructor(self):
        agents = [
            SwarmAgent(agent=_make_agent("a", ""), description="Agent A"),
            SwarmAgent(agent=_make_agent("b", ""), description="Agent B"),
        ]
        swarm = SwarmOrchestrator(agents=agents, entry_agent="a")
        assert swarm.entry_agent == "a"

    def test_dict_constructor(self):
        agents = {
            "a": _make_agent("a", ""),
            "b": _make_agent("b", ""),
        }
        swarm = SwarmOrchestrator(agents=agents)
        assert swarm.entry_agent == "a"

    def test_empty_agents_raises(self):
        with pytest.raises(ValueError, match="empty"):
            SwarmOrchestrator(agents=[])

    def test_invalid_entry_agent_raises(self):
        agents = [SwarmAgent(agent=_make_agent("a", ""), description="A")]
        with pytest.raises(ValueError, match="not found"):
            SwarmOrchestrator(agents=agents, entry_agent="nonexistent")

    def test_default_entry_agent(self):
        agents = [
            SwarmAgent(agent=_make_agent("first", ""), description="First"),
            SwarmAgent(agent=_make_agent("second", ""), description="Second"),
        ]
        swarm = SwarmOrchestrator(agents=agents)
        assert swarm.entry_agent == "first"


# ---------------------------------------------------------------------------
# SwarmOrchestrator.run() tests
# ---------------------------------------------------------------------------

class TestSwarmRun:
    async def test_single_agent_no_handoff(self):
        agent = _make_agent("solo", "The answer is 42. [COMPLETE]")
        swarm = SwarmOrchestrator(
            agents=[SwarmAgent(agent=agent, description="Solo agent")],
        )
        result = await swarm.run("What is the answer?")
        assert "42" in result.text
        assert result.final_agent == "solo"
        assert len(result.handoff_trace) == 0
        assert "[COMPLETE]" not in result.text

    async def test_handoff_chain(self):
        researcher = Agent(
            "researcher",
            ISAAgentOptions(),
            runner=_make_runner(
                "Research findings: AI is great.\n[HANDOFF: writer] Please write summary"
            ),
        )
        writer = Agent(
            "writer",
            ISAAgentOptions(),
            runner=_make_runner("Summary: AI is great. [COMPLETE]"),
        )
        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=researcher, description="Research expert"),
                SwarmAgent(agent=writer, description="Technical writer"),
            ],
            entry_agent="researcher",
        )
        result = await swarm.run("Research AI and write summary")
        assert result.final_agent == "writer"
        assert len(result.handoff_trace) == 1
        assert result.handoff_trace[0]["from"] == "researcher"
        assert result.handoff_trace[0]["to"] == "writer"
        assert "Summary" in result.text

    async def test_max_handoffs(self):
        """Infinite handoff loop should be capped."""
        # Both agents always hand off to each other
        a = Agent("a", ISAAgentOptions(), runner=_make_runner("[HANDOFF: b] your turn"))
        b = Agent("b", ISAAgentOptions(), runner=_make_runner("[HANDOFF: a] your turn"))
        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="Agent A"),
                SwarmAgent(agent=b, description="Agent B"),
            ],
            entry_agent="a",
            max_handoffs=3,
        )
        result = await swarm.run("ping pong")
        # Should complete after max_handoffs + 1 iterations
        assert len(result.handoff_trace) <= 4

    async def test_shared_state(self):
        """Shared state accumulates across handoffs."""
        async def runner_a(prompt, *, options=None):
            yield AgentMessage.result("Done A. [HANDOFF: b] continue")

        async def runner_b(prompt, *, options=None):
            yield AgentMessage.result("Done B. [COMPLETE]")

        a = Agent("a", ISAAgentOptions(), runner=runner_a)
        b = Agent("b", ISAAgentOptions(), runner=runner_b)

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="A"),
                SwarmAgent(agent=b, description="B"),
            ],
            entry_agent="a",
            shared_state={"initial": True},
        )
        result = await swarm.run("test")
        assert result.shared_state["initial"] is True

    async def test_no_directive_treated_as_complete(self):
        """Response without any directive should complete."""
        agent = _make_agent("solo", "Here is my answer with no directive.")
        swarm = SwarmOrchestrator(
            agents=[SwarmAgent(agent=agent, description="Solo")],
        )
        result = await swarm.run("question")
        assert result.final_agent == "solo"
        assert result.text == "Here is my answer with no directive."


# ---------------------------------------------------------------------------
# SwarmOrchestrator.stream() tests
# ---------------------------------------------------------------------------

class TestSwarmStream:
    async def test_stream_annotations(self):
        agent = _make_agent("solo", "Streamed response. [COMPLETE]")
        swarm = SwarmOrchestrator(
            agents=[SwarmAgent(agent=agent, description="Solo")],
        )
        messages = []
        async for msg in swarm.stream("test"):
            messages.append(msg)

        # Should have at least: agent_start system message + result message
        agent_starts = [m for m in messages if m.metadata.get("event") == "swarm_agent_start"]
        assert len(agent_starts) == 1
        assert agent_starts[0].metadata["agent"] == "solo"

        # Result messages should have agent annotation
        result_msgs = [m for m in messages if m.is_complete]
        assert all(m.metadata.get("agent") == "solo" for m in result_msgs)

    async def test_stream_handoff_events(self):
        researcher = Agent(
            "researcher",
            ISAAgentOptions(),
            runner=_make_runner("Research done. [HANDOFF: writer] write it up"),
        )
        writer = Agent(
            "writer",
            ISAAgentOptions(),
            runner=_make_runner("Written! [COMPLETE]"),
        )
        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=researcher, description="Research"),
                SwarmAgent(agent=writer, description="Writing"),
            ],
            entry_agent="researcher",
        )

        messages = []
        async for msg in swarm.stream("do both"):
            messages.append(msg)

        handoff_events = [m for m in messages if m.metadata.get("event") == "swarm_handoff"]
        assert len(handoff_events) == 1
        assert handoff_events[0].metadata["from_agent"] == "researcher"
        assert handoff_events[0].metadata["to_agent"] == "writer"


# ---------------------------------------------------------------------------
# Options mutation tests
# ---------------------------------------------------------------------------

class TestOptionsNotMutated:
    async def test_options_not_mutated(self):
        original_prompt = "You are a researcher."
        agent = _make_agent("researcher", "Done. [COMPLETE]", system_prompt=original_prompt)
        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=agent, description="Research"),
                SwarmAgent(agent=_make_agent("writer", ""), description="Writing"),
            ],
            entry_agent="researcher",
        )
        await swarm.run("test")
        # Original agent options should be unchanged
        assert agent.options.system_prompt == original_prompt

    async def test_handoff_with_system_prompt_config(self):
        sp_config = SystemPromptConfig(preset="reason", append="Custom instructions.")
        agent = _make_agent("researcher", "Done. [COMPLETE]")
        agent.options.system_prompt = sp_config

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=agent, description="Research"),
                SwarmAgent(agent=_make_agent("writer", ""), description="Writing"),
            ],
            entry_agent="researcher",
        )
        await swarm.run("test")
        # Original config should be unchanged
        assert agent.options.system_prompt is sp_config
        assert agent.options.system_prompt.append == "Custom instructions."


# ---------------------------------------------------------------------------
# DAG execution tests
# ---------------------------------------------------------------------------

class TestSwarmDAG:
    async def test_dag_execution(self):
        researcher = Agent(
            "researcher", ISAAgentOptions(),
            runner=_make_runner("Research data gathered."),
        )
        writer = Agent(
            "writer", ISAAgentOptions(),
            runner=_make_runner("Report written."),
        )
        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=researcher, description="Research"),
                SwarmAgent(agent=writer, description="Writing"),
            ],
            entry_agent="researcher",
        )

        result = await swarm.run_dag([
            {
                "id": "research",
                "title": "Research",
                "description": "Gather data",
                "agent": "researcher",
            },
            {
                "id": "write",
                "title": "Write",
                "description": "Write report",
                "agent": "writer",
                "depends_on": ["research"],
            },
        ])

        assert "researcher:research" in result.agent_outputs
        assert "writer:write" in result.agent_outputs
        assert "Report written." in result.text

    async def test_dag_concurrent_wavefront(self):
        """Tasks on different agents in the same wavefront run concurrently."""
        call_order = []

        async def runner_a(prompt, *, options=None):
            call_order.append("a")
            yield AgentMessage.result("A done")

        async def runner_b(prompt, *, options=None):
            call_order.append("b")
            yield AgentMessage.result("B done")

        a = Agent("a", ISAAgentOptions(), runner=runner_a)
        b = Agent("b", ISAAgentOptions(), runner=runner_b)

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="A"),
                SwarmAgent(agent=b, description="B"),
            ],
            entry_agent="a",
        )

        result = await swarm.run_dag([
            {"id": "t1", "title": "Task 1", "description": "D1", "agent": "a"},
            {"id": "t2", "title": "Task 2", "description": "D2", "agent": "b"},
        ])

        # Both agents should have been called
        assert "a" in call_order
        assert "b" in call_order
        assert "a:t1" in result.agent_outputs
        assert "b:t2" in result.agent_outputs

    async def test_dag_invalid_raises(self):
        agent = _make_agent("a", "")
        swarm = SwarmOrchestrator(
            agents=[SwarmAgent(agent=agent, description="A")],
        )
        with pytest.raises(ValueError, match="validation failed"):
            await swarm.run_dag([
                {"id": "t1", "title": "T1", "description": "D", "depends_on": ["t1"]},
            ])

    async def test_dag_unknown_agent(self):
        """Tasks assigned to unknown agents should fail gracefully."""
        agent = _make_agent("known", "result")
        swarm = SwarmOrchestrator(
            agents=[SwarmAgent(agent=agent, description="Known")],
        )
        result = await swarm.run_dag([
            {"id": "t1", "title": "T1", "description": "D", "agent": "unknown"},
        ])
        # Task should have failed, so no agent outputs for it
        assert "unknown:t1" not in result.agent_outputs


# ---------------------------------------------------------------------------
# Fix 5A: Unknown handoff includes reason
# ---------------------------------------------------------------------------

class TestUnknownHandoffReason:
    def test_unknown_handoff_includes_reason(self):
        text = "Done.\n[HANDOFF: nonexistent_agent] some reason"
        result = parse_handoff_from_response(text, {"writer", "reviewer"})
        assert result.action == HandoffAction.COMPLETE
        assert result.reason is not None
        assert "nonexistent_agent" in result.reason

    def test_unknown_handoff_preserves_context(self):
        text = "Here is output.\n[HANDOFF: ghost] reason"
        result = parse_handoff_from_response(text, {"writer"})
        assert result.action == HandoffAction.COMPLETE
        assert result.context_for_next == text
        assert "ghost" in result.reason


# ---------------------------------------------------------------------------
# Fix 5B: Context truncation marker
# ---------------------------------------------------------------------------

class TestContextTruncation:
    def test_truncation_adds_marker(self):
        long_context = "x" * 3000
        handoff = HandoffResult(
            action=HandoffAction.HANDOFF,
            target_agent="writer",
            reason="continue",
            context_for_next=long_context,
        )
        state = SwarmState(active_agent="researcher")
        prompt = SwarmOrchestrator._build_handoff_prompt("original", handoff, state)
        assert "[...context truncated...]" in prompt

    def test_short_context_no_marker(self):
        short_context = "x" * 100
        handoff = HandoffResult(
            action=HandoffAction.HANDOFF,
            target_agent="writer",
            reason="continue",
            context_for_next=short_context,
        )
        state = SwarmState(active_agent="researcher")
        prompt = SwarmOrchestrator._build_handoff_prompt("original", handoff, state)
        assert "[...context truncated...]" not in prompt
        assert short_context in prompt


# ---------------------------------------------------------------------------
# Issue #33: Concurrent execution consistency tests
# ---------------------------------------------------------------------------

class TestSwarmConcurrentExecution:
    """Test concurrent DAG execution race condition fixes."""

    async def test_swarm_concurrent_execution_consistency(self):
        """Verify message counts are consistent when running multiple agents concurrently.

        This test ensures that the race condition fix (status update under lock)
        works correctly when multiple agents execute tasks in parallel.
        """
        import asyncio

        # Track call order and timing
        call_order = []

        async def runner_a(prompt, *, options=None):
            call_order.append("a_start")
            await asyncio.sleep(0.05)  # Simulate some work
            call_order.append("a_end")
            yield AgentMessage.result("A result")

        async def runner_b(prompt, *, options=None):
            call_order.append("b_start")
            await asyncio.sleep(0.05)  # Simulate some work
            call_order.append("b_end")
            yield AgentMessage.result("B result")

        async def runner_c(prompt, *, options=None):
            call_order.append("c_start")
            await asyncio.sleep(0.05)  # Simulate some work
            call_order.append("c_end")
            yield AgentMessage.result("C result")

        a = Agent("a", ISAAgentOptions(), runner=runner_a)
        b = Agent("b", ISAAgentOptions(), runner=runner_b)
        c = Agent("c", ISAAgentOptions(), runner=runner_c)

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="Agent A"),
                SwarmAgent(agent=b, description="Agent B"),
                SwarmAgent(agent=c, description="Agent C"),
            ],
            entry_agent="a",
        )

        # Run DAG with parallel tasks on different agents
        result = await swarm.run_dag([
            {"id": "t1", "title": "Task 1", "description": "D1", "agent": "a"},
            {"id": "t2", "title": "Task 2", "description": "D2", "agent": "b"},
            {"id": "t3", "title": "Task 3", "description": "D3", "agent": "c"},
        ], parallel=True)

        # Verify all agents were called
        assert "a:t1" in result.agent_outputs
        assert "b:t2" in result.agent_outputs
        assert "c:t3" in result.agent_outputs

        # Verify all messages were collected (3 tasks * 1 message each)
        assert len(result.messages) == 3

        # Verify results are correct
        assert result.agent_outputs["a:t1"].text == "A result"
        assert result.agent_outputs["b:t2"].text == "B result"
        assert result.agent_outputs["c:t3"].text == "C result"

    async def test_swarm_sequential_dag_consistency(self):
        """Test that sequential DAG execution also maintains consistency."""
        call_order = []

        async def runner_a(prompt, *, options=None):
            call_order.append("a")
            yield AgentMessage.result("A done")

        async def runner_b(prompt, *, options=None):
            call_order.append("b")
            yield AgentMessage.result("B done")

        a = Agent("a", ISAAgentOptions(), runner=runner_a)
        b = Agent("b", ISAAgentOptions(), runner=runner_b)

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=a, description="A"),
                SwarmAgent(agent=b, description="B"),
            ],
            entry_agent="a",
        )

        # Run DAG with dependencies (sequential)
        result = await swarm.run_dag([
            {"id": "t1", "title": "Task 1", "description": "D1", "agent": "a"},
            {"id": "t2", "title": "Task 2", "description": "D2", "agent": "b", "depends_on": ["t1"]},
        ], parallel=True)

        # Verify execution order (t1 must complete before t2)
        assert call_order == ["a", "b"]

        # Verify all outputs collected
        assert "a:t1" in result.agent_outputs
        assert "b:t2" in result.agent_outputs
        assert len(result.messages) == 2
