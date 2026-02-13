"""
Swarm multi-agent orchestrator.

Provides dynamic agent handoff orchestration where agents decide when to
hand off control based on their specialization. Inspired by LangGraph Swarm.

Agents are complete Agent instances. The swarm orchestrates *across* agents
without modifying their internals. Handoff works via system prompt injection
(teach agents about peers) + response parsing (detect [HANDOFF: agent_name]).
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from isa_agent_sdk._messages import AgentMessage
from isa_agent_sdk.options import ISAAgentOptions, SystemPromptConfig

from .agent import Agent, AgentRunResult
from .handoff import (
    build_handoff_system_prompt_section,
    parse_handoff_from_response,
    strip_handoff_directive,
)
from .swarm_types import (
    HandoffAction,
    SwarmAgent,
    SwarmRunResult,
    SwarmState,
)

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """
    Orchestrate multiple agents with dynamic handoff.

    Agents decide when to hand off control by emitting [HANDOFF: agent_name]
    directives in their responses. The orchestrator injects peer information
    into each agent's system prompt and parses responses for handoff signals.

    Args:
        agents: List of SwarmAgent wrappers, or Dict[str, Agent] for convenience
        entry_agent: Name of the first agent to run (defaults to first agent)
        max_handoffs: Maximum number of handoffs before forcing completion
        shared_state: Initial shared state passed across agents

    Example::

        swarm = SwarmOrchestrator(
            agents=[
                SwarmAgent(agent=researcher, description="Research expert"),
                SwarmAgent(agent=writer, description="Technical writer"),
            ],
            entry_agent="researcher",
        )
        result = await swarm.run("Research AI agents and write a summary")
    """

    def __init__(
        self,
        agents: Union[List[SwarmAgent], Dict[str, Agent]],
        entry_agent: Optional[str] = None,
        max_handoffs: int = 10,
        shared_state: Optional[Dict[str, Any]] = None,
    ):
        # Normalize to dict of SwarmAgent
        if isinstance(agents, dict):
            self._agents: Dict[str, SwarmAgent] = {
                name: SwarmAgent(agent=agent, description=agent.name)
                for name, agent in agents.items()
            }
        elif isinstance(agents, list):
            if not agents:
                raise ValueError("agents list cannot be empty")
            self._agents = {sa.name: sa for sa in agents}
        else:
            raise TypeError(f"agents must be List[SwarmAgent] or Dict[str, Agent], got {type(agents)}")

        if not self._agents:
            raise ValueError("agents cannot be empty")

        self.entry_agent = entry_agent or next(iter(self._agents.keys()))
        if self.entry_agent not in self._agents:
            raise ValueError(
                f"entry_agent '{self.entry_agent}' not found. "
                f"Available: {list(self._agents.keys())}"
            )

        self.max_handoffs = max_handoffs
        self._initial_shared_state = shared_state or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        state: Optional[SwarmState] = None,
    ) -> SwarmRunResult:
        """Run the swarm and return the final result.

        Args:
            prompt: User prompt to start with
            state: Optional pre-existing SwarmState to resume from

        Returns:
            SwarmRunResult with final text, messages, handoff trace, etc.
        """
        swarm_state = state or SwarmState(
            active_agent=self.entry_agent,
            shared_state=dict(self._initial_shared_state),
        )

        agent_outputs: Dict[str, AgentRunResult] = {}
        all_messages: List[AgentMessage] = []
        current_prompt = prompt

        for _ in range(self.max_handoffs + 1):
            active_name = swarm_state.active_agent
            swarm_agent = self._agents.get(active_name)
            if not swarm_agent:
                raise KeyError(
                    f"Unknown agent '{active_name}'. "
                    f"Available: {list(self._agents.keys())}"
                )

            logger.debug("Swarm: running agent '%s'", active_name)

            # Build options with handoff prompt injected
            modified_options = self._build_agent_options(
                swarm_agent, swarm_state, current_prompt,
            )

            # Run the agent
            result = await swarm_agent.agent.run(
                current_prompt, options=modified_options,
            )
            agent_outputs[active_name] = result
            all_messages.extend(result.messages)

            # Merge shared state
            if result.shared_state:
                swarm_state.shared_state.update(result.shared_state)

            # Record conversation turn
            swarm_state.conversation_history.append({
                "agent": active_name,
                "prompt": current_prompt,
                "response": result.text,
            })

            # Parse response for handoff
            peer_names = {
                n for n in self._agents if n != active_name
            }
            handoff = parse_handoff_from_response(result.text, peer_names)

            if handoff.action == HandoffAction.HANDOFF:
                swarm_state.record_handoff(
                    active_name, handoff.target_agent, handoff.reason or "",
                )
                current_prompt = self._build_handoff_prompt(
                    prompt, handoff, swarm_state,
                )
                logger.debug(
                    "Swarm: handoff from '%s' to '%s' — %s",
                    active_name, handoff.target_agent, handoff.reason,
                )
            else:
                # COMPLETE — strip directive and return
                clean_text = strip_handoff_directive(result.text)
                return SwarmRunResult(
                    text=clean_text,
                    messages=all_messages,
                    final_agent=active_name,
                    handoff_trace=swarm_state.handoff_trace,
                    shared_state=swarm_state.shared_state,
                    agent_outputs=agent_outputs,
                )

        # Max handoffs reached
        logger.warning(
            "Swarm: max handoffs (%d) reached. Returning last result.",
            self.max_handoffs,
        )
        last_agent = swarm_state.active_agent
        last_output = agent_outputs.get(last_agent)
        return SwarmRunResult(
            text=strip_handoff_directive(last_output.text) if last_output else "",
            messages=all_messages,
            final_agent=last_agent,
            handoff_trace=swarm_state.handoff_trace,
            shared_state=swarm_state.shared_state,
            agent_outputs=agent_outputs,
        )

    async def stream(
        self,
        prompt: str,
        *,
        state: Optional[SwarmState] = None,
    ) -> AsyncIterator[AgentMessage]:
        """Stream messages from the swarm with agent annotations.

        Yields AgentMessage instances annotated with metadata indicating
        which agent produced them and swarm lifecycle events.

        Args:
            prompt: User prompt to start with
            state: Optional pre-existing SwarmState to resume from

        Yields:
            AgentMessage instances with metadata["agent"] set
        """
        swarm_state = state or SwarmState(
            active_agent=self.entry_agent,
            shared_state=dict(self._initial_shared_state),
        )

        current_prompt = prompt

        for _ in range(self.max_handoffs + 1):
            active_name = swarm_state.active_agent
            swarm_agent = self._agents.get(active_name)
            if not swarm_agent:
                raise KeyError(
                    f"Unknown agent '{active_name}'. "
                    f"Available: {list(self._agents.keys())}"
                )

            # Emit agent start event
            yield AgentMessage(
                type="system",
                content=f"Agent '{active_name}' started",
                metadata={"event": "swarm_agent_start", "agent": active_name},
            )

            modified_options = self._build_agent_options(
                swarm_agent, swarm_state, current_prompt,
            )

            # Stream and collect text for handoff parsing
            collected_text = ""
            async for msg in swarm_agent.agent.stream(
                current_prompt, options=modified_options,
            ):
                msg.metadata["agent"] = active_name
                yield msg

                if msg.is_complete and msg.content:
                    collected_text = msg.content
                elif msg.is_text and msg.content:
                    collected_text += msg.content

            # Record conversation turn
            swarm_state.conversation_history.append({
                "agent": active_name,
                "prompt": current_prompt,
                "response": collected_text,
            })

            # Parse for handoff
            peer_names = {n for n in self._agents if n != active_name}
            handoff = parse_handoff_from_response(collected_text, peer_names)

            if handoff.action == HandoffAction.HANDOFF:
                swarm_state.record_handoff(
                    active_name, handoff.target_agent, handoff.reason or "",
                )

                yield AgentMessage(
                    type="system",
                    content=f"Handoff from '{active_name}' to '{handoff.target_agent}'",
                    metadata={
                        "event": "swarm_handoff",
                        "from_agent": active_name,
                        "to_agent": handoff.target_agent,
                        "reason": handoff.reason or "",
                    },
                )

                current_prompt = self._build_handoff_prompt(
                    prompt, handoff, swarm_state,
                )
            else:
                # Done
                return

        logger.warning(
            "Swarm stream: max handoffs (%d) reached.", self.max_handoffs,
        )

    async def run_dag(
        self,
        tasks: List[Dict[str, Any]],
        *,
        shared_state: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
    ) -> SwarmRunResult:
        """Execute tasks as a DAG with agent assignments.

        Each task dict should include an ``agent`` field specifying which
        SwarmAgent should execute it. Tasks within the same wavefront that
        belong to different agents run concurrently; same-agent tasks within
        a wavefront run sequentially.

        Args:
            tasks: List of task dicts with id, title, description, agent, depends_on
            shared_state: Optional initial shared state

        Returns:
            SwarmRunResult with combined outputs
        """
        from isa_agent_sdk.dag import DAGScheduler

        dag = DAGScheduler.build_dag(tasks)
        errors = DAGScheduler.validate(dag)
        if errors:
            raise ValueError(f"DAG validation failed: {'; '.join(errors)}")

        wavefronts = DAGScheduler.compute_wavefronts(dag)

        state = shared_state or dict(self._initial_shared_state)
        agent_outputs: Dict[str, AgentRunResult] = {}
        all_messages: List[AgentMessage] = []
        task_results: Dict[str, str] = {}

        for wavefront in wavefronts:
            # Group tasks by assigned agent
            agent_tasks: Dict[str, List[str]] = {}
            for task_id in wavefront:
                dag_task = dag.tasks[task_id]
                agent_name = dag_task.metadata.get("agent", self.entry_agent)
                if agent_name not in self._agents:
                    DAGScheduler.mark_task_failed(
                        dag, task_id,
                        f"Unknown agent '{agent_name}'",
                    )
                    continue
                agent_tasks.setdefault(agent_name, []).append(task_id)

            async def _run_agent_tasks(
                agent_name: str, task_ids: List[str],
            ) -> None:
                swarm_agent = self._agents[agent_name]
                for task_id in task_ids:
                    dag_task = dag.tasks[task_id]
                    dag_task.status = _dag_running_status()

                    task_prompt = self._build_dag_task_prompt(
                        dag_task, task_results,
                    )

                    try:
                        result = await swarm_agent.agent.run(task_prompt)
                        DAGScheduler.mark_task_completed(dag, task_id, result.text)
                        task_results[task_id] = result.text
                        agent_outputs[f"{agent_name}:{task_id}"] = result
                        all_messages.extend(result.messages)
                        if result.shared_state:
                            state.update(result.shared_state)
                    except Exception as exc:
                        DAGScheduler.mark_task_failed(dag, task_id, str(exc))
                        logger.error(
                            "DAG task '%s' (agent '%s') failed: %s",
                            task_id, agent_name, exc,
                        )

            if parallel:
                await asyncio.gather(
                    *[
                        _run_agent_tasks(agent_name, task_ids)
                        for agent_name, task_ids in agent_tasks.items()
                    ]
                )
            else:
                for agent_name, task_ids in agent_tasks.items():
                    await _run_agent_tasks(agent_name, task_ids)

        # Build summary text from final wavefront results
        final_texts = []
        for task_id in (wavefronts[-1] if wavefronts else []):
            if task_id in task_results:
                final_texts.append(task_results[task_id])

        return SwarmRunResult(
            text="\n\n".join(final_texts),
            messages=all_messages,
            final_agent=self.entry_agent,
            handoff_trace=[],
            shared_state=state,
            agent_outputs=agent_outputs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_agent_options(
        self,
        swarm_agent: SwarmAgent,
        swarm_state: SwarmState,
        prompt: str,
    ) -> ISAAgentOptions:
        """Clone agent options with handoff prompt section injected."""
        # Build conversation summary from last 3 turns
        summary = None
        recent = swarm_state.conversation_history[-3:]
        if recent:
            lines = []
            for turn in recent:
                snippet = (turn["response"] or "")[:200]
                lines.append(f"- **{turn['agent']}**: {snippet}")
            summary = "\n".join(lines)

        handoff_section = build_handoff_system_prompt_section(
            swarm_agent.name,
            list(self._agents.values()),
            conversation_summary=summary,
        )

        if not handoff_section:
            return swarm_agent.agent._clone_options()

        current_sp = swarm_agent.agent.options.system_prompt

        if current_sp is None:
            new_sp = handoff_section
        elif isinstance(current_sp, str):
            new_sp = current_sp + handoff_section
        elif isinstance(current_sp, SystemPromptConfig):
            if current_sp.replace:
                new_sp = SystemPromptConfig(
                    replace=current_sp.replace + handoff_section,
                )
            else:
                existing_append = current_sp.append or ""
                new_sp = SystemPromptConfig(
                    preset=current_sp.preset,
                    append=existing_append + handoff_section,
                )
        else:
            new_sp = handoff_section

        return swarm_agent.agent._clone_options({"system_prompt": new_sp})

    @staticmethod
    def _build_handoff_prompt(
        original_prompt: str,
        handoff: Any,
        swarm_state: SwarmState,
    ) -> str:
        """Build the prompt for the next agent after a handoff."""
        parts = [f"Original request: {original_prompt}"]

        if handoff.context_for_next:
            context = handoff.context_for_next
            if len(context) > 2000:
                context = "[...context truncated...]\n" + context[-1970:]
            parts.append(f"\nContext from previous agent:\n{context}")

        if handoff.reason:
            parts.append(f"\nHandoff reason: {handoff.reason}")

        return "\n".join(parts)

    @staticmethod
    def _build_dag_task_prompt(
        dag_task: Any,
        prior_results: Dict[str, str],
    ) -> str:
        """Build prompt for a DAG task, including dependency results."""
        parts = [f"Task: {dag_task.title}", f"\n{dag_task.description}"]

        dep_results = []
        for dep_id in dag_task.depends_on:
            if dep_id in prior_results:
                result_text = prior_results[dep_id]
                if len(result_text) > 1000:
                    result_text = result_text[:1000] + "..."
                dep_results.append(f"- [{dep_id}]: {result_text}")

        if dep_results:
            parts.append("\nResults from prerequisite tasks:")
            parts.extend(dep_results)

        return "\n".join(parts)


def _dag_running_status():
    """Get the RUNNING status from the DAG module (lazy import)."""
    from isa_agent_sdk.dag import DAGTaskStatus
    return DAGTaskStatus.RUNNING
