#!/usr/bin/env python3
"""
Multi-Agent Graph Builder

Composes multiple LangGraph subgraphs into a single supervisor workflow
with explicit handoff and shared state.

Pattern:
START -> router -> agent_x -> router -> agent_y -> ... -> END

Each agent node invokes its subgraph and returns state updates that are
merged into the shared AgentState. Handoffs are controlled via:
- state["next_agent"] (preferred)
- state["next_action"] (fallback)
"""

from typing import Dict, Optional, Any, List, Callable, Awaitable

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig

from isa_agent_sdk.agent_types import AgentState
from .base_graph import BaseGraph
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger


AgentCallable = Callable[[AgentState, RunnableConfig], Awaitable[Dict[str, Any]]]


class MultiAgentGraph(BaseGraph):
    """
    Multi-agent supervisor graph that orchestrates subgraphs.

    Args:
        agent_graphs: Mapping of agent name -> compiled LangGraph (invoke/ainvoke)
                      or async callable(state, config) -> updates dict.
        entry_agent: First agent to run if no routing info is present.
        config: Optional graph config (recursion limits, etc.)
    """

    def __init__(
        self,
        agent_graphs: Dict[str, Any],
        entry_agent: str,
        config: Optional[Dict] = None,
    ):
        self.agent_graphs = agent_graphs
        self.entry_agent = entry_agent
        super().__init__(config=config)

    def get_graph_name(self) -> str:
        return "multi_agent_graph"

    def get_graph_type(self) -> str:
        return "multi_agent"

    def get_description(self) -> str:
        return "Supervisor graph that routes between agent subgraphs with handoffs"

    def _initialize_nodes(self):
        # Nodes are created dynamically in _build_workflow
        self.nodes = {}

    def _build_workflow(self, workflow: StateGraph) -> StateGraph:
        # Router node decides which agent to run next
        workflow.add_node("router", self._route_node)

        # Agent nodes
        for agent_name in self.agent_graphs.keys():
            workflow.add_node(agent_name, self._make_agent_node(agent_name))
            workflow.add_edge(agent_name, "router")

        # Entry point
        workflow.add_edge(START, "router")

        # Routing logic from router
        workflow.add_conditional_edges("router", self._route_next_agent)

        return workflow

    def get_node_list(self) -> List[str]:
        return ["router"] + list(self.agent_graphs.keys())

    def get_features(self) -> List[str]:
        return [
            "multi_agent_orchestration",
            "explicit_handoff",
            "shared_state",
            "subgraph_composition",
        ]

    # ============================
    # Router
    # ============================

    async def _route_node(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Router node does not modify state; it only determines next step."""
        return {}

    def _route_next_agent(self, state: AgentState) -> str:
        """
        Routing decision:
        - next_agent (preferred)
        - next_action (fallback)
        - entry_agent (default)
        - END (if next_agent == "end" or "done")
        """
        next_agent = state.get("next_agent") or state.get("next_action")

        if not next_agent:
            next_agent = self.entry_agent

        if isinstance(next_agent, str) and next_agent.lower() in {"end", "done", "finish", "complete"}:
            return END

        if next_agent not in self.agent_graphs:
            logger.warning(f"Unknown next_agent '{next_agent}', falling back to entry_agent '{self.entry_agent}'")
            return self.entry_agent

        return next_agent

    # ============================
    # Agent execution
    # ============================

    def _make_agent_node(self, agent_name: str):
        async def _agent_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
            result = await self._invoke_subgraph(agent_name, state, config)

            updates: Dict[str, Any] = {}
            if isinstance(result, dict):
                updates.update(result)
            else:
                updates["agent_outputs"] = {agent_name: result}

            # Always track active agent and outputs
            updates["active_agent"] = agent_name
            outputs = updates.get("agent_outputs") or {}
            if not isinstance(outputs, dict):
                outputs = {agent_name: outputs}
            else:
                outputs = {**outputs, agent_name: result}
            updates["agent_outputs"] = outputs

            return updates

        return _agent_node

    async def _invoke_subgraph(
        self,
        agent_name: str,
        state: AgentState,
        config: RunnableConfig,
    ) -> Any:
        graph = self.agent_graphs[agent_name]

        # Compiled LangGraph exposes ainvoke/invoke
        if hasattr(graph, "ainvoke"):
            return await graph.ainvoke(state, config=config)
        if hasattr(graph, "invoke"):
            return graph.invoke(state, config=config)

        # Callable graph
        if callable(graph):
            result = graph(state, config)
            if hasattr(result, "__await__"):
                return await result
            return result

        raise TypeError(f"Unsupported agent graph type for '{agent_name}': {type(graph)}")
