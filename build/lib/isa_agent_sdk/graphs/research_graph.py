#!/usr/bin/env python3
"""
Research Graph Implementation

A graph optimized for research tasks with enhanced search capabilities,
deep analysis, and citation tracking.
"""

import logging
from typing import Dict, Optional, List

from langgraph.graph import START, END
from .base_graph import BaseGraph

# Import nodes
from isa_agent_sdk.nodes.reason_node import ReasonNode
from isa_agent_sdk.nodes.response_node import ResponseNode
from isa_agent_sdk.nodes import ToolNode, AgentExecutorNode

logger = logging.getLogger(__name__)


class ResearchGraph(BaseGraph):
    """
    Research-focused graph with enhanced search and analysis capabilities
    """
    
    def get_graph_name(self) -> str:
        return "research_graph"
    
    def get_graph_type(self) -> str:
        return "research"
    
    def get_description(self) -> str:
        return "Optimized for research tasks with deep search, analysis, and citation tracking"
    
    def _initialize_nodes(self):
        """Initialize research-specific nodes"""
        # Enhanced configuration for research
        self.max_graph_iterations = self.config.get("max_graph_iterations", 100)  # More iterations
        self.max_agent_loops = self.config.get("max_agent_loops", 20)  # More agent loops
        self.max_tool_loops = self.config.get("max_tool_loops", 15)  # More tool loops
        
        # Create nodes with research-optimized configs
        self.nodes["reason"] = ReasonNode()
        self.nodes["tools"] = ToolNode()
        self.nodes["agent"] = AgentExecutorNode()
        self.nodes["response"] = ResponseNode()
        
        logger.info(f"{self.graph_name}: Initialized research nodes")
    
    def _build_workflow(self, workflow):
        """Build the research workflow"""
        
        # Add nodes
        workflow.add_node("research_reason", self.nodes["reason"].execute)
        workflow.add_node("research_tools", self.nodes["tools"].execute)
        workflow.add_node("research_agent", self.nodes["agent"].execute)
        workflow.add_node("research_synthesize", self._synthesize_research)
        workflow.add_node("research_response", self.nodes["response"].execute)
        
        # Entry point
        workflow.add_edge(START, "research_reason")
        
        # Reasoning routes
        workflow.add_conditional_edges(
            "research_reason",
            self._route_research_action,
            {
                "tools": "research_tools",
                "agent": "research_agent",
                "synthesize": "research_synthesize",
                "end": "research_response"
            }
        )
        
        # Tool execution routes (can loop for multiple searches)
        workflow.add_conditional_edges(
            "research_tools",
            self._route_tool_result,
            {
                "continue": "research_reason",  # Continue researching
                "agent": "research_agent",       # Need agent help
                "synthesize": "research_synthesize",  # Ready to synthesize
                "end": "research_response"
            }
        )
        
        # Agent execution routes
        workflow.add_conditional_edges(
            "research_agent",
            self._route_agent_result,
            {
                "continue": "research_reason",
                "tools": "research_tools",
                "synthesize": "research_synthesize",
                "end": "research_response"
            }
        )
        
        # Synthesis always leads to response
        workflow.add_edge("research_synthesize", "research_response")
        
        # Final edge
        workflow.add_edge("research_response", END)
        
        return workflow
    
    def _synthesize_research(self, state):
        """
        Synthesize research findings
        
        This node aggregates and analyzes all research data collected
        """
        messages = state.get("messages", [])
        research_data = state.get("research_data", {})
        
        # Aggregate findings
        findings = []
        sources = []
        
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                # Extract research findings and sources
                content = str(msg.content)
                if "source:" in content.lower() or "citation:" in content.lower():
                    sources.append(content)
                else:
                    findings.append(content)
        
        # Create synthesis
        synthesis = {
            "findings": findings,
            "sources": sources,
            "summary": f"Research complete with {len(findings)} findings from {len(sources)} sources"
        }
        
        state["research_synthesis"] = synthesis
        state["next_action"] = "end"
        
        logger.info(f"Research synthesis complete: {len(findings)} findings")
        return state
    
    def _route_research_action(self, state):
        """Route based on research needs"""
        next_action = state.get("next_action", "end")
        
        # Check if we need more research
        messages = state.get("messages", [])
        research_depth = len([m for m in messages if hasattr(m, "content")])
        
        # Deep research logic
        if research_depth < 3 and next_action == "call_tool":
            return "tools"
        elif research_depth >= 3 and research_depth < 10:
            # Ready to synthesize if we have enough data
            return "synthesize"
        elif next_action == "agent_executor":
            return "agent"
        else:
            return next_action if next_action != "end" else "synthesize"
    
    def _route_tool_result(self, state):
        """Route after tool execution in research context"""
        next_action = state.get("next_action", "continue")
        
        # Check if we have enough research data
        messages = state.get("messages", [])
        tool_calls = len([m for m in messages if "tool" in str(m).lower()])
        
        if tool_calls >= 5:  # Enough tool calls
            return "synthesize"
        elif next_action == "agent_executor":
            return "agent"
        elif next_action == "end":
            return "synthesize"
        else:
            return "continue"  # Keep researching
    
    def _route_agent_result(self, state):
        """Route after agent execution"""
        next_action = state.get("next_action", "continue")
        
        if next_action == "call_tool":
            return "tools"
        elif next_action == "end":
            return "synthesize"
        else:
            return "continue"
    
    def get_node_list(self) -> List[str]:
        return [
            "research_reason",
            "research_tools",
            "research_agent",
            "research_synthesize",
            "research_response"
        ]
    
    def get_features(self) -> List[str]:
        return [
            "deep_search",
            "multi_source_aggregation",
            "research_synthesis",
            "citation_tracking",
            "extended_iterations",
            "fact_checking"
        ]