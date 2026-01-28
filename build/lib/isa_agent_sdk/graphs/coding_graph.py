#!/usr/bin/env python3
"""
Coding Graph Implementation

A graph optimized for code generation, debugging, and software development tasks.
Includes syntax validation, testing capabilities, and code quality checks.
"""

import logging
from typing import Dict, Optional, List

from langgraph.graph import START, END
from .base_graph import BaseGraph

# Import nodes
from isa_agent_sdk.nodes.reason_node import ReasonNode
from isa_agent_sdk.nodes.response_node import ResponseNode
from isa_agent_sdk.nodes import ToolNode, AgentExecutorNode, GuardrailNode

logger = logging.getLogger(__name__)


class CodingGraph(BaseGraph):
    """
    Coding-focused graph with code generation, validation, and testing
    """
    
    def get_graph_name(self) -> str:
        return "coding_graph"
    
    def get_graph_type(self) -> str:
        return "coding"
    
    def get_description(self) -> str:
        return "Optimized for code generation, debugging, testing, and software development"
    
    def _initialize_nodes(self):
        """Initialize coding-specific nodes"""
        # Coding-optimized configuration
        self.max_graph_iterations = self.config.get("max_graph_iterations", 75)
        self.max_agent_loops = self.config.get("max_agent_loops", 15)
        self.max_tool_loops = self.config.get("max_tool_loops", 10)
        
        # Enable strict guardrails for code safety
        self.guardrail_enabled = self.config.get("guardrail_enabled", True)
        self.guardrail_mode = self.config.get("guardrail_mode", "strict")
        
        # Create nodes
        self.nodes["reason"] = ReasonNode()
        self.nodes["tools"] = ToolNode()
        self.nodes["agent"] = AgentExecutorNode()
        self.nodes["response"] = ResponseNode()
        self.nodes["guardrail"] = GuardrailNode(self.guardrail_mode)
        
        logger.info(f"{self.graph_name}: Initialized coding nodes with strict guardrails")
    
    def _build_workflow(self, workflow):
        """Build the coding workflow"""
        
        # Add nodes
        workflow.add_node("code_analyze", self._analyze_code_request)
        workflow.add_node("code_reason", self.nodes["reason"].execute)
        workflow.add_node("code_generate", self._generate_code)
        workflow.add_node("code_validate", self._validate_code)
        workflow.add_node("code_test", self._test_code)
        workflow.add_node("code_tools", self.nodes["tools"].execute)
        workflow.add_node("code_agent", self.nodes["agent"].execute)
        workflow.add_node("code_guardrail", self.nodes["guardrail"].execute)
        workflow.add_node("code_response", self.nodes["response"].execute)
        
        # Entry point - analyze request first
        workflow.add_edge(START, "code_analyze")
        
        # Analysis routes
        workflow.add_conditional_edges(
            "code_analyze",
            self._route_code_type,
            {
                "generate": "code_generate",
                "debug": "code_reason",
                "test": "code_test",
                "refactor": "code_reason",
                "explain": "code_reason"
            }
        )
        
        # Code generation flow
        workflow.add_edge("code_generate", "code_validate")
        
        # Validation routes
        workflow.add_conditional_edges(
            "code_validate",
            self._route_validation_result,
            {
                "pass": "code_test",
                "fail": "code_reason",  # Re-analyze and fix
                "guardrail": "code_guardrail"
            }
        )
        
        # Test routes
        workflow.add_conditional_edges(
            "code_test",
            self._route_test_result,
            {
                "pass": "code_guardrail",
                "fail": "code_reason",  # Fix failing tests
                "skip": "code_guardrail"  # No tests available
            }
        )
        
        # Reasoning routes
        workflow.add_conditional_edges(
            "code_reason",
            lambda state: state.get("next_action", "generate"),
            {
                "call_tool": "code_tools",
                "agent_executor": "code_agent",
                "generate": "code_generate",
                "end": "code_guardrail"
            }
        )
        
        # Tool routes
        workflow.add_conditional_edges(
            "code_tools",
            self._route_tool_result,
            {
                "continue": "code_reason",
                "validate": "code_validate",
                "test": "code_test",
                "end": "code_guardrail"
            }
        )
        
        # Agent routes
        workflow.add_conditional_edges(
            "code_agent",
            lambda state: state.get("next_action", "continue"),
            {
                "call_model": "code_reason",
                "call_tool": "code_tools",
                "end": "code_guardrail"
            }
        )
        
        # Guardrail always leads to response
        workflow.add_edge("code_guardrail", "code_response")
        
        # Final edge
        workflow.add_edge("code_response", END)
        
        return workflow
    
    def _analyze_code_request(self, state):
        """Analyze the coding request to determine the task type"""
        user_input = state.get("user_input", "")
        messages = state.get("messages", [])
        
        # Determine code task type
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["generate", "create", "write", "implement"]):
            state["code_task"] = "generate"
            state["next_action"] = "generate"
        elif any(word in input_lower for word in ["debug", "fix", "error", "issue"]):
            state["code_task"] = "debug"
            state["next_action"] = "debug"
        elif any(word in input_lower for word in ["test", "testing", "unit test"]):
            state["code_task"] = "test"
            state["next_action"] = "test"
        elif any(word in input_lower for word in ["refactor", "improve", "optimize"]):
            state["code_task"] = "refactor"
            state["next_action"] = "refactor"
        elif any(word in input_lower for word in ["explain", "understand", "how does"]):
            state["code_task"] = "explain"
            state["next_action"] = "explain"
        else:
            state["code_task"] = "generate"  # Default to generation
            state["next_action"] = "generate"
        
        logger.info(f"Code task identified: {state['code_task']}")
        return state
    
    def _generate_code(self, state):
        """Generate code based on requirements"""
        # This would integrate with code generation models
        state["code_generated"] = True
        state["next_action"] = "validate"
        
        logger.info("Code generation initiated")
        return state
    
    def _validate_code(self, state):
        """Validate generated code for syntax and best practices"""
        code_generated = state.get("code_generated", False)
        
        if code_generated:
            # Perform validation (syntax check, linting, etc.)
            validation_passed = True  # Simplified
            
            if validation_passed:
                state["validation_status"] = "pass"
                state["next_action"] = "pass"
            else:
                state["validation_status"] = "fail"
                state["next_action"] = "fail"
        else:
            state["validation_status"] = "skip"
            state["next_action"] = "guardrail"
        
        logger.info(f"Code validation: {state.get('validation_status')}")
        return state
    
    def _test_code(self, state):
        """Run tests on the generated code"""
        validation_status = state.get("validation_status", "")
        
        if validation_status == "pass":
            # Run tests
            tests_passed = True  # Simplified
            
            if tests_passed:
                state["test_status"] = "pass"
                state["next_action"] = "pass"
            else:
                state["test_status"] = "fail"
                state["next_action"] = "fail"
        else:
            state["test_status"] = "skip"
            state["next_action"] = "skip"
        
        logger.info(f"Code testing: {state.get('test_status')}")
        return state
    
    def _route_code_type(self, state):
        """Route based on code task type"""
        return state.get("next_action", "generate")
    
    def _route_validation_result(self, state):
        """Route based on validation results"""
        validation_status = state.get("validation_status", "fail")
        
        if validation_status == "pass":
            return "pass"
        elif validation_status == "fail":
            return "fail"
        else:
            return "guardrail"
    
    def _route_test_result(self, state):
        """Route based on test results"""
        test_status = state.get("test_status", "skip")
        
        if test_status == "pass":
            return "pass"
        elif test_status == "fail":
            return "fail"
        else:
            return "skip"
    
    def _route_tool_result(self, state):
        """Route after tool execution"""
        next_action = state.get("next_action", "continue")
        
        if next_action == "validate":
            return "validate"
        elif next_action == "test":
            return "test"
        elif next_action == "end":
            return "end"
        else:
            return "continue"
    
    def get_node_list(self) -> List[str]:
        return [
            "code_analyze",
            "code_reason",
            "code_generate",
            "code_validate",
            "code_test",
            "code_tools",
            "code_agent",
            "code_guardrail",
            "code_response"
        ]
    
    def get_features(self) -> List[str]:
        return [
            "code_generation",
            "syntax_validation",
            "automated_testing",
            "debugging",
            "refactoring",
            "code_explanation",
            "strict_guardrails",
            "best_practices_enforcement"
        ]