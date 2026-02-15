"""
LangGraph nodes for OptimizedEnhancedMCPClient
Each node is separated for better modularity and testing
"""

# Lazy imports to avoid circular dependencies

def __getattr__(name):
    """Lazy import for node classes"""
    if name == "BaseNode":
        from .base_node import BaseNode
        return BaseNode
    elif name == "MinimalBaseNode":
        from .base_node import MinimalBaseNode
        return MinimalBaseNode
    elif name == "ToolExecutionNode":
        from .base_node import ToolExecutionNode
        return ToolExecutionNode
    elif name == "ReasoningNode":
        from .base_node import ReasoningNode
        return ReasoningNode
    elif name == "SenseNode":
        from .sense_node import SenseNode
        return SenseNode
    elif name == "ReasonNode":
        from .reason_node import ReasonNode
        return ReasonNode
    elif name == "ResponseNode":
        from .response_node import ResponseNode
        return ResponseNode
    elif name == "ToolNode":
        from .tool_node import ToolNode
        return ToolNode
    elif name == "AgentExecutorNode":
        from .agent_executor_node import AgentExecutorNode
        return AgentExecutorNode
    elif name == "GuardrailNode":
        from .guardrail_node import GuardrailNode
        return GuardrailNode
    elif name == "FailsafeNode":
        from .failsafe_node import FailsafeNode
        return FailsafeNode
    raise AttributeError(f"module 'isa_agent_sdk.nodes' has no attribute '{name}'")

__all__ = [
    "BaseNode",  # Deprecated - use MinimalBaseNode, ToolExecutionNode, or ReasoningNode
    "MinimalBaseNode",
    "ToolExecutionNode",
    "ReasoningNode",
    "SenseNode",
    "ReasonNode",
    "ResponseNode",
    "ToolNode",
    "AgentExecutorNode",
    "GuardrailNode",
    "FailsafeNode"
]
