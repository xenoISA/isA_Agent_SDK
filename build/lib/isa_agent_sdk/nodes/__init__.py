"""
LangGraph nodes for OptimizedEnhancedMCPClient
Each node is separated for better modularity and testing
"""

# Lazy imports to avoid circular dependencies

def __getattr__(name):
    """Lazy import for node classes"""
    if name == "SenseNode":
        from .sense_node import SenseNode
        return SenseNode
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
    "SenseNode",
    "ToolNode",
    "AgentExecutorNode",
    "GuardrailNode",
    "FailsafeNode"
]
