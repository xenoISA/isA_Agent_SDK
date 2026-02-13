"""
Graph building modules for SmartAgent v3.0
"""

# Lazy imports to avoid circular dependencies

def __getattr__(name):
    """Lazy import for graph builders"""
    if name == "SmartAgentGraphBuilder":
        from .smart_agent_graph import SmartAgentGraphBuilder
        return SmartAgentGraphBuilder
    if name == "MultiAgentGraph":
        from .multi_agent_graph import MultiAgentGraph
        return MultiAgentGraph
    raise AttributeError(f"module 'isa_agent_sdk.graphs' has no attribute '{name}'")

__all__ = ["SmartAgentGraphBuilder", "MultiAgentGraph"]
