#!/usr/bin/env python3
"""
LangGraph Config Context Schema Definition

This module defines the TypedDict schema for LangGraph config context,
specifying the structure of configuration data that can be passed to
graph execution at runtime through config["configurable"].

Features:
- Type-safe context definition using TypedDict
- Clear documentation of each context field
- Support for MCP service integration
- Memory and session management
- Default prompts and resources configuration

Context Schema Fields:
- user_id: User identifier for session management
- thread_id: Thread/session identifier for state persistence
- mcp_service: MCP service instance for tool/prompt/resource access
- session_manager: Session state management instance
- memory_context: Aggregated memory context string
- default_prompts: Cached default prompts from MCP
- default_tools: Available default tools
- default_resources: Global resources accessible to all users
- runtime_initialized: Initialization status flag

Usage in LangGraph (Current Version):
1. Define schema: StateGraph(State, config_schema=ContextSchema)
2. Pass context: graph.invoke(state, config={"configurable": {"user_id": "...", ...}})
3. Access in nodes: def node(state, config: RunnableConfig)
4. Use context: config["configurable"]["user_id"]
"""

from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict
from isa_agent_sdk.clients.mcp_client import MCPClient


class ContextSchema(TypedDict):
    """
    Runtime context schema for LangGraph execution
    
    This schema defines all the configuration and dependency injection
    data that can be passed to graph nodes at runtime.
    """
    
    # Core identifiers
    user_id: str
    """User identifier (UUID format recommended)"""
    
    thread_id: str  
    """Thread/session identifier for state persistence"""
    
    # Service dependencies
    mcp_service: MCPClient
    """MCP service instance for tool/prompt/resource operations"""
    
    session_manager: Optional[Any]
    """Session management instance (type varies by implementation)"""
    
    # Default data from MCP (cached for performance)
    default_prompts: Dict[str, str]
    """Cached default prompts by node type (e.g., 'entry_node_prompt')"""
    
    default_tools: List[Dict[str, Any]]
    """List of default tools available to all sessions"""
    
    default_resources: List[Dict[str, Any]]
    """List of global resources accessible to all users"""
    
    # Runtime status and metadata
    runtime_initialized: bool
    """Flag indicating if runtime context was properly initialized"""
    
    cache_status: Dict[str, Any]
    """MCP cache status and loading information"""


class MinimalContextSchema(TypedDict):
    """
    Minimal context schema for basic graph execution
    
    Use this when you only need essential identifiers and MCP service,
    without the full default data caching.
    """
    
    user_id: str
    thread_id: str
    mcp_service: MCPClient
    runtime_initialized: bool


class ExtendedContextSchema(ContextSchema):
    """
    Extended context schema with additional optional fields
    
    Use this for advanced use cases that need extra context data.
    """
    
    # User-specific data (optional)
    user_tools: Optional[List[Dict[str, Any]]]
    """User-specific tools (loaded on demand)"""
    
    user_resources: Optional[List[Dict[str, Any]]]
    """User-specific resources (loaded on demand)"""
    
    # Execution context
    execution_mode: Optional[str]
    """Execution mode: 'interactive', 'autonomous', 'background'"""
    
    trace_id: Optional[str]
    """Trace ID for request tracking and debugging"""
    
    # Custom context data
    custom_data: Optional[Dict[str, Any]]
    """Additional custom context data"""


# Export the main schema as default
__all__ = ['ContextSchema', 'MinimalContextSchema', 'ExtendedContextSchema']