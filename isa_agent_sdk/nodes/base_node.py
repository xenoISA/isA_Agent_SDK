"""
Base Node for all LangGraph nodes with dependency injection

This module provides the BaseNode abstract base class that all LangGraph nodes inherit from.
Functionality is organized into mixins for maintainability:
- ConfigExtractorMixin: Config/context extraction methods
- MCPSearchMixin: MCP search operations
- MCPSecurityMixin: MCP security level operations
- MCPToolExecutionMixin: MCP tool execution
- MCPAssetsMixin: MCP prompt/resource retrieval
- StreamingMixin: Streaming methods
- ModelCallingMixin: LLM model calling
"""
from abc import ABC, abstractmethod
import contextvars
import logging
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.agent_types.agent_state import AgentState
from isa_agent_sdk.clients.mcp_client import MCPClient
from isa_agent_sdk.graphs.utils.context_schema import ContextSchema
from isa_agent_sdk.utils.logger import agent_logger

# Context variables for concurrency-safe state/config access
_node_state_var: contextvars.ContextVar[Optional[AgentState]] = contextvars.ContextVar(
    '_node_state_var', default=None
)
_node_config_var: contextvars.ContextVar[Optional[RunnableConfig]] = contextvars.ContextVar(
    '_node_config_var', default=None
)

# Import mixins
from .mixins import (
    ConfigExtractorMixin,
    MCPSearchMixin,
    MCPSecurityMixin,
    MCPToolExecutionMixin,
    MCPAssetsMixin,
    StreamingMixin,
    ModelCallingMixin,
)


class BaseNode(
    ConfigExtractorMixin,
    MCPSearchMixin,
    MCPSecurityMixin,
    MCPToolExecutionMixin,
    MCPAssetsMixin,
    StreamingMixin,
    ModelCallingMixin,
    ABC
):
    """
    Base class for all LangGraph nodes with dependency injection

    Features:
    - Dependency injection management (MCP service, session manager, etc.)
    - Unified logging and error handling
    - Streaming updates support
    - Runtime context management

    Methods are organized into mixins for maintainability:
    - ConfigExtractorMixin: Config/context extraction (get_runtime_context, get_user_id, etc.)
    - MCPSearchMixin: MCP search (mcp_search_all, mcp_search_tools, etc.)
    - MCPSecurityMixin: MCP security (mcp_get_tool_security_levels, etc.)
    - MCPToolExecutionMixin: Tool execution (mcp_call_tool, etc.)
    - MCPAssetsMixin: Asset retrieval (mcp_get_prompt, mcp_get_resource)
    - StreamingMixin: Streaming (stream_custom, stream_tool)
    - ModelCallingMixin: Model calling (call_model, _messages_to_prompt)
    """

    def __init__(self, node_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize base node

        Args:
            node_name: Node name for logging and streaming updates
            logger: Optional logger instance
        """
        self.node_name = node_name
        self.logger = logger or agent_logger

        # Cache dependency services to avoid repeated retrieval
        self._mcp_service_cache = None

    @property
    def state(self) -> Optional[AgentState]:
        """Get current state from context var (concurrency-safe)."""
        return _node_state_var.get(None)

    @state.setter
    def state(self, value: AgentState):
        """Set current state in context var (concurrency-safe)."""
        _node_state_var.set(value)

    @property
    def config(self) -> Optional[RunnableConfig]:
        """Get current config from context var (concurrency-safe)."""
        return _node_config_var.get(None)

    @config.setter
    def config(self, value: RunnableConfig):
        """Set current config in context var (concurrency-safe)."""
        _node_config_var.set(value)

    async def get_initialized_mcp_service(self, config: RunnableConfig) -> Optional[MCPClient]:
        """
        Get MCP service from config context with automatic initialization

        Args:
            config: LangGraph RunnableConfig

        Returns:
            Initialized MCPClient instance or None if not available
        """
        mcp_service = self.get_mcp_service(config)
        if not mcp_service:
            return None

        # Initialize if needed
        # MCPClient.session is a property that checks the underlying isa_mcp client
        if not mcp_service.session:
            await mcp_service.initialize()

        return mcp_service

    @abstractmethod
    async def _execute_logic(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Core execution logic to be implemented by subclasses

        Args:
            state: Current state
            config: LangGraph RunnableConfig with context in configurable

        Returns:
            Updated state
        """
        pass

    async def execute(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Unified execution entry point with config context support

        Args:
            state: Current state
            config: LangGraph RunnableConfig with context in configurable

        Returns:
            Updated state
        """
        try:
            # Store state and config for trace callbacks to access session_id
            self.state = state
            self.config = config

            # Simple logging - no streaming overhead
            self.logger.info(f"{self.node_name} starting")

            # Call subclass core logic with config context
            updated_state = await self._execute_logic(state, config)

            self.logger.info(f"{self.node_name} completed")
            return updated_state

        except Exception as e:
            self.logger.error(f"{self.node_name} error: {e}")
            raise
