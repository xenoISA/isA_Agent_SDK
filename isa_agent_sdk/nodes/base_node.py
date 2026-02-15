"""
Base Node for all LangGraph nodes with dependency injection

This module provides base classes following Interface Segregation Principle (ISP):
- MinimalBaseNode: Core functionality only (execute, state/config)
- ToolExecutionNode: Tool execution + streaming (for ToolNode)
- ReasoningNode: Model calling + streaming (for ReasonNode, ResponseNode)
- BaseNode: DEPRECATED - inherits all mixins (backward compatibility)

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
import warnings
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


class MinimalBaseNode(ABC):
    """
    Minimal base class with only core node functionality (ISP-compliant)

    Use this for custom nodes that don't need MCP or model calling.
    Provides:
    - execute() entry point with error handling
    - _execute_logic() abstract method for subclass implementation
    - state/config properties for concurrency-safe access
    - MCP service initialization
    - Logging

    Does NOT provide:
    - MCP tool execution
    - Model calling
    - Streaming
    - Config extraction helpers

    For typical nodes, use ToolExecutionNode or ReasoningNode instead.
    """

    def __init__(self, node_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize minimal base node

        Args:
            node_name: Node name for logging
            logger: Optional logger instance
        """
        self.node_name = node_name
        self.logger = logger or agent_logger
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

        Note: Requires ConfigExtractorMixin.get_mcp_service() to be available.
        MinimalBaseNode does NOT include this - use ToolExecutionNode or ReasoningNode.

        Args:
            config: LangGraph RunnableConfig

        Returns:
            Initialized MCPClient instance or None if not available
        """
        # This will only work if subclass includes ConfigExtractorMixin
        if not hasattr(self, 'get_mcp_service'):
            self.logger.warning(
                f"{self.node_name} attempting to get MCP service but ConfigExtractorMixin not included. "
                f"Use ToolExecutionNode or ReasoningNode instead of MinimalBaseNode."
            )
            return None

        mcp_service = self.get_mcp_service(config)  # type: ignore
        if not mcp_service:
            return None

        # Initialize if needed
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


class ToolExecutionNode(
    MinimalBaseNode,
    ConfigExtractorMixin,
    MCPToolExecutionMixin,
    MCPSecurityMixin,
    StreamingMixin
):
    """
    Specialized base for tool execution nodes (ISP-compliant)

    Use this for nodes that execute MCP tools.
    Provides:
    - All MinimalBaseNode functionality
    - Config/context extraction (get_runtime_context, get_user_id, etc.)
    - MCP tool execution (mcp_call_tool, etc.)
    - MCP security (mcp_get_tool_security_levels, etc.)
    - Streaming (stream_tool, stream_custom)

    Example: ToolNode
    """
    pass


class ReasoningNode(
    MinimalBaseNode,
    ConfigExtractorMixin,
    ModelCallingMixin,
    StreamingMixin,
    MCPAssetsMixin
):
    """
    Specialized base for reasoning/response nodes (ISP-compliant)

    Use this for nodes that call LLMs for reasoning or responses.
    Provides:
    - All MinimalBaseNode functionality
    - Config/context extraction (get_runtime_context, get_user_id, etc.)
    - Model calling (call_model, _messages_to_prompt)
    - Streaming (stream_custom)
    - MCP assets (mcp_get_prompt, mcp_get_resource)

    Example: ReasonNode, ResponseNode
    """
    pass


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
    DEPRECATED: Use MinimalBaseNode, ToolExecutionNode, or ReasoningNode instead.

    This class violates the Interface Segregation Principle by forcing all nodes
    to inherit 7 mixins they may not need. It's kept for backward compatibility.

    Migration guide:
    - For tool execution nodes (ToolNode): Use ToolExecutionNode
    - For reasoning/response nodes (ReasonNode, ResponseNode): Use ReasoningNode
    - For custom minimal nodes: Use MinimalBaseNode

    Will be removed in version 2.0.0.

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
        Initialize base node (DEPRECATED)

        Args:
            node_name: Node name for logging and streaming updates
            logger: Optional logger instance
        """
        # Issue deprecation warning
        warnings.warn(
            f"BaseNode is deprecated and will be removed in version 2.0.0. "
            f"Use ToolExecutionNode for tool nodes, ReasoningNode for reasoning/response nodes, "
            f"or MinimalBaseNode for minimal custom nodes. "
            f"See https://github.com/yourorg/isa_agent_sdk/issues/44 for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )

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
