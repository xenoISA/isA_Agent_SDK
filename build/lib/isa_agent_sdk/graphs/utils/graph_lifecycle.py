#!/usr/bin/env python3
"""
Graph Lifecycle Manager - Async initialization lifecycle for graph components

This module provides lifecycle management for the agent graph, enabling:
1. Async initialization of nodes that require setup (e.g., SenseNode with EventTriggerManager)
2. Graceful shutdown of resources
3. Coordinated startup/shutdown across all graph components

Follows the pattern established in LangGraph for component lifecycle management.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Initializable(Protocol):
    """Protocol for nodes that support async initialization"""

    async def initialize(self) -> None:
        """Initialize the node's async resources"""
        ...


@runtime_checkable
class Shutdownable(Protocol):
    """Protocol for nodes that support graceful shutdown"""

    async def shutdown(self) -> None:
        """Shutdown the node's resources gracefully"""
        ...


class GraphLifecycleManager:
    """
    Manages the lifecycle of graph nodes with async initialization.

    Provides coordinated startup and shutdown for nodes that require
    async setup (like SenseNode with EventTriggerManager).

    Example:
        manager = GraphLifecycleManager()
        manager.register_node("sense", sense_node)

        # Initialize all nodes
        await manager.initialize(config)

        # ... run graph ...

        # Shutdown all nodes
        await manager.shutdown()
    """

    def __init__(self):
        """Initialize the lifecycle manager"""
        self._nodes: Dict[str, Any] = {}
        self._initialized: bool = False
        self._shutdown_callbacks: List[callable] = []

    def register_node(self, name: str, node: Any) -> None:
        """
        Register a node for lifecycle management

        Args:
            name: Unique identifier for the node
            node: Node instance (may implement Initializable/Shutdownable)
        """
        self._nodes[name] = node
        logger.debug(f"Registered node '{name}' for lifecycle management")

    def register_shutdown_callback(self, callback: callable) -> None:
        """
        Register a callback to run during shutdown

        Args:
            callback: Async function to call during shutdown
        """
        self._shutdown_callbacks.append(callback)

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize all registered nodes that support initialization

        Args:
            config: Optional configuration to pass to nodes

        Returns:
            True if all initializations succeeded, False otherwise
        """
        if self._initialized:
            logger.debug("GraphLifecycleManager already initialized")
            return True

        config = config or {}
        success = True
        initialized_nodes = []

        logger.info(f"GraphLifecycleManager initializing {len(self._nodes)} nodes")

        for name, node in self._nodes.items():
            if isinstance(node, Initializable):
                try:
                    logger.debug(f"Initializing node: {name}")
                    await node.initialize()
                    initialized_nodes.append(name)
                    logger.info(f"Node '{name}' initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize node '{name}': {e}")
                    success = False
                    # Continue initializing other nodes

        self._initialized = success

        if success:
            logger.info(f"GraphLifecycleManager initialized: {initialized_nodes}")
        else:
            logger.warning(f"GraphLifecycleManager initialization had failures. Initialized: {initialized_nodes}")

        return success

    async def shutdown(self) -> bool:
        """
        Shutdown all registered nodes gracefully

        Returns:
            True if all shutdowns succeeded, False otherwise
        """
        if not self._initialized:
            logger.debug("GraphLifecycleManager not initialized, nothing to shutdown")
            return True

        success = True
        shutdown_nodes = []

        logger.info(f"GraphLifecycleManager shutting down {len(self._nodes)} nodes")

        # Shutdown nodes in reverse order of registration
        for name in reversed(list(self._nodes.keys())):
            node = self._nodes[name]
            if isinstance(node, Shutdownable):
                try:
                    logger.debug(f"Shutting down node: {name}")
                    await node.shutdown()
                    shutdown_nodes.append(name)
                    logger.info(f"Node '{name}' shutdown successfully")
                except Exception as e:
                    logger.error(f"Failed to shutdown node '{name}': {e}")
                    success = False
                    # Continue shutting down other nodes

        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
                success = False

        self._initialized = False

        if success:
            logger.info(f"GraphLifecycleManager shutdown complete: {shutdown_nodes}")
        else:
            logger.warning(f"GraphLifecycleManager shutdown had failures. Shutdown: {shutdown_nodes}")

        return success

    @property
    def is_initialized(self) -> bool:
        """Check if the lifecycle manager has been initialized"""
        return self._initialized

    def get_node(self, name: str) -> Optional[Any]:
        """
        Get a registered node by name

        Args:
            name: Node identifier

        Returns:
            Node instance or None if not found
        """
        return self._nodes.get(name)

    def has_node(self, name: str) -> bool:
        """
        Check if a node is registered

        Args:
            name: Node identifier

        Returns:
            True if node is registered
        """
        return name in self._nodes


# Singleton instance for global access
_lifecycle_manager: Optional[GraphLifecycleManager] = None


def get_lifecycle_manager() -> GraphLifecycleManager:
    """
    Get the singleton lifecycle manager instance

    Returns:
        GraphLifecycleManager instance
    """
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = GraphLifecycleManager()
    return _lifecycle_manager


async def initialize_graph_lifecycle(
    nodes: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> GraphLifecycleManager:
    """
    Convenience function to initialize graph lifecycle

    Args:
        nodes: Dictionary of node_name -> node_instance
        config: Optional configuration

    Returns:
        Initialized GraphLifecycleManager
    """
    manager = get_lifecycle_manager()

    for name, node in nodes.items():
        manager.register_node(name, node)

    await manager.initialize(config)
    return manager


async def shutdown_graph_lifecycle() -> bool:
    """
    Convenience function to shutdown graph lifecycle

    Returns:
        True if shutdown succeeded
    """
    global _lifecycle_manager
    if _lifecycle_manager is not None:
        result = await _lifecycle_manager.shutdown()
        _lifecycle_manager = None
        return result
    return True


__all__ = [
    "GraphLifecycleManager",
    "Initializable",
    "Shutdownable",
    "get_lifecycle_manager",
    "initialize_graph_lifecycle",
    "shutdown_graph_lifecycle",
]
