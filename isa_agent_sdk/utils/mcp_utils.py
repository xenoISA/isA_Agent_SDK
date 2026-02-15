"""
MCP utility functions for proper resource management.

Provides context managers and helpers for working with MCP clients
in a safe, resource-leak-free manner.

Example:
    from isa_agent_sdk.utils.mcp_utils import get_initialized_mcp

    async with get_initialized_mcp(config=config) as mcp:
        result = await mcp.call_tool("search", {"query": "test"})
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.clients.mcp_client import MCPClient
from isa_agent_sdk.errors import ISAConnectionError


def extract_mcp_url(config: Optional[RunnableConfig]) -> Optional[str]:
    """Extract MCP URL from RunnableConfig.

    Looks for 'mcp_url' or 'isa_mcp_url' in the configurable dict.

    Args:
        config: Optional RunnableConfig to extract URL from

    Returns:
        MCP URL string or None if not found
    """
    if not config:
        return None
    configurable = config.get("configurable", {})
    return configurable.get("mcp_url") or configurable.get("isa_mcp_url")


@asynccontextmanager
async def get_initialized_mcp(
    mcp_url: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AsyncIterator[MCPClient]:
    """Get initialized MCP client with automatic cleanup.

    This context manager ensures the MCP client is properly initialized
    and cleaned up, preventing resource leaks.

    Usage:
        async with get_initialized_mcp(config=config) as mcp:
            result = await mcp.call_tool("search", {"query": "test"})

    Args:
        mcp_url: Optional explicit MCP URL
        config: Optional RunnableConfig to extract URL from
        session_id: Optional session ID for tracking
        user_id: Optional user ID for context

    Yields:
        Initialized MCPClient

    Raises:
        ISAConnectionError: If no MCP URL is configured or connection fails
    """
    url = mcp_url or extract_mcp_url(config)

    if not url:
        # Try to get from settings as fallback
        try:
            from isa_agent_sdk.core.config import settings
            url = settings.resolved_mcp_server_url
        except (ImportError, AttributeError):
            pass

    if not url:
        raise ISAConnectionError(
            "No MCP URL configured. Provide mcp_url argument, "
            "include it in config, or set ISA_MCP_SERVER_URL environment variable.",
            service="mcp"
        )

    client = MCPClient(
        mcp_url=url,
        session_id=session_id,
        user_id=user_id
    )

    try:
        await client.initialize()
        yield client
    except Exception as e:
        raise ISAConnectionError(
            f"MCP connection failed: {e}",
            service="mcp",
            url=url
        ) from e
    finally:
        await client.close()


__all__ = [
    "extract_mcp_url",
    "get_initialized_mcp",
]
