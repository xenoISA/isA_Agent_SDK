#!/usr/bin/env python3
"""
Desktop Execution Client - Routes tool execution to user's desktop via Pool Manager

This client connects to the Pool Manager's desktop gateway to execute tools
on the user's registered desktop agent.

Flow:
    isA Agent SDK → Pool Manager (/desktop/execute) → Desktop Agent → Local Filesystem

Usage:
    from isa_agent_sdk.clients import DesktopExecutionClient, get_desktop_client

    client = await get_desktop_client(user_id="xenodennis")
    result = await client.execute_tool("read_file", {"file_path": "/path/to/file"})
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import aiohttp

logger = logging.getLogger(__name__)


class DesktopExecutionClient:
    """
    Client for executing tools on user's desktop via Pool Manager.

    The Pool Manager maintains WebSocket connections to desktop agents.
    This client uses the HTTP API to route tool calls through that gateway.
    """

    def __init__(
        self,
        pool_manager_url: str = "http://localhost:8090",
        request_timeout: int = 120,
    ):
        """
        Initialize desktop execution client.

        Args:
            pool_manager_url: Pool Manager service URL
            request_timeout: Timeout for tool execution (default 120s for long operations)
        """
        self.pool_manager_url = pool_manager_url.rstrip('/')
        self.request_timeout = request_timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._desktop_id: Optional[str] = None

        logger.info(f"DesktopExecutionClient initialized: {pool_manager_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                    self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def get_user_desktop(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the user's online desktop.

        Args:
            user_id: Owner ID to find desktop for

        Returns:
            Desktop info dict or None if no desktop found
        """
        session = await self._get_session()

        try:
            # First check online desktops
            async with session.get(
                f"{self.pool_manager_url}/desktop/online"
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to get online desktops: {resp.status}")
                    return None

                online_data = await resp.json()
                online_ids = set(online_data.get("desktop_ids", []))

            # Get user's desktops
            async with session.get(
                f"{self.pool_manager_url}/admin/desktops",
                params={"owner_id": user_id}
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to get user desktops: {resp.status}")
                    return None

                data = await resp.json()
                desktops = data.get("desktops", [])

            # Find first online desktop for this user
            for desktop in desktops:
                desktop_id = desktop.get("desktop_id")
                if desktop_id in online_ids:
                    logger.info(f"Found online desktop for {user_id}: {desktop_id}")
                    return desktop

            logger.warning(f"No online desktop found for user {user_id}")
            return None

        except Exception as e:
            logger.error(f"Error finding user desktop: {e}")
            return None

    async def acquire_desktop(self, user_id: str) -> Optional[str]:
        """
        Acquire user's desktop for execution.

        Args:
            user_id: User ID to find desktop for

        Returns:
            Desktop ID if found and online, None otherwise
        """
        desktop = await self.get_user_desktop(user_id)
        if desktop:
            self._desktop_id = desktop.get("desktop_id")
            return self._desktop_id
        return None

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        desktop_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool on the desktop.

        Args:
            tool_name: Tool name (e.g., "read_file", "bash_execute")
            arguments: Tool arguments
            desktop_id: Desktop ID (uses acquired desktop if not specified)
            timeout: Custom timeout for this call

        Returns:
            Tool execution result

        Raises:
            ConnectionError: If desktop is not connected
            TimeoutError: If execution times out
            RuntimeError: If execution fails
        """
        target_desktop = desktop_id or self._desktop_id
        if not target_desktop:
            raise ValueError("No desktop_id specified and none acquired")

        session = await self._get_session()

        # Map tool name to desktop agent action
        # The desktop agent uses slightly different action names
        action_map = {
            # Standard MCP tool names -> Desktop agent actions
            "read_file": "read_file",
            "write_file": "write_file",
            "edit_file": "edit_file",
            "bash_execute": "bash_execute",
            "bash": "bash_execute",  # Alias
            "glob_files": "glob_files",
            "glob": "glob_files",  # Alias
            "grep_search": "grep_search",
            "grep": "grep_search",  # Alias
            "ls_directory": "ls_directory",
            "ls": "ls_directory",  # Alias
            "system_info": "system_info",
            "get_capabilities": "get_capabilities",
            "ping": "ping",
        }

        action = action_map.get(tool_name, tool_name)

        payload = {
            "action": action,
            "params": arguments,
            "timeout": timeout or self.request_timeout,
        }

        try:
            async with session.post(
                f"{self.pool_manager_url}/desktop/execute",
                params={"desktop_id": target_desktop},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout or self.request_timeout)
            ) as resp:
                if resp.status == 503:
                    error_text = await resp.text()
                    raise ConnectionError(f"Desktop not connected: {error_text}")

                if resp.status == 504:
                    raise TimeoutError(f"Desktop execution timed out")

                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Execution failed ({resp.status}): {error_text}")

                result = await resp.json()

                # Extract data from response
                if result.get("success"):
                    return result.get("data", {})
                else:
                    raise RuntimeError(f"Execution failed: {result}")

        except asyncio.TimeoutError:
            raise TimeoutError(f"Desktop execution timed out after {timeout or self.request_timeout}s")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error: {e}")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Call a tool with MCP-compatible interface.

        This method provides compatibility with the MCPClient interface,
        making it easy to swap in for desktop execution.

        Args:
            tool_name: Tool name
            arguments: Tool arguments
            progress_callback: Optional callback for progress (not used for desktop)

        Returns:
            Result in MCP-compatible format: {"result": ..., "context": {}}
        """
        result = await self.execute_tool(tool_name, arguments)

        # Format response like MCP client
        return {
            "result": result,
            "context": {
                "execution_env": "desktop",
                "desktop_id": self._desktop_id,
            },
            "progress_messages": [],
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools on the desktop.

        Returns:
            List of tool definitions
        """
        # Desktop agent has fixed set of tools
        return [
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to file"},
                        "offset": {"type": "integer", "description": "Line number to start from"},
                        "limit": {"type": "integer", "description": "Number of lines to read"},
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": "Edit a file with string replacement",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to file"},
                        "old_string": {"type": "string", "description": "String to replace"},
                        "new_string": {"type": "string", "description": "Replacement string"},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
            {
                "name": "bash_execute",
                "description": "Execute a bash command",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds"},
                        "working_directory": {"type": "string", "description": "Working directory"},
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "glob_files",
                "description": "Find files matching a glob pattern",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern"},
                        "path": {"type": "string", "description": "Directory to search in"},
                        "max_results": {"type": "integer", "description": "Maximum results"},
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "grep_search",
                "description": "Search file contents with regex",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern"},
                        "path": {"type": "string", "description": "Directory to search"},
                        "file_pattern": {"type": "string", "description": "File glob pattern"},
                        "max_results": {"type": "integer", "description": "Maximum results"},
                        "case_insensitive": {"type": "boolean", "description": "Case insensitive"},
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "ls_directory",
                "description": "List directory contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"},
                    },
                    "required": [],
                },
            },
            {
                "name": "system_info",
                "description": "Get system information",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    async def health_check(self) -> bool:
        """Check if Pool Manager is available"""
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.pool_manager_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def close(self):
        """Clean shutdown"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("DesktopExecutionClient closed")


# Global singleton
_desktop_client: Optional[DesktopExecutionClient] = None
_desktop_lock = asyncio.Lock()


async def get_desktop_client(
    pool_manager_url: str = None,
    user_id: str = None,
    request_timeout: int = 120,
) -> DesktopExecutionClient:
    """
    Get thread-safe singleton desktop execution client.

    Args:
        pool_manager_url: Pool Manager URL (default from settings)
        user_id: User ID to acquire desktop for (optional)
        request_timeout: Request timeout in seconds

    Returns:
        DesktopExecutionClient instance
    """
    global _desktop_client

    if _desktop_client is None:
        async with _desktop_lock:
            if _desktop_client is None:
                if pool_manager_url is None:
                    from isa_agent_sdk.core.config import settings
                    pool_manager_url = getattr(settings, 'pool_manager_url', 'http://localhost:8090')

                _desktop_client = DesktopExecutionClient(
                    pool_manager_url=pool_manager_url,
                    request_timeout=request_timeout,
                )
                logger.info("Global DesktopExecutionClient initialized")

    # Acquire desktop for user if specified
    if user_id and not _desktop_client._desktop_id:
        await _desktop_client.acquire_desktop(user_id)

    return _desktop_client


async def close_desktop_client():
    """Close the global desktop client"""
    global _desktop_client
    if _desktop_client:
        await _desktop_client.close()
        _desktop_client = None
