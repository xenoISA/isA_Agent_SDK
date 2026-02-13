"""
MCP Security Mixin - Methods for MCP security level operations
"""
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnableConfig


class MCPSecurityMixin:
    """Mixin providing MCP security methods for BaseNode"""

    async def mcp_get_tool_security_levels(
        self,
        config: RunnableConfig,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get security levels for all available tools

        Args:
            config: LangGraph RunnableConfig
            user_id: Optional user ID for access control (defaults to config user_id)

        Returns:
            Dict with tools and their security levels
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return {"tools": {}, "metadata": {}}

        user_id = user_id or self.get_user_id(config)

        try:
            return await mcp_service.get_tool_security_levels(user_id)
        except Exception as e:
            self.logger.error(f"Failed to get tool security levels: {e}")
            return {"tools": {}, "metadata": {}}

    async def mcp_search_tools_by_security_level(
        self,
        security_level: str,
        config: RunnableConfig,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for tools by security level

        Args:
            security_level: Security level (LOW, MEDIUM, HIGH, CRITICAL)
            config: LangGraph RunnableConfig
            query: Optional search query to filter tools
            user_id: Optional user ID for access control (defaults to config user_id)
            max_results: Maximum number of results

        Returns:
            List of tools matching the security level and query
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return []

        user_id = user_id or self.get_user_id(config)

        try:
            return await mcp_service.search_tools_by_security_level(security_level, query, user_id, max_results)
        except Exception as e:
            self.logger.error(f"Security level search failed: {e}")
            return []

    async def mcp_get_tool_security_level(
        self,
        tool_name: str,
        config: RunnableConfig,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get security level for a specific tool

        Args:
            tool_name: Name of the tool
            config: LangGraph RunnableConfig
            user_id: Optional user ID for access control (defaults to config user_id)

        Returns:
            Security level string (LOW, MEDIUM, HIGH, CRITICAL) or None if not found
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return None

        user_id = user_id or self.get_user_id(config)

        try:
            return await mcp_service.get_tool_security_level(tool_name, user_id)
        except Exception as e:
            self.logger.error(f"Failed to get tool security level: {e}")
            return None

    async def mcp_check_tool_security_authorized(
        self,
        tool_name: str,
        required_level: str,
        config: RunnableConfig,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Check if a tool meets the required security level

        Args:
            tool_name: Name of the tool
            required_level: Required security level (LOW, MEDIUM, HIGH, CRITICAL)
            config: LangGraph RunnableConfig
            user_id: Optional user ID for access control (defaults to config user_id)

        Returns:
            True if tool meets or exceeds required security level
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return False

        user_id = user_id or self.get_user_id(config)

        try:
            return await mcp_service.check_tool_security_authorized(tool_name, required_level, user_id)
        except Exception as e:
            self.logger.error(f"Security authorization check failed: {e}")
            return False
