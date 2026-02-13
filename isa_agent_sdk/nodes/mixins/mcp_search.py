"""
MCP Search Mixin - Methods for searching MCP capabilities
"""
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnableConfig


class MCPSearchMixin:
    """Mixin providing MCP search methods for BaseNode"""

    async def mcp_search_all(
        self,
        query: str,
        config: RunnableConfig,
        user_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search across all MCP capabilities (tools, prompts, resources)

        Args:
            query: Search query
            config: LangGraph RunnableConfig
            user_id: Optional user ID for access control (defaults to config user_id)
            filters: Optional search filters
            max_results: Maximum number of results

        Returns:
            Search results with all capability types
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return {"results": [], "result_count": 0}

        user_id = user_id or self.get_user_id(config)

        try:
            return await mcp_service.search_all(query, user_id, filters, max_results)
        except Exception as e:
            self.logger.error(f"Universal search failed: {e}")
            return {"results": [], "result_count": 0}

    async def mcp_search_tools(
        self,
        query: str,
        config: RunnableConfig,
        user_id: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for tools using MCP service

        Args:
            query: Search query for tools
            config: LangGraph RunnableConfig
            user_id: Optional user ID for access control (defaults to config user_id)
            max_results: Maximum number of results

        Returns:
            List of matching tools
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        user_id = user_id or self.get_user_id(config)

        try:
            return await mcp_service.search_tools(query, user_id, max_results)
        except Exception as e:
            self.logger.error(f"Tool search failed: {e}")
            return []

    async def mcp_search_prompts(
        self,
        query: str,
        config: RunnableConfig,
        user_id: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for prompts using MCP service

        Args:
            query: Search query for prompts
            config: LangGraph RunnableConfig
            user_id: Optional user ID for access control (defaults to config user_id)
            max_results: Maximum number of results

        Returns:
            List of matching prompts
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        user_id = user_id or self.get_user_id(config)

        try:
            return await mcp_service.search_prompts(query, user_id, max_results)
        except Exception as e:
            self.logger.error(f"Prompt search failed: {e}")
            return []

    async def mcp_search_resources(
        self,
        user_id: str,
        config: RunnableConfig,
        query: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for user resources using MCP service

        Args:
            user_id: User ID (required for resource access)
            config: LangGraph RunnableConfig
            query: Optional search query (if None, returns all user resources)
            max_results: Maximum number of results

        Returns:
            List of matching resources
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return []

        try:
            return await mcp_service.search_resources(user_id, query, max_results)
        except Exception as e:
            self.logger.error(f"Resource search failed: {e}")
            return []
