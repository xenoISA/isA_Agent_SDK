"""
MCP Assets Mixin - Methods for getting prompts and resources from MCP
"""
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.services.trace.mcp_callback import trace_mcp_operation


class MCPAssetsMixin:
    """Mixin providing MCP asset retrieval methods for BaseNode"""

    @trace_mcp_operation("BaseNode")
    async def mcp_get_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, Any],
        config: RunnableConfig
    ) -> Optional[str]:
        """
        Get assembled prompt from MCP service

        Args:
            prompt_name: Name of prompt template
            arguments: Prompt arguments for template substitution
            config: LangGraph Runtime object

        Returns:
            Assembled prompt text or None if failed
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return None

        try:
            return await mcp_service.get_prompt(prompt_name, arguments)
        except Exception as e:
            self.logger.error(f"Prompt retrieval failed: {e}")
            return None

    @trace_mcp_operation("BaseNode")
    async def mcp_get_resource(
        self,
        uri: str,
        config: RunnableConfig
    ) -> Optional[Dict[str, Any]]:
        """
        Get resource content from MCP service

        Args:
            uri: Resource URI to read
            config: LangGraph Runtime object

        Returns:
            Resource data with contents and metadata, or None if failed
        """
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return None

        try:
            return await mcp_service.get_resource(uri)
        except Exception as e:
            self.logger.error(f"Resource retrieval failed: {e}")
            return None
