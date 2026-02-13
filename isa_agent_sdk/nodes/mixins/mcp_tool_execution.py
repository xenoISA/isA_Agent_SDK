"""
MCP Tool Execution Mixin - Methods for executing MCP tools
"""
from typing import Dict, Any, Optional, Callable
from langchain_core.runnables import RunnableConfig

from isa_agent_sdk.services.trace.mcp_callback import trace_mcp_operation


class MCPToolExecutionMixin:
    """Mixin providing MCP tool execution methods for BaseNode"""

    @trace_mcp_operation("BaseNode")
    async def mcp_call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        config: RunnableConfig,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Execute tool using SDK server, MCP service, or Desktop Agent based on config.

        Execution priority:
        1. SDK tool (project-based @tool decorator) - in-process
        2. Desktop Agent (env=desktop) - via Pool Manager
        3. MCP service (default) - via isA_MCP

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            config: LangGraph Runtime object
            progress_callback: Optional callback for MCP progress messages (SSE stream)

        Returns:
            Tool execution result (string)
        """
        env = config.get("configurable", {}).get("env", "cloud_shared")
        user_id = config.get("configurable", {}).get("user_id", "unknown")
        self.logger.info(f"[TOOL_ROUTE] mcp_call_tool | tool={tool_name} | env={env} | user_id={user_id}")

        # 1. Check if this is an SDK tool (project-based @tool decorator)
        sdk_servers = config.get("configurable", {}).get("sdk_mcp_servers", {})
        sdk_result = await self._try_sdk_tool(tool_name, arguments, sdk_servers)
        if sdk_result is not None:
            self.logger.info(f"[SDK_TOOL] Executed in-process: {tool_name}")
            return sdk_result

        # 2. Check if desktop execution is requested
        if env == "desktop":
            return await self._call_tool_via_desktop(tool_name, arguments, config)

        # Standard MCP execution
        mcp_service = await self.get_initialized_mcp_service(config)
        if not mcp_service:
            return "Error: MCP service not available"

        # Inject user_id into arguments if not already present
        # This ensures MCP tools receive the correct user context
        if "user_id" not in arguments and user_id and user_id != "unknown":
            arguments = {**arguments, "user_id": user_id}
            self.logger.debug(f"Injected user_id={user_id} into tool arguments for {tool_name}")

        try:
            # Call MCP tool with progress callback support
            result = await mcp_service.call_tool(tool_name, arguments, progress_callback=progress_callback)

            # Extract result string from MCP response
            # MCPClient.call_tool returns Dict with 'result', 'context', 'progress_messages'
            if isinstance(result, dict):
                # Parse the result from MCP response format
                if 'result' in result:
                    mcp_result = result['result']
                    if 'content' in mcp_result and len(mcp_result['content']) > 0:
                        # Extract text from content array
                        content_item = mcp_result['content'][0]
                        if isinstance(content_item, dict) and 'text' in content_item:
                            return content_item['text']
                        elif isinstance(content_item, str):
                            return content_item
                    # Fallback: return string representation
                    return str(mcp_result)
                # If no 'result', return string representation of entire response
                return str(result)
            # If already a string, return as-is
            return str(result)
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            self.logger.error(error_msg)
            return error_msg

    async def _call_tool_via_desktop(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        config: RunnableConfig
    ) -> str:
        """
        Execute tool via Desktop Agent through Pool Manager

        Routes tool calls to user's desktop for local execution.
        Flow: SDK -> Pool Manager -> Desktop Agent -> Local Filesystem
        """
        from isa_agent_sdk.clients.desktop_execution_client import get_desktop_client

        user_id = config.get("configurable", {}).get("user_id", "unknown")

        try:
            # Get desktop client and acquire user's desktop
            desktop_client = await get_desktop_client(user_id=user_id)

            if not desktop_client._desktop_id:
                await desktop_client.acquire_desktop(user_id)

            if not desktop_client._desktop_id:
                return f"Error: No online desktop found for user {user_id}. Please ensure your desktop agent is running."

            self.logger.info(f"[Desktop] Routing {tool_name} to desktop {desktop_client._desktop_id}")

            # Inject user_id into arguments if not already present
            if "user_id" not in arguments and user_id and user_id != "unknown":
                arguments = {**arguments, "user_id": user_id}
                self.logger.debug(f"[Desktop] Injected user_id={user_id} into tool arguments for {tool_name}")

            # Execute tool on desktop
            result = await desktop_client.execute_tool(tool_name, arguments)

            # Format result based on tool type
            if result.get("status") == "success":
                data = result.get("data", {})

                # Handle different tool response formats
                if "content" in data:
                    return data["content"]
                elif "stdout" in data:
                    output = data.get("stdout", "")
                    stderr = data.get("stderr", "")
                    exit_code = data.get("exit_code", 0)
                    result_str = f"Exit code: {exit_code}\n"
                    if output:
                        result_str += f"Output:\n{output}"
                    if stderr:
                        result_str += f"\nStderr:\n{stderr}"
                    return result_str
                elif "entries" in data:
                    # Directory listing
                    entries = data.get("entries", [])
                    return "\n".join([f"{'[DIR]' if e.get('type') == 'directory' else '[FILE]'} {e.get('name')}" for e in entries])
                elif "bytes_written" in data:
                    return f"File written successfully ({data['bytes_written']} bytes)"
                else:
                    return str(data)
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            self.logger.error(f"[Desktop] Tool execution failed: {e}")
            return f"Desktop execution failed: {e}"

    async def _try_sdk_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        sdk_servers: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to execute a tool via SDK MCP server (project-based @tool decorator).

        Args:
            tool_name: Full tool name (e.g., "mcp__project__greet")
            arguments: Tool arguments
            sdk_servers: Dict of SDK MCP servers from config

        Returns:
            Tool result string if this is an SDK tool, None otherwise
        """
        # Import here to avoid circular imports
        from isa_agent_sdk._tools import execute_sdk_tool, is_sdk_tool

        # Check if this is an SDK tool
        if not is_sdk_tool(tool_name, sdk_servers):
            return None

        try:
            result = await execute_sdk_tool(tool_name, arguments, sdk_servers)
            if result is None:
                return None

            # Extract text from MCP result format
            if isinstance(result, dict):
                if result.get("isError"):
                    content = result.get("content", [])
                    if content and isinstance(content[0], dict):
                        return f"Error: {content[0].get('text', 'Unknown error')}"
                    return "Error: SDK tool execution failed"

                content = result.get("content", [])
                if content and isinstance(content[0], dict):
                    return content[0].get("text", str(result))
                return str(result)

            return str(result)

        except Exception as e:
            self.logger.error(f"[SDK_TOOL] Execution failed: {e}")
            return f"SDK tool execution failed: {e}"
