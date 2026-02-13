"""
MCP Client - Wrapper around isa_mcp.AsyncMCPClient

This module provides a thin wrapper around the isa_mcp client for use in the Agent SDK.
All actual MCP communication is delegated to isa_mcp.AsyncMCPClient.

Features:
- Wraps isa_mcp.AsyncMCPClient for MCP JSON-RPC communication
- Adds SDK-specific features: session tracking, metrics, context extraction
- Provides MCPService compatibility layer for existing code
- HIL (Human-in-the-Loop) response parsing

Usage:
    from isa_agent_sdk.clients.mcp_client import MCPClient

    client = MCPClient(mcp_url="http://localhost:8081")
    await client.initialize()

    # Call tools
    result = await client.call_tool("get_weather", {"city": "Tokyo"})

    # Get prompts
    prompt = await client.get_prompt("default_reason_prompt", {})

    await client.close()
"""

import json
import logging
from time import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# Import isa_mcp client - the single source of truth for MCP communication
try:
    from isa_mcp.mcp_client import AsyncMCPClient as IsAMCPClient
    ISA_MCP_AVAILABLE = True
except ImportError:
    ISA_MCP_AVAILABLE = False
    IsAMCPClient = None


class MCPClient:
    """
    SDK wrapper around isa_mcp.AsyncMCPClient

    This is a thin wrapper that delegates all MCP communication to isa_mcp.AsyncMCPClient.
    It adds SDK-specific features like session tracking and metrics.

    Features:
    - Delegates to isa_mcp.AsyncMCPClient for all MCP operations
    - Session tracking with X-Session-ID, X-Client-ID headers
    - Performance metrics
    - HIL response parsing (via isa_mcp)
    """

    def __init__(
        self,
        mcp_url: str = "http://localhost:8081",
        session_id: Optional[str] = None,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        session_service=None
    ):
        """
        Initialize MCP client wrapper

        Args:
            mcp_url: MCP server URL (e.g., 'http://localhost:8081' or 'http://localhost:8081/mcp')
            session_id: Optional session ID for tracking
            client_id: Optional client identifier (e.g., 'agent_v1')
            user_id: Optional user ID for operations
            logger: Optional logger instance
            session_service: Optional SessionService instance for dynamic session tracking
        """
        # Normalize URL: extract base URL (without /mcp suffix)
        self.mcp_url = mcp_url.rstrip('/')
        if self.mcp_url.endswith('/mcp'):
            self.mcp_url = self.mcp_url[:-4]
        self.mcp_url = self.mcp_url.replace('localhost', '127.0.0.1')

        self.session_id = session_id
        self.client_id = client_id or "isA_agent_v1"
        self.user_id = user_id
        self.logger = logger or logging.getLogger(__name__)

        # Session service for dynamic session tracking
        self._session_service = session_service

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'total_errors': 0,
            'total_duration': 0.0,
            'context_extractions': 0,
            'progress_notifications': 0
        }

        # Initialize isa_mcp client (the actual MCP client)
        self._client: Optional[IsAMCPClient] = None
        if ISA_MCP_AVAILABLE:
            # Use localhost for isa_mcp client (it handles its own URL normalization)
            base_url = self.mcp_url.replace('127.0.0.1', 'localhost')
            self._client = IsAMCPClient(base_url=base_url)
            self.logger.info(f"MCPClient initialized with isa_mcp.AsyncMCPClient: {base_url}")
        else:
            self.logger.error("isa_mcp not available - MCP operations will fail")

    @property
    def session(self) -> bool:
        """Check if session is initialized (for backward compatibility)"""
        if self._client:
            return self._client._session is not None
        return False

    async def initialize(self):
        """Initialize the underlying isa_mcp client"""
        if self._client:
            await self._client.connect()
            self.logger.info("MCPClient initialized (isa_mcp connected)")

    async def close(self):
        """Close the underlying isa_mcp client"""
        if self._client:
            await self._client.close()
            self.logger.info("MCPClient closed", extra={'metrics': self.metrics})

    # =========================================================================
    # Core MCP Operations (delegated to isa_mcp)
    # =========================================================================

    async def call_tool(
        self,
        name: str,
        arguments: Dict,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """
        Call MCP tool with progress support

        Args:
            name: Tool name
            arguments: Tool arguments
            progress_callback: Optional progress callback

        Returns:
            Tool result dict (raw MCP response format)
        """
        if not self._client:
            return {"error": "MCP client not available"}

        start_time = time()
        self.metrics['total_requests'] += 1

        try:
            # Delegate to isa_mcp client
            result = await self._client.call_tool(name, arguments)

            duration = time() - start_time
            self.metrics['total_duration'] += duration

            self.logger.info(f"MCP call_tool success: {name}", extra={
                'tool_name': name,
                'duration_ms': int(duration * 1000)
            })

            # Wrap result in expected format for compatibility
            return {
                'result': result if isinstance(result, dict) else {"data": result},
                'context': self._extract_context(result) if isinstance(result, dict) else None,
                'progress_messages': [],
                'duration_ms': int(duration * 1000)
            }

        except Exception as e:
            self.metrics['total_errors'] += 1
            self.logger.error(f"MCP call_tool failed: {name}", extra={'error': str(e)})
            return {"error": str(e)}

    async def call_tool_and_parse(
        self,
        name: str,
        arguments: Dict,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Call MCP tool and parse response (convenience method)

        This method calls the tool and parses the response automatically.
        It handles HIL responses, structured content, and errors.

        Args:
            name: Tool name
            arguments: Tool arguments
            progress_callback: Optional progress callback

        Returns:
            Parsed response dict with standardized format
        """
        if not self._client:
            return {"status": "error", "error": "MCP client not available"}

        try:
            # Delegate to isa_mcp client - it returns parsed result
            result = await self._client.call_tool(name, arguments)
            return result if isinstance(result, dict) else {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get prompt from MCP server

        Args:
            name: Prompt name (e.g., 'default_reason_prompt')
            arguments: Prompt arguments

        Returns:
            Prompt text or None if not found
        """
        if not self._client:
            return None

        try:
            result = await self._client.get_prompt(name, arguments or {})

            # Extract prompt text from result
            if 'messages' in result:
                messages = result['messages']
                if messages and len(messages) > 0:
                    content = messages[0].get('content', {})
                    if isinstance(content, dict) and 'text' in content:
                        return content['text']
                    elif isinstance(content, str):
                        return content

            self.logger.warning(f"No prompt content for: {name}")
            return None

        except Exception as e:
            self.logger.error(f"get_prompt failed: {name}", extra={'error': str(e)})
            return None

    async def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Read MCP resource

        Args:
            uri: Resource URI (e.g., 'event://status')

        Returns:
            Resource content with context or None
        """
        if not self._client:
            return None

        try:
            result = await self._client.read_resource(uri)
            return result
        except Exception as e:
            self.logger.error(f"get_resource failed: {uri}", extra={'error': str(e)})
            return None

    async def read_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Read MCP resource by URI (alias for get_resource)

        This method matches the isa_mcp.AsyncMCPClient API for compatibility.

        Args:
            uri: Resource URI (e.g., 'vibe://skill/tdd')

        Returns:
            Resource content or None
        """
        return await self.get_resource(uri)

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: int = 10,
        score_threshold: float = 0.3
    ) -> Dict:
        """
        Search MCP capabilities using hierarchical search

        Args:
            query: Search query
            filters: Optional filters (e.g., {'types': ['tool']})
            max_results: Maximum results to return
            score_threshold: Minimum similarity score

        Returns:
            Search results
        """
        if not self._client:
            return {'status': 'error', 'results': []}

        try:
            # Determine item type from filters
            item_type = None
            if filters and 'types' in filters:
                types_list = filters['types']
                if isinstance(types_list, list) and len(types_list) == 1:
                    item_type = types_list[0]

            # Delegate to isa_mcp discover
            result = await self._client.discover(
                query=query,
                item_type=item_type,
                limit=max_results
            )

            # Convert SearchResult to dict format
            return {
                'status': 'success',
                'results': [
                    {
                        'name': m.name,
                        'type': m.type,
                        'description': m.description,
                        'score': m.score,
                        'skill': m.skill,
                        'inputSchema': m.input_schema
                    }
                    for m in result.matches
                ],
                'count': result.total_found
            }

        except Exception as e:
            self.logger.error(f"search failed: {query}", extra={'error': str(e)})
            return {'status': 'error', 'results': []}

    async def search_tools(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for tools

        Args:
            query: Search query
            user_id: Optional user ID (kept for API compatibility)
            max_results: Maximum results

        Returns:
            List of matching tools
        """
        result = await self.search(query, filters={"types": ["tool"]}, max_results=max_results)
        return result.get('results', [])

    async def search_prompts(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for prompts"""
        result = await self.search(query, filters={"types": ["prompt"]}, max_results=max_results)
        return result.get('results', [])

    async def search_resources(
        self,
        user_id: str,
        query: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for resources"""
        if query:
            result = await self.search(query, filters={"types": ["resource"]}, max_results=max_results)
            return result.get('results', [])
        # If no query, list all resources
        return await self.get_default_resources(max_results)

    # =========================================================================
    # Default/List Operations
    # =========================================================================

    async def get_default_tools(self, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get default tools (meta-tools)

        Returns:
            List of default tool definitions
        """
        if not self._client:
            return []

        try:
            tools = await self._client.get_default_tools()
            self.logger.info(f"get_default_tools | count: {len(tools)}")
            return tools[:max_results]
        except Exception as e:
            self.logger.error(f"get_default_tools failed: {e}")
            return []

    async def get_default_prompts(self, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get all default prompts"""
        if not self._client:
            return []

        try:
            prompts = await self._client.list_prompts()
            return prompts[:max_results]
        except Exception as e:
            self.logger.error(f"get_default_prompts failed: {e}")
            return []

    async def get_default_resources(self, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get all default resources"""
        if not self._client:
            return []

        try:
            resources = await self._client.list_resources()
            return resources[:max_results]
        except Exception as e:
            self.logger.error(f"get_default_resources failed: {e}")
            return []

    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the input/output schema for a specific tool

        Args:
            tool_name: Exact name of the tool

        Returns:
            Dict with name, description, inputSchema or None
        """
        if not self._client:
            return None

        try:
            schema = await self._client.get_tool_schema(tool_name)
            if schema and not schema.get('error'):
                return {
                    "name": tool_name,
                    "description": schema.get("description", ""),
                    "inputSchema": schema.get("input_schema", schema.get("inputSchema", {}))
                }
            return None
        except Exception as e:
            self.logger.error(f"get_tool_schema failed: {tool_name}", extra={'error': str(e)})
            return None

    # =========================================================================
    # Security Level Operations (integrated with isa_mcp SecurityManager)
    # =========================================================================

    async def get_tool_security_levels(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get security levels for all tools from isa_MCP SecurityPolicy.

        Returns:
            Dict with tool names mapped to their security levels and metadata
        """
        try:
            # Try to import isa_MCP security module
            from isa_mcp.core.security import security_policy, SecurityLevel

            tools = await self.get_default_tools()
            levels = {}

            for tool in tools:
                tool_name = tool.get("name", "")
                if tool_name:
                    # Get security level from policy, default to LOW
                    level = security_policy.tool_policies.get(tool_name, SecurityLevel.LOW)
                    levels[tool_name] = {
                        "security_level": level.name,
                        "security_level_value": level.value
                    }

            return {
                "tools": levels,
                "metadata": {
                    "source": "isa_mcp",
                    "policy_version": "1.0"
                }
            }
        except ImportError:
            self.logger.debug("isa_mcp security module not available, using defaults")
            return {"tools": {}, "metadata": {"error": "security module unavailable"}}
        except Exception as e:
            self.logger.warning(f"Failed to get tool security levels: {e}")
            return {"tools": {}, "metadata": {"error": str(e)}}

    async def search_tools_by_security_level(
        self,
        security_level: str,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for tools matching a specific security level.

        Args:
            security_level: Security level to filter by (LOW, MEDIUM, HIGH, CRITICAL)
            query: Optional additional search query
            user_id: Optional user ID
            max_results: Maximum results to return

        Returns:
            List of tools matching the security level
        """
        try:
            from isa_mcp.core.security import security_policy, SecurityLevel

            # Parse requested security level
            try:
                requested_level = SecurityLevel[security_level.upper()]
            except KeyError:
                self.logger.warning(f"Invalid security level: {security_level}")
                return []

            tools = await self.get_default_tools()
            matching_tools = []

            for tool in tools:
                tool_name = tool.get("name", "")
                if tool_name:
                    level = security_policy.tool_policies.get(tool_name, SecurityLevel.LOW)
                    if level == requested_level:
                        # If query provided, filter by name/description match
                        if query:
                            search_text = f"{tool_name} {tool.get('description', '')}".lower()
                            if query.lower() not in search_text:
                                continue
                        matching_tools.append({
                            **tool,
                            "security_level": level.name
                        })

            return matching_tools[:max_results]
        except ImportError:
            self.logger.debug("isa_mcp security module not available")
            return []
        except Exception as e:
            self.logger.warning(f"Failed to search tools by security level: {e}")
            return []

    async def get_tool_security_level(
        self,
        tool_name: str,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get security level for a specific tool.

        Args:
            tool_name: Name of the tool
            user_id: Optional user ID

        Returns:
            Security level string (LOW, MEDIUM, HIGH, CRITICAL) or None
        """
        try:
            from isa_mcp.core.security import security_policy, SecurityLevel

            level = security_policy.tool_policies.get(tool_name, SecurityLevel.LOW)
            return level.name
        except ImportError:
            self.logger.debug("isa_mcp security module not available")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get tool security level: {e}")
            return None

    async def check_tool_security_authorized(
        self,
        tool_name: str,
        required_level: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Check if a tool meets or exceeds the required security level.

        Args:
            tool_name: Name of the tool
            required_level: Required security level (LOW, MEDIUM, HIGH, CRITICAL)
            user_id: Optional user ID

        Returns:
            True if tool meets or exceeds required security level
        """
        try:
            from isa_mcp.core.security import security_policy, SecurityLevel

            # Parse required level
            try:
                required = SecurityLevel[required_level.upper()]
            except KeyError:
                self.logger.warning(f"Invalid security level: {required_level}")
                return True  # Fail-open on invalid level

            # Get tool's security level
            tool_level = security_policy.tool_policies.get(tool_name, SecurityLevel.LOW)

            # Tool is authorized if its level value is >= required level value
            # (lower value = lower security requirement)
            return tool_level.value >= required.value
        except ImportError:
            self.logger.debug("isa_mcp security module not available, allowing access")
            return True  # Fail-open if security module unavailable
        except Exception as e:
            self.logger.warning(f"Security authorization check failed: {e}")
            return True  # Fail-open on error

    # =========================================================================
    # Universal Search
    # =========================================================================

    async def search_all(
        self,
        query: str,
        user_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search across all MCP capabilities

        Args:
            query: Search query
            user_id: Optional user ID
            filters: Optional filters
            max_results: Maximum results

        Returns:
            Search results with all capability types
        """
        return await self.search(query, filters, max_results)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_context(self, result: Dict) -> Optional[Dict]:
        """Extract context from MCP result"""
        if not isinstance(result, dict):
            return None

        # Check direct context field
        if 'context' in result:
            self.metrics['context_extractions'] += 1
            return result['context']

        # Check structured content
        if 'structuredContent' in result:
            content = result['structuredContent']
            if isinstance(content, dict) and 'result' in content:
                inner = content['result']
                if isinstance(inner, dict) and 'context' in inner:
                    self.metrics['context_extractions'] += 1
                    return inner['context']

        return None

    def parse_tool_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse MCP tool response (for backward compatibility)

        The isa_mcp client already parses responses, but this method
        is kept for code that expects the old format.
        """
        # If response already has 'result', extract it
        if 'result' in response:
            result = response['result']

            # Check for error
            if isinstance(result, dict) and result.get('isError'):
                if 'content' in result and result['content']:
                    return {
                        'status': 'error',
                        'error': result['content'][0].get('text', 'Unknown error')
                    }

            # Check for structured content
            if isinstance(result, dict) and 'structuredContent' in result:
                structured = result['structuredContent']
                if 'result' in structured:
                    return structured['result']
                return structured

            # Normal result
            if isinstance(result, dict) and 'content' in result and result['content']:
                content = result['content'][0]
                if content.get('type') == 'text':
                    try:
                        return json.loads(content['text'])
                    except json.JSONDecodeError:
                        return {'text': content['text'], 'status': 'success'}

            return result

        # If response has error
        if 'error' in response:
            return {
                'status': 'error',
                'error': response['error'].get('message', str(response['error']))
            }

        return response

    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        avg_duration = (
            self.metrics['total_duration'] / self.metrics['total_requests']
            if self.metrics['total_requests'] > 0
            else 0
        )

        return {
            'total_requests': self.metrics['total_requests'],
            'total_errors': self.metrics['total_errors'],
            'average_duration_ms': int(avg_duration * 1000),
            'success_rate': (
                (self.metrics['total_requests'] - self.metrics['total_errors'])
                / self.metrics['total_requests'] * 100
                if self.metrics['total_requests'] > 0
                else 0
            )
        }
