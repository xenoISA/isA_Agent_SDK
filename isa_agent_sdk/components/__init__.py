"""
Modular components for isA_Agent
Includes production-grade MCP service, model service and session management
"""

from .session_service import SessionService
# MCPService removed - use MCPClient from isa_agent_sdk.clients.mcp_client instead
# ModelService removed - use ModelClient from isa_agent_sdk.clients.model_client instead
from .billing_service import billing_service, create_billing_handler, track_model_call, track_tool_call
# user_service replaced with user_client from isa_agent_sdk.clients.user_client for direct microservices communication
# from .user_service import user_service  # Deprecated - use user_client instead

__all__ = [
    "SessionService",
    # "MCPService",  # Removed - use MCPClient from isa_agent_sdk.clients.mcp_client instead
    # "ModelService",  # Removed - use ModelClient from isa_agent_sdk.clients.model_client instead
    # "get_model_service",  # Removed - use get_model_client from isa_agent_sdk.clients.model_client instead
    "billing_service",
    "create_billing_handler",
    "track_model_call",
    "track_tool_call",
    # "user_service"  # Deprecated - use user_client from isa_agent_sdk.clients.user_client instead
]