#!/usr/bin/env python3
"""
isA Agent SDK - Error Classes
=============================

Unified error hierarchy for the isA Agent SDK.
Provides typed exceptions for better error handling and debugging.

Example:
    from isa_agent_sdk import ISASDKError, ToolExecutionError, SessionError

    try:
        async for msg in query("Do something"):
            ...
    except ToolExecutionError as e:
        print(f"Tool failed: {e.tool_name} - {e}")
    except SessionError as e:
        print(f"Session issue: {e.session_id} - {e}")
    except ISASDKError as e:
        print(f"SDK error: {e}")
"""

from typing import Optional, Dict, Any


# ============================================================================
# Base Error
# ============================================================================

class ISASDKError(Exception):
    """
    Base exception for all isA Agent SDK errors.

    All SDK-specific exceptions inherit from this class,
    allowing users to catch all SDK errors with a single except clause.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


# ============================================================================
# Connection & Infrastructure Errors
# ============================================================================

class ConnectionError(ISASDKError):
    """
    Failed to connect to a required service.

    Raised when:
    - MCP server is unreachable
    - Model API is unavailable
    - Database connection fails
    """

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.service = service
        self.url = url
        self.details["service"] = service
        self.details["url"] = url


class TimeoutError(ISASDKError):
    """
    Operation timed out.

    Raised when:
    - API request exceeds timeout
    - Tool execution takes too long
    - Model response times out
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.details["timeout_seconds"] = timeout_seconds
        self.details["operation"] = operation


class CircuitBreakerError(ISASDKError):
    """
    Circuit breaker is open, blocking requests.

    Raised when a service has failed too many times
    and the circuit breaker is preventing further requests.
    """

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.service = service
        self.retry_after_seconds = retry_after_seconds
        self.details["service"] = service
        self.details["retry_after_seconds"] = retry_after_seconds


class RateLimitError(ISASDKError):
    """
    Rate limit exceeded.

    Raised when API rate limits are hit.
    """

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[float] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.retry_after_seconds = retry_after_seconds
        self.limit = limit
        self.details["retry_after_seconds"] = retry_after_seconds
        self.details["limit"] = limit


# ============================================================================
# Execution Errors
# ============================================================================

class ExecutionError(ISASDKError):
    """
    Base class for execution-related errors.

    Parent class for errors that occur during agent execution.
    """

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.session_id = session_id
        self.details["session_id"] = session_id


class ToolExecutionError(ExecutionError):
    """
    Tool execution failed.

    Raised when:
    - Tool returns an error
    - Tool throws an exception
    - Tool input validation fails
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, session_id=session_id, **kwargs)
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.details["tool_name"] = tool_name
        self.details["tool_args"] = tool_args


class ModelError(ExecutionError):
    """
    Model/LLM error.

    Raised when:
    - Model returns an error response
    - Model output parsing fails
    - Model is unavailable
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, session_id=session_id, **kwargs)
        self.model = model
        self.error_code = error_code
        self.details["model"] = model
        self.details["error_code"] = error_code


class MaxIterationsError(ExecutionError):
    """
    Maximum iterations/turns exceeded.

    Raised when the agent exceeds the configured
    maximum number of reasoning iterations.
    """

    def __init__(
        self,
        message: str,
        max_iterations: Optional[int] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, session_id=session_id, **kwargs)
        self.max_iterations = max_iterations
        self.details["max_iterations"] = max_iterations


class GraphExecutionError(ExecutionError):
    """
    LangGraph execution error.

    Raised when the underlying graph execution fails.
    """

    def __init__(
        self,
        message: str,
        node: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, session_id=session_id, **kwargs)
        self.node = node
        self.details["node"] = node


# ============================================================================
# Session & State Errors
# ============================================================================

class SessionError(ISASDKError):
    """
    Session-related error.

    Base class for errors related to session management.
    """

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.session_id = session_id
        self.details["session_id"] = session_id


class SessionNotFoundError(SessionError):
    """
    Session not found.

    Raised when trying to resume or access a non-existent session.
    """
    pass


class SessionExpiredError(SessionError):
    """
    Session has expired.

    Raised when a session's TTL has passed.
    """

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        expired_at: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, session_id=session_id, **kwargs)
        self.expired_at = expired_at
        self.details["expired_at"] = expired_at


class CheckpointError(SessionError):
    """
    Checkpoint operation failed.

    Raised when:
    - Checkpoint save fails
    - Checkpoint load fails
    - Checkpoint is corrupted
    """

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, session_id=session_id, **kwargs)
        self.checkpoint_id = checkpoint_id
        self.details["checkpoint_id"] = checkpoint_id


class ResumeError(SessionError):
    """
    Failed to resume session.

    Raised when session resume fails due to state issues.
    """
    pass


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(ISASDKError):
    """
    Input validation failed.

    Base class for validation-related errors.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.field = field
        self.value = value
        self.details["field"] = field
        # Don't include value in details (might be sensitive)


class SchemaError(ValidationError):
    """
    Schema validation failed.

    Raised when:
    - JSON schema validation fails
    - Pydantic model validation fails
    - Structured output doesn't match schema
    """

    def __init__(
        self,
        message: str,
        schema: Optional[Dict[str, Any]] = None,
        errors: Optional[list] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.schema = schema
        self.errors = errors or []
        self.details["errors"] = self.errors


class ConfigurationError(ValidationError):
    """
    Configuration is invalid.

    Raised when ISAAgentOptions or other config is misconfigured.
    """
    pass


# ============================================================================
# Permission & Authorization Errors
# ============================================================================

class PermissionError(ISASDKError):
    """
    Permission denied.

    Base class for permission-related errors.
    """

    def __init__(
        self,
        message: str,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.action = action
        self.resource = resource
        self.details["action"] = action
        self.details["resource"] = resource


class ToolPermissionError(PermissionError):
    """
    Tool execution not permitted.

    Raised when a tool is not in allowed_tools or user denied permission.
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, action="execute_tool", resource=tool_name, **kwargs)
        self.tool_name = tool_name


class HILDeniedError(PermissionError):
    """
    Human-in-the-loop request was denied.

    Raised when user denies a permission or authorization request.
    """

    def __init__(
        self,
        message: str,
        request_type: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.request_type = request_type
        self.session_id = session_id
        self.details["request_type"] = request_type
        self.details["session_id"] = session_id


class HILTimeoutError(PermissionError):
    """
    Human-in-the-loop request timed out.

    Raised when user doesn't respond to HIL request within timeout.
    """

    def __init__(
        self,
        message: str,
        request_type: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.request_type = request_type
        self.timeout_seconds = timeout_seconds
        self.details["request_type"] = request_type
        self.details["timeout_seconds"] = timeout_seconds


# ============================================================================
# MCP Errors
# ============================================================================

class MCPError(ISASDKError):
    """
    MCP (Model Context Protocol) error.

    Base class for MCP-related errors.
    """

    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, kwargs)
        self.server_name = server_name
        self.details["server_name"] = server_name


class MCPConnectionError(MCPError, ConnectionError):
    """
    Failed to connect to MCP server.
    """
    pass


class MCPToolNotFoundError(MCPError):
    """
    Tool not found on MCP server.
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        server_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, server_name=server_name, **kwargs)
        self.tool_name = tool_name
        self.details["tool_name"] = tool_name


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base
    "ISASDKError",

    # Connection & Infrastructure
    "ConnectionError",
    "TimeoutError",
    "CircuitBreakerError",
    "RateLimitError",

    # Execution
    "ExecutionError",
    "ToolExecutionError",
    "ModelError",
    "MaxIterationsError",
    "GraphExecutionError",

    # Session & State
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "CheckpointError",
    "ResumeError",

    # Validation
    "ValidationError",
    "SchemaError",
    "ConfigurationError",

    # Permission & Authorization
    "PermissionError",
    "ToolPermissionError",
    "HILDeniedError",
    "HILTimeoutError",

    # MCP
    "MCPError",
    "MCPConnectionError",
    "MCPToolNotFoundError",
]
