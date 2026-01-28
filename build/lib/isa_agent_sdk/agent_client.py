#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
isA_Agent Client SDK - OpenAI-Compatible Agent API
====================================================

Ultra-simple agent client that matches OpenAI SDK patterns.
Handles complex event streaming, session management, and multi-agent workflows.

This is the main client for agent interactions. Hardware/IoT clients are separate.

Usage:
    from isa_agent_sdk.agent_client import ISAAgent

    # Initialize
    client = ISAAgent()  # or ISAAgent(api_key="key", base_url="url")

    # Simple chat
    response = client.chat.create(
        message="Hello!",
        user_id="user123"
    )
    print(response.content)

    # Streaming with events
    for event in client.chat.stream(
        message="Explain quantum computing",
        user_id="user123"
    ):
        if event.is_content:
            print(event.content, end="", flush=True)
        elif event.is_thinking:
            print(f"[Thinking: {event.content}]")
        elif event.is_tool_call:
            print(f"[Tool: {event.tool_name}]")
"""

import os
import json
import httpx
import asyncio
from typing import Optional, List, Dict, Any, Iterator, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ============================================================================
# Event Types (matching your event_types.py)
# ============================================================================

class EventType(str, Enum):
    """Agent event types"""
    # Session
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    SESSION_PAUSED = "session.paused"
    SESSION_RESUMED = "session.resumed"

    # Content
    CONTENT_THINKING = "content.thinking"
    CONTENT_TOKEN = "content.token"
    CONTENT_COMPLETE = "content.complete"

    # Tools
    TOOL_CALL = "tool.call"
    TOOL_EXECUTING = "tool.executing"
    TOOL_RESULT = "tool.result"

    # Tasks
    TASK_PLAN = "task.plan"
    TASK_START = "task.start"
    TASK_PROGRESS = "task.progress"
    TASK_COMPLETE = "task.complete"

    # Context
    CONTEXT_LOADING = "context.loading"
    CONTEXT_TOOLS_READY = "context.tools_ready"
    CONTEXT_MEMORY_READY = "context.memory_ready"
    CONTEXT_COMPLETE = "context.complete"

    # Memory
    MEMORY_STORING = "memory.storing"
    MEMORY_STORED = "memory.stored"

    # System
    SYSTEM_READY = "system.ready"
    SYSTEM_BILLING = "system.billing"
    SYSTEM_ERROR = "system.error"

    # HIL (Human-in-the-loop)
    HIL_REQUEST = "hil.request"

    # Node
    NODE_ENTER = "node.enter"
    NODE_EXIT = "node.exit"

    # Legacy compatibility
    THINKING = "thinking"
    TOKEN = "token"
    CONTENT = "content"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AgentEvent:
    """
    Agent event wrapper - unified interface for all event types
    """
    type: str
    session_id: str
    timestamp: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def event_type(self) -> EventType:
        """Get enum event type"""
        try:
            return EventType(self.type)
        except ValueError:
            # Handle legacy types
            legacy_map = {
                "thinking": EventType.CONTENT_THINKING,
                "token": EventType.CONTENT_TOKEN,
                "content": EventType.CONTENT_COMPLETE,
                "tool_calls": EventType.TOOL_CALL,
                "error": EventType.SYSTEM_ERROR,
            }
            return legacy_map.get(self.type, EventType.SYSTEM_ERROR)

    # Convenience properties
    @property
    def is_content(self) -> bool:
        """Is this a content/token event?"""
        return self.type in ["content.token", "content.complete", "token", "content"]

    @property
    def is_thinking(self) -> bool:
        """Is this a thinking event?"""
        return self.type in ["content.thinking", "thinking"]

    @property
    def is_tool_call(self) -> bool:
        """Is this a tool call event?"""
        return self.type in ["tool.call", "tool_calls"]

    @property
    def is_tool_executing(self) -> bool:
        """Is tool executing?"""
        return self.type == "tool.executing"

    @property
    def is_tool_result(self) -> bool:
        """Is tool result?"""
        return self.type == "tool.result"

    @property
    def is_session_start(self) -> bool:
        """Is session start?"""
        return self.type == "session.start"

    @property
    def is_session_end(self) -> bool:
        """Is session end?"""
        return self.type == "session.end"

    @property
    def is_error(self) -> bool:
        """Is error event?"""
        return self.type in ["system.error", "error"]

    @property
    def is_billing(self) -> bool:
        """Is billing event?"""
        return self.type == "system.billing"

    @property
    def is_hil_request(self) -> bool:
        """Is human-in-the-loop request?"""
        return self.type == "hil.request"

    @property
    def is_context_ready(self) -> bool:
        """Is context ready?"""
        return self.type == "context.complete"

    # Tool information
    @property
    def tool_name(self) -> Optional[str]:
        """Get tool name if tool event"""
        if self.is_tool_call and self.metadata:
            return self.metadata.get("tool")
        return None

    @property
    def tool_args(self) -> Optional[Dict]:
        """Get tool arguments if tool call"""
        if self.is_tool_call and self.metadata:
            return self.metadata.get("args")
        return None

    # Billing information
    @property
    def credits_used(self) -> Optional[float]:
        """Get credits used if billing event"""
        if self.is_billing and self.metadata:
            return self.metadata.get("credits")
        return None

    # HIL information
    @property
    def hil_question(self) -> Optional[str]:
        """Get HIL question if HIL request"""
        return self.content if self.is_hil_request else None

    @property
    def hil_type(self) -> Optional[str]:
        """Get HIL request type"""
        if self.is_hil_request and self.metadata:
            return self.metadata.get("request_type")
        return None


@dataclass
class AgentResponse:
    """
    Complete agent response (non-streaming)
    """
    content: str
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    events: List[AgentEvent] = field(default_factory=list)

    @property
    def credits_used(self) -> Optional[float]:
        """Get total credits used"""
        for event in self.events:
            if event.is_billing:
                return event.credits_used
        return None

    @property
    def tools_used(self) -> List[str]:
        """Get list of tools that were called"""
        tools = []
        for event in self.events:
            if event.is_tool_call and event.tool_name:
                tools.append(event.tool_name)
        return tools

    @property
    def thinking_steps(self) -> List[str]:
        """Get list of thinking steps"""
        steps = []
        for event in self.events:
            if event.is_thinking and event.content:
                steps.append(event.content)
        return steps


@dataclass
class SessionInfo:
    """Session information"""
    session_id: str
    user_id: str
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Chat Interface
# ============================================================================

class Chat:
    """Chat interface for agent conversations"""

    def __init__(self, client: 'ISAAgent'):
        self._client = client

    def create(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_args: Optional[Dict] = None,
        graph_type: Optional[str] = None,
        auto_select_graph: bool = True,
        output_format: Optional[str] = None,
        device_context: Optional[Dict] = None,
        media_files: Optional[List[Dict]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Create a chat completion (non-streaming)

        Args:
            message: User message
            user_id: User identifier
            session_id: Session ID (auto-generated if not provided)
            prompt_name: Custom prompt template name
            prompt_args: Arguments for prompt template
            graph_type: Specific graph type to use
            auto_select_graph: Auto-select optimal graph
            output_format: Output format ("json" for structured)
            device_context: Hardware device context
            media_files: Media files (images, audio, etc.)
            **kwargs: Additional parameters

        Returns:
            AgentResponse with complete response
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._create_async(
                message, user_id, session_id, prompt_name, prompt_args,
                graph_type, auto_select_graph, output_format,
                device_context, media_files, **kwargs
            )
        )

    def stream(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_args: Optional[Dict] = None,
        graph_type: Optional[str] = None,
        auto_select_graph: bool = True,
        device_context: Optional[Dict] = None,
        media_files: Optional[List[Dict]] = None,
        **kwargs
    ) -> Iterator[AgentEvent]:
        """
        Stream agent response with events

        Args:
            message: User message
            user_id: User identifier
            session_id: Session ID (auto-generated if not provided)
            prompt_name: Custom prompt template name
            prompt_args: Arguments for prompt template
            graph_type: Specific graph type to use
            auto_select_graph: Auto-select optimal graph
            device_context: Hardware device context
            media_files: Media files (images, audio, etc.)
            **kwargs: Additional parameters

        Yields:
            AgentEvent objects for each event
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        gen = self._stream_async(
            message, user_id, session_id, prompt_name, prompt_args,
            graph_type, auto_select_graph, device_context, media_files, **kwargs
        )

        while True:
            try:
                event = loop.run_until_complete(gen.__anext__())
                yield event
            except StopAsyncIteration:
                break

    async def _create_async(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str],
        prompt_name: Optional[str],
        prompt_args: Optional[Dict],
        graph_type: Optional[str],
        auto_select_graph: bool,
        output_format: Optional[str],
        device_context: Optional[Dict],
        media_files: Optional[List[Dict]],
        **kwargs
    ) -> AgentResponse:
        """Async implementation of create"""
        session_id = session_id or self._client._generate_session_id(user_id)

        # Collect all events
        events = []
        content_parts = []

        async for event in self._stream_async(
            message, user_id, session_id, prompt_name, prompt_args,
            graph_type, auto_select_graph, device_context, media_files,
            output_format=output_format, **kwargs
        ):
            events.append(event)
            if event.is_content and event.content:
                content_parts.append(event.content)

        full_content = "".join(content_parts)

        return AgentResponse(
            content=full_content,
            session_id=session_id,
            metadata=events[-1].metadata if events else {},
            events=events
        )

    async def _stream_async(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str],
        prompt_name: Optional[str],
        prompt_args: Optional[Dict],
        graph_type: Optional[str],
        auto_select_graph: bool,
        device_context: Optional[Dict],
        media_files: Optional[List[Dict]],
        output_format: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[AgentEvent]:
        """Async implementation of stream"""
        session_id = session_id or self._client._generate_session_id(user_id)

        # Build request payload
        payload = {
            "message": message,
            "user_id": user_id,
            "session_id": session_id,
        }

        if prompt_name:
            payload["prompt_name"] = prompt_name
        if prompt_args:
            payload["prompt_args"] = prompt_args
        if graph_type:
            payload["graph_type"] = graph_type
        if auto_select_graph is not None:
            payload["auto_select_graph"] = auto_select_graph
        if output_format:
            payload["output_format"] = output_format
        if device_context:
            payload["device_context"] = device_context
        if media_files:
            payload["media_files"] = media_files

        payload.update(kwargs)

        # Stream SSE events
        async with self._client._http_client.stream(
            "POST",
            f"{self._client.base_url}/api/v1/agents/chat",
            json=payload,
            headers=self._client._get_headers(),
            timeout=300.0
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()

                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]  # Remove "data: " prefix

                if data == "[DONE]":
                    break

                try:
                    event_data = json.loads(data)

                    event = AgentEvent(
                        type=event_data.get("type", "unknown"),
                        session_id=event_data.get("session_id", session_id),
                        timestamp=event_data.get("timestamp", datetime.now().isoformat()),
                        content=event_data.get("content"),
                        metadata=event_data.get("metadata")
                    )

                    yield event

                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue


class Resume:
    """Resume interrupted sessions"""

    def __init__(self, client: 'ISAAgent'):
        self._client = client

    def create(
        self,
        user_id: str,
        session_id: str,
        resume_value: Optional[Dict] = None,
        prompt_name: Optional[str] = None,
        prompt_args: Optional[Dict] = None,
        **kwargs
    ) -> AgentResponse:
        """Resume an interrupted session (non-streaming)"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._create_async(
                user_id, session_id, resume_value,
                prompt_name, prompt_args, **kwargs
            )
        )

    def stream(
        self,
        user_id: str,
        session_id: str,
        resume_value: Optional[Dict] = None,
        prompt_name: Optional[str] = None,
        prompt_args: Optional[Dict] = None,
        **kwargs
    ) -> Iterator[AgentEvent]:
        """Resume session with streaming"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        gen = self._stream_async(
            user_id, session_id, resume_value,
            prompt_name, prompt_args, **kwargs
        )

        while True:
            try:
                event = loop.run_until_complete(gen.__anext__())
                yield event
            except StopAsyncIteration:
                break

    async def _create_async(
        self,
        user_id: str,
        session_id: str,
        resume_value: Optional[Dict],
        prompt_name: Optional[str],
        prompt_args: Optional[Dict],
        **kwargs
    ) -> AgentResponse:
        """Async implementation"""
        events = []
        content_parts = []

        async for event in self._stream_async(
            user_id, session_id, resume_value,
            prompt_name, prompt_args, **kwargs
        ):
            events.append(event)
            if event.is_content and event.content:
                content_parts.append(event.content)

        full_content = "".join(content_parts)

        return AgentResponse(
            content=full_content,
            session_id=session_id,
            events=events
        )

    async def _stream_async(
        self,
        user_id: str,
        session_id: str,
        resume_value: Optional[Dict],
        prompt_name: Optional[str],
        prompt_args: Optional[Dict],
        **kwargs
    ) -> AsyncIterator[AgentEvent]:
        """Async streaming implementation"""
        payload = {
            "user_id": user_id,
            "session_id": session_id,
        }

        if resume_value:
            payload["resume_value"] = resume_value
        if prompt_name:
            payload["prompt_name"] = prompt_name
        if prompt_args:
            payload["prompt_args"] = prompt_args

        payload.update(kwargs)

        async with self._client._http_client.stream(
            "POST",
            f"{self._client.base_url}/api/v1/agents/chat/resume",
            json=payload,
            headers=self._client._get_headers(),
            timeout=300.0
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()

                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                try:
                    event_data = json.loads(data)

                    event = AgentEvent(
                        type=event_data.get("type", "unknown"),
                        session_id=event_data.get("session_id", session_id),
                        timestamp=event_data.get("timestamp", datetime.now().isoformat()),
                        content=event_data.get("content"),
                        metadata=event_data.get("metadata")
                    )

                    yield event

                except json.JSONDecodeError:
                    continue


# ============================================================================
# Main Client
# ============================================================================

class ISAAgent:
    """
    OpenAI-compatible client for isA_Agent service

    Usage:
        # Local/default mode
        client = ISAAgent()

        # Remote API mode
        client = ISAAgent(api_key="your-key", base_url="http://localhost:8000")

        # Simple chat
        response = client.chat.create(
            message="Hello!",
            user_id="user123"
        )
        print(response.content)

        # Streaming
        for event in client.chat.stream(
            message="Explain AI",
            user_id="user123"
        ):
            if event.is_content:
                print(event.content, end="")
            elif event.is_tool_call:
                print(f"[Tool: {event.tool_name}]")

        # Resume interrupted session
        response = client.resume.create(
            user_id="user123",
            session_id="session456",
            resume_value={"authorized": True}
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        **kwargs
    ):
        """
        Initialize isA_Agent client

        Args:
            api_key: API key (uses ISA_API_KEY env if not provided)
            base_url: Service URL (uses ISA_AGENT_URL env if not provided)
            timeout: Request timeout in seconds
            **kwargs: Additional HTTP client configuration
        """
        # Get API key
        self.api_key = api_key or os.getenv("ISA_API_KEY") or "dev_key_test"

        # Get base URL
        self.base_url = (
            base_url or
            os.getenv("ISA_AGENT_URL") or
            os.getenv("ISA_API_URL") or
            "http://localhost:8000"
        ).rstrip('/')

        # Create HTTP client
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            **kwargs
        )

        # Initialize endpoint namespaces
        self.chat = Chat(self)
        self.resume = Resume(self)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ISAAgent-Python/1.0.0"
        }

    def _generate_session_id(self, user_id: str) -> str:
        """Generate session ID"""
        import time
        timestamp = int(time.time() * 1000)
        return f"{user_id}_{timestamp}"

    async def close(self):
        """Close HTTP client"""
        await self._http_client.aclose()

    def __repr__(self):
        return f"ISAAgent(base_url='{self.base_url}')"

    # Context manager support
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ============================================================================
# Synchronous context manager
# ============================================================================

class ISAAgentSync:
    """Synchronous wrapper with context manager support"""

    def __init__(self, *args, **kwargs):
        self._agent = ISAAgent(*args, **kwargs)

    def __enter__(self):
        return self._agent

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self._agent.close())

    def __getattr__(self, name):
        return getattr(self._agent, name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ISAAgent",
    "ISAAgentSync",
    "AgentEvent",
    "AgentResponse",
    "SessionInfo",
    "EventType",
]
