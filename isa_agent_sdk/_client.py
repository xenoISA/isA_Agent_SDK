#!/usr/bin/env python3
"""
isA Agent SDK - Bidirectional Client
====================================

Unified client for interactive agent conversations.
Supports both local (LangGraph) and remote (HTTP API) execution.

Example:
    from isa_agent_sdk import ISAAgentClient, ISAAgentOptions

    # Local execution (uses LangGraph directly)
    async with ISAAgentClient() as client:
        await client.query("Analyze this codebase")
        async for msg in client.receive():
            print(msg.content)

        # Continue same conversation
        await client.query("Now fix the bug you found")
        async for msg in client.receive():
            print(msg.content)

    # Remote execution (uses HTTP API)
    async with ISAAgentClient(base_url="http://api.example.com") as client:
        await client.query("Hello!")
        async for msg in client.receive():
            print(msg.content)

    # With options
    async with ISAAgentClient(
        options=ISAAgentOptions(
            allowed_tools=["Read", "Edit"],
            model="deepseek-reasoner"
        )
    ) as client:
        ...
"""

import asyncio
import uuid
import json
import logging
from typing import Optional, Dict, Any, List, AsyncIterator, Union
from dataclasses import dataclass, field
from enum import Enum

from .options import ISAAgentOptions, ExecutionEnv
from .errors import (
    ISASDKError,
    SessionError,
    SessionNotFoundError,
    ConnectionError,
    ExecutionError,
)

logger = logging.getLogger(__name__)


class ClientMode(str, Enum):
    """Client execution mode"""
    LOCAL = "local"      # Direct LangGraph execution
    REMOTE = "remote"    # HTTP API calls


@dataclass
class ClientState:
    """Internal client state"""
    session_id: str
    user_id: str
    message_count: int = 0
    is_streaming: bool = False
    pending_query: Optional[str] = None
    last_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ISAAgentClient:
    """
    Bidirectional client for interactive agent conversations.

    Provides a unified interface for both local (LangGraph) and remote (HTTP)
    execution, with session management and streaming support.

    Usage:
        async with ISAAgentClient() as client:
            await client.query("Hello")
            async for msg in client.receive():
                print(msg.content)

            # Continue conversation
            await client.query("Tell me more")
            async for msg in client.receive():
                print(msg.content)

    Args:
        options: ISAAgentOptions for configuration
        base_url: If provided, use remote HTTP mode
        api_key: API key for remote mode
        session_id: Explicit session ID (auto-generated if not provided)
        user_id: User identifier
    """

    def __init__(
        self,
        options: Optional[ISAAgentOptions] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.options = options or ISAAgentOptions()

        # Determine mode
        self._base_url = base_url
        self._api_key = api_key
        self._mode = ClientMode.REMOTE if base_url else ClientMode.LOCAL

        # Session state
        self._state = ClientState(
            session_id=session_id or f"client_{uuid.uuid4().hex[:12]}",
            user_id=user_id or self.options.user_id or "sdk_user",
        )

        # Internal components (initialized in __aenter__)
        self._http_client = None
        self._executor = None
        self._current_stream: Optional[AsyncIterator] = None
        self._initialized = False

    @property
    def session_id(self) -> str:
        """Current session ID"""
        return self._state.session_id

    @property
    def user_id(self) -> str:
        """Current user ID"""
        return self._state.user_id

    @property
    def message_count(self) -> int:
        """Number of messages in this session"""
        return self._state.message_count

    @property
    def mode(self) -> ClientMode:
        """Current execution mode (local or remote)"""
        return self._mode

    @property
    def is_connected(self) -> bool:
        """Whether client is connected/initialized"""
        return self._initialized

    async def __aenter__(self) -> "ISAAgentClient":
        """Initialize client resources"""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup client resources"""
        await self.close()

    async def _initialize(self):
        """Initialize client based on mode"""
        if self._initialized:
            return

        if self._mode == ClientMode.REMOTE:
            await self._initialize_remote()
        else:
            await self._initialize_local()

        self._initialized = True
        logger.info(
            f"ISAAgentClient initialized | "
            f"mode={self._mode.value} | "
            f"session_id={self._state.session_id}"
        )

    async def _initialize_remote(self):
        """Initialize HTTP client for remote mode"""
        import httpx

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=300.0,
            headers={
                "Authorization": f"Bearer {self._api_key or 'dev_key'}",
                "Content-Type": "application/json",
            }
        )

    async def _initialize_local(self):
        """Initialize LangGraph executor for local mode"""
        from ._query import QueryExecutor

        # Configure options with session info
        self.options.session_id = self._state.session_id
        self.options.user_id = self._state.user_id

        self._executor = QueryExecutor(self.options)

    async def close(self):
        """Close client and cleanup resources"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._executor = None
        self._current_stream = None
        self._initialized = False

        logger.info(f"ISAAgentClient closed | session_id={self._state.session_id}")

    async def query(self, prompt: str, **kwargs) -> None:
        """
        Send a query to the agent.

        This queues the query for processing. Call receive() to get responses.

        Args:
            prompt: The user's message/request
            **kwargs: Additional options to merge with client options

        Raises:
            SessionError: If client is not initialized
            ExecutionError: If a query is already pending

        Example:
            await client.query("What files are in this directory?")
            async for msg in client.receive():
                print(msg.content)
        """
        if not self._initialized:
            raise SessionError(
                "Client not initialized. Use 'async with ISAAgentClient() as client:'",
                session_id=self._state.session_id
            )

        if self._state.is_streaming:
            raise ExecutionError(
                "Cannot send new query while receiving. Consume receive() first.",
                session_id=self._state.session_id
            )

        self._state.pending_query = prompt
        self._state.message_count += 1

        # Merge any kwargs into options
        merged_options = self._merge_options(**kwargs)

        # Start the stream based on mode
        if self._mode == ClientMode.REMOTE:
            self._current_stream = self._stream_remote(prompt, merged_options)
        else:
            self._current_stream = self._stream_local(prompt, merged_options)

        logger.debug(
            f"Query queued | "
            f"session_id={self._state.session_id} | "
            f"message_count={self._state.message_count}"
        )

    async def receive(self) -> AsyncIterator["AgentMessage"]:
        """
        Receive response messages from the agent.

        Yields messages as they stream in. Must be called after query().

        Yields:
            AgentMessage objects (text, tool_use, tool_result, etc.)

        Raises:
            SessionError: If no query is pending

        Example:
            await client.query("Hello")
            async for msg in client.receive():
                if msg.is_text:
                    print(msg.content, end="")
                elif msg.is_tool_use:
                    print(f"[Using tool: {msg.tool_name}]")
        """
        if self._current_stream is None:
            raise SessionError(
                "No pending query. Call query() first.",
                session_id=self._state.session_id
            )

        self._state.is_streaming = True
        response_parts = []

        try:
            async for msg in self._current_stream:
                # Collect response content
                if hasattr(msg, 'is_text') and msg.is_text and msg.content:
                    response_parts.append(msg.content)
                elif hasattr(msg, 'is_complete') and msg.is_complete and msg.content:
                    response_parts.append(msg.content)

                yield msg

            # Store last response
            self._state.last_response = "".join(response_parts)

        finally:
            self._state.is_streaming = False
            self._state.pending_query = None
            self._current_stream = None

    async def ask(self, prompt: str, **kwargs) -> str:
        """
        Convenience method: send query and return full text response.

        Combines query() and receive() into a single call.

        Args:
            prompt: The user's message/request
            **kwargs: Additional options

        Returns:
            Complete text response as string

        Example:
            response = await client.ask("What is 2 + 2?")
            print(response)  # "4"
        """
        await self.query(prompt, **kwargs)

        result = []
        async for msg in self.receive():
            if hasattr(msg, 'is_text') and msg.is_text and msg.content:
                result.append(msg.content)
            elif hasattr(msg, 'is_complete') and msg.is_complete and msg.content:
                result.append(msg.content)

        return "".join(result)

    async def resume(
        self,
        resume_value: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator["AgentMessage"]:
        """
        Resume from a checkpoint/HIL interrupt.

        Use this after receiving a checkpoint message to continue execution.

        Args:
            resume_value: Value to pass for resumption (e.g., {"authorized": True})

        Yields:
            AgentMessage objects from resumed execution

        Example:
            async for msg in client.receive():
                if msg.is_checkpoint:
                    # User approves
                    async for resumed_msg in client.resume({"continue": True}):
                        print(resumed_msg.content)
        """
        if self._mode == ClientMode.REMOTE:
            async for msg in self._resume_remote(resume_value):
                yield msg
        else:
            async for msg in self._resume_local(resume_value):
                yield msg

    def fork(self, new_session_id: Optional[str] = None) -> "ISAAgentClient":
        """
        Create a new client forked from current session state.

        The forked client shares history but diverges from this point.

        Args:
            new_session_id: Session ID for fork (auto-generated if not provided)

        Returns:
            New ISAAgentClient instance

        Example:
            # Explore an alternative approach
            forked = client.fork()
            async with forked:
                await forked.query("Try a different approach")
                ...
        """
        fork_id = new_session_id or f"{self._state.session_id}_fork_{uuid.uuid4().hex[:6]}"

        forked = ISAAgentClient(
            options=self.options,
            base_url=self._base_url,
            api_key=self._api_key,
            session_id=fork_id,
            user_id=self._state.user_id,
        )

        # Copy state
        forked._state.message_count = self._state.message_count
        forked._state.metadata = self._state.metadata.copy()

        logger.info(
            f"Session forked | "
            f"from={self._state.session_id} | "
            f"to={fork_id}"
        )

        return forked

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get current session information.

        Returns:
            Dict with session_id, user_id, message_count, mode, etc.
        """
        return {
            "session_id": self._state.session_id,
            "user_id": self._state.user_id,
            "message_count": self._state.message_count,
            "mode": self._mode.value,
            "is_streaming": self._state.is_streaming,
            "is_connected": self._initialized,
            "last_response_length": len(self._state.last_response) if self._state.last_response else 0,
        }

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _merge_options(self, **kwargs) -> ISAAgentOptions:
        """Merge kwargs into options"""
        if not kwargs:
            return self.options

        # Create a copy and update
        merged = ISAAgentOptions(
            **{k: v for k, v in self.options.__dict__.items() if not k.startswith('_')}
        )

        for key, value in kwargs.items():
            if hasattr(merged, key):
                setattr(merged, key, value)

        return merged

    async def _stream_local(
        self,
        prompt: str,
        options: ISAAgentOptions
    ) -> AsyncIterator["AgentMessage"]:
        """Stream using local LangGraph executor"""
        from ._query import query

        # Ensure session continuity
        options.session_id = self._state.session_id
        options.user_id = self._state.user_id

        async for msg in query(prompt, options):
            yield msg

    async def _stream_remote(
        self,
        prompt: str,
        options: ISAAgentOptions
    ) -> AsyncIterator["AgentMessage"]:
        """Stream using remote HTTP API"""
        from ._messages import AgentMessage

        payload = {
            "message": prompt,
            "user_id": self._state.user_id,
            "session_id": self._state.session_id,
        }

        # Add options to payload
        if options.model:
            payload["model"] = options.model
        if options.allowed_tools:
            payload["allowed_tools"] = options.allowed_tools

        try:
            async with self._http_client.stream(
                "POST",
                "/api/v1/agents/chat",
                json=payload,
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
                        msg = self._parse_remote_event(event_data)
                        if msg:
                            yield msg
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Remote stream error: {e}")
            raise ConnectionError(
                f"Failed to stream from remote API: {e}",
                service="agent_api",
                url=self._base_url
            )

    def _parse_remote_event(self, event_data: Dict[str, Any]) -> Optional["AgentMessage"]:
        """Parse remote API event into AgentMessage"""
        from ._messages import AgentMessage

        event_type = event_data.get("type", "")
        content = event_data.get("content", "")
        metadata = event_data.get("metadata", {})

        if event_type in ("content.token", "token", "text"):
            return AgentMessage.text(content, self._state.session_id)
        elif event_type in ("content.thinking", "thinking"):
            return AgentMessage.thinking(content, self._state.session_id)
        elif event_type in ("content.complete", "result"):
            return AgentMessage.result(content, self._state.session_id)
        elif event_type in ("tool.call", "tool_use"):
            return AgentMessage.tool_use(
                tool_name=metadata.get("tool", "unknown"),
                args=metadata.get("args", {}),
                session_id=self._state.session_id
            )
        elif event_type in ("tool.result", "tool_result"):
            return AgentMessage.tool_result(
                tool_name=metadata.get("tool", "unknown"),
                result=content,
                session_id=self._state.session_id
            )
        elif event_type == "system.error":
            return AgentMessage.error(content, self._state.session_id)

        return None

    async def _resume_local(
        self,
        resume_value: Optional[Dict[str, Any]]
    ) -> AsyncIterator["AgentMessage"]:
        """Resume using local LangGraph"""
        from ._query import resume

        async for msg in resume(
            self._state.session_id,
            resume_value,
            self.options
        ):
            yield msg

    async def _resume_remote(
        self,
        resume_value: Optional[Dict[str, Any]]
    ) -> AsyncIterator["AgentMessage"]:
        """Resume using remote HTTP API"""
        from ._messages import AgentMessage

        payload = {
            "user_id": self._state.user_id,
            "session_id": self._state.session_id,
        }

        if resume_value:
            payload["resume_value"] = resume_value

        try:
            async with self._http_client.stream(
                "POST",
                "/api/v1/agents/chat/resume",
                json=payload,
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
                        msg = self._parse_remote_event(event_data)
                        if msg:
                            yield msg
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Remote resume error: {e}")
            raise ConnectionError(
                f"Failed to resume from remote API: {e}",
                service="agent_api",
                url=self._base_url
            )


# ============================================================================
# Synchronous Wrapper
# ============================================================================

class ISAAgentClientSync:
    """
    Synchronous wrapper for ISAAgentClient.

    For use in non-async contexts.

    Example:
        with ISAAgentClientSync() as client:
            client.query("Hello")
            for msg in client.receive():
                print(msg.content)
    """

    def __init__(self, *args, **kwargs):
        self._async_client = ISAAgentClient(*args, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "ISAAgentClientSync":
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_client._initialize())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._loop:
            self._loop.run_until_complete(self._async_client.close())
            self._loop.close()
            self._loop = None

    @property
    def session_id(self) -> str:
        return self._async_client.session_id

    @property
    def user_id(self) -> str:
        return self._async_client.user_id

    def query(self, prompt: str, **kwargs) -> None:
        """Send query (sync)"""
        self._loop.run_until_complete(
            self._async_client.query(prompt, **kwargs)
        )

    def receive(self):
        """Receive messages (sync iterator)"""
        async_gen = self._async_client.receive()

        while True:
            try:
                msg = self._loop.run_until_complete(async_gen.__anext__())
                yield msg
            except StopAsyncIteration:
                break

    def ask(self, prompt: str, **kwargs) -> str:
        """Send query and get response (sync)"""
        return self._loop.run_until_complete(
            self._async_client.ask(prompt, **kwargs)
        )

    def get_session_info(self) -> Dict[str, Any]:
        return self._async_client.get_session_info()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ISAAgentClient",
    "ISAAgentClientSync",
    "ClientMode",
    "ClientState",
]
