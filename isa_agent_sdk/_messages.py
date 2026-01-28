#!/usr/bin/env python3
"""
isA Agent SDK - Message Types
=============================

SDK message adapter that wraps the existing isA event type system.
Provides Claude Agent SDK-compatible interface while using isA's EventData underneath.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Awaitable, Union, Type, TypeVar, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar('T', bound='BaseModel')

# Import existing isA event types - REUSE, don't duplicate!
from .agent_types.event_types import (
    EventType as ISAEventType,
    EventData,
    EventEmitter,
    EventCategory,
    is_streaming_event,
    is_content_event,
)


# Mapping from Claude SDK message type names to isA EventType
# This allows Claude SDK-style usage while using isA types internally
MESSAGE_TYPE_MAP = {
    # Claude SDK compatible names
    "text": ISAEventType.CONTENT_COMPLETE,
    "tool_use": ISAEventType.TOOL_CALL,
    "tool_result": ISAEventType.TOOL_RESULT,
    "result": ISAEventType.CONTENT_COMPLETE,
    "system": ISAEventType.SYSTEM_INFO,
    "thinking": ISAEventType.CONTENT_THINKING,
    "error": ISAEventType.SYSTEM_ERROR,

    # isA specific
    "checkpoint": ISAEventType.HIL_REQUEST,  # Map checkpoint to HIL
    "progress": ISAEventType.TASK_PROGRESS,
    "intent": ISAEventType.NODE_EXIT,  # Sense node classifies intent
    "session_start": ISAEventType.SESSION_START,
    "session_end": ISAEventType.SESSION_END,
    "hil_request": ISAEventType.HIL_REQUEST,
    "hil_response": ISAEventType.HIL_RESPONSE,
}

# Result subtypes for structured output handling
class ResultSubtype(str, Enum):
    """Subtypes for result messages (Claude Agent SDK compatible)"""
    SUCCESS = "success"
    """Output was generated and validated successfully"""

    ERROR_MAX_TURNS = "error_max_turns"
    """Agent hit maximum iteration limit"""

    ERROR_MAX_STRUCTURED_OUTPUT_RETRIES = "error_max_structured_output_retries"
    """Agent couldn't produce valid structured output after multiple attempts"""

    ERROR_TOOL_EXECUTION = "error_tool_execution"
    """Tool execution failed"""

    ERROR_MODEL = "error_model"
    """Model inference error"""

    INTERRUPTED = "interrupted"
    """Execution was interrupted (HIL)"""


# Reverse mapping for converting isA events to SDK message types
REVERSE_TYPE_MAP = {
    ISAEventType.CONTENT_COMPLETE: "text",
    ISAEventType.CONTENT_TOKEN: "text",
    ISAEventType.CONTENT_THINKING: "thinking",
    ISAEventType.TOOL_CALL: "tool_use",
    ISAEventType.TOOL_RESULT: "tool_result",
    ISAEventType.TOOL_EXECUTING: "tool_use",
    ISAEventType.SYSTEM_ERROR: "error",
    ISAEventType.SYSTEM_INFO: "system",
    ISAEventType.SYSTEM_WARNING: "system",
    ISAEventType.SESSION_START: "session_start",
    ISAEventType.SESSION_END: "session_end",
    ISAEventType.HIL_REQUEST: "hil_request",
    ISAEventType.HIL_RESPONSE: "hil_response",
    ISAEventType.TASK_PROGRESS: "progress",
    ISAEventType.TASK_PLAN: "progress",
    ISAEventType.TASK_START: "progress",
    ISAEventType.TASK_COMPLETE: "result",
    ISAEventType.NODE_ENTER: "node_enter",
    ISAEventType.NODE_EXIT: "node_exit",
    ISAEventType.CONTEXT_COMPLETE: "system",
}


@dataclass
class AgentMessage:
    """
    Agent message - SDK wrapper around isA's EventData.

    Provides Claude Agent SDK-compatible interface while using isA's event system internally.

    Example:
        async for msg in query(prompt):
            if msg.is_text:
                print(msg.content, end="")
            elif msg.is_tool_use:
                print(f"[Using tool: {msg.tool_name}]")
            elif msg.is_checkpoint:
                await msg.respond({"continue": True})
    """

    type: str
    """Message type string (e.g., "text", "tool_use", "thinking")"""

    content: Optional[str] = None
    """Message content (for text, thinking, error, etc.)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata specific to message type"""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """ISO timestamp of when message was created"""

    session_id: Optional[str] = None
    """Session this message belongs to"""

    structured_output: Optional[Dict[str, Any]] = None
    """
    Validated structured output data (for result messages with output_format).

    When using structured outputs, this field contains the validated JSON
    matching your schema. Use `parse()` to convert to a Pydantic model.

    Example:
        if msg.structured_output:
            data = msg.structured_output
            # Or parse to Pydantic model:
            user = msg.parse(UserModel)
    """

    subtype: Optional[str] = None
    """
    Result subtype indicating success or error type (Claude Agent SDK compatible).

    Values: 'success', 'error_max_turns', 'error_max_structured_output_retries',
            'error_tool_execution', 'error_model', 'interrupted'
    """

    # Reference to original isA EventData (if created from one)
    _event_data: Optional[EventData] = field(default=None, repr=False)

    # Internal: Pydantic model reference for parsing
    _pydantic_model: Optional[Type["BaseModel"]] = field(default=None, repr=False)

    # Internal callback for checkpoint responses
    _response_callback: Optional[Callable[[Dict], Awaitable[None]]] = field(
        default=None, repr=False
    )

    @classmethod
    def from_event_data(cls, event: EventData) -> "AgentMessage":
        """
        Create AgentMessage from isA EventData.

        This is the primary way to convert isA's internal events to SDK messages.

        Args:
            event: isA EventData instance

        Returns:
            AgentMessage with SDK-compatible type
        """
        # Map isA EventType to SDK message type string
        isa_type = ISAEventType(event.type) if isinstance(event.type, str) else event.type
        sdk_type = REVERSE_TYPE_MAP.get(isa_type, "system")

        return cls(
            type=sdk_type,
            content=event.content,
            metadata=event.metadata or {},
            timestamp=event.timestamp,
            session_id=event.session_id,
            _event_data=event,
        )

    def to_event_data(self) -> EventData:
        """
        Convert to isA EventData.

        Returns:
            EventData instance for use with isA's internal systems
        """
        if self._event_data:
            return self._event_data

        # Map SDK type to isA EventType
        isa_type = MESSAGE_TYPE_MAP.get(self.type, ISAEventType.SYSTEM_INFO)

        return EventData(
            type=isa_type.value,
            session_id=self.session_id or "unknown",
            timestamp=self.timestamp,
            content=self.content,
            metadata=self.metadata,
        )

    @property
    def event_type(self) -> ISAEventType:
        """Get the underlying isA EventType"""
        if self._event_data:
            return ISAEventType(self._event_data.type)
        return MESSAGE_TYPE_MAP.get(self.type, ISAEventType.SYSTEM_INFO)

    # === Convenience Properties ===

    @property
    def is_text(self) -> bool:
        """Is this a text content message?"""
        return self.type == "text"

    @property
    def is_thinking(self) -> bool:
        """Is this a thinking/reasoning message?"""
        return self.type == "thinking"

    @property
    def is_tool_use(self) -> bool:
        """Is this a tool use message?"""
        return self.type == "tool_use"

    @property
    def is_tool_result(self) -> bool:
        """Is this a tool result message?"""
        return self.type == "tool_result"

    @property
    def is_checkpoint(self) -> bool:
        """Is this a checkpoint requiring human input?"""
        return self.type in ("checkpoint", "hil_request")

    @property
    def is_error(self) -> bool:
        """Is this an error message?"""
        return self.type == "error"

    @property
    def is_hil_request(self) -> bool:
        """Is this a human-in-the-loop request?"""
        return self.type == "hil_request"

    @property
    def is_complete(self) -> bool:
        """Is this the final result message?"""
        return self.type == "result"

    @property
    def is_streaming(self) -> bool:
        """Is this a streaming content event?"""
        if self._event_data:
            return is_streaming_event(ISAEventType(self._event_data.type))
        return self.type in ("text", "thinking")

    # === Tool Information ===

    @property
    def tool_name(self) -> Optional[str]:
        """Get tool name if this is a tool message"""
        if self.type in ("tool_use", "tool_result"):
            return self.metadata.get("tool_name") or self.metadata.get("tool")
        return None

    @property
    def tool_args(self) -> Optional[Dict]:
        """Get tool arguments if this is a tool_use message"""
        if self.type == "tool_use":
            return self.metadata.get("args") or self.metadata.get("input")
        return None

    @property
    def tool_result_value(self) -> Optional[Any]:
        """Get tool result if this is a tool_result message"""
        if self.type == "tool_result":
            return self.metadata.get("result") or self.metadata.get("output") or self.metadata.get("result_preview")
        return None

    @property
    def tool_error(self) -> Optional[str]:
        """Get tool error if tool execution failed"""
        if self.type == "tool_result":
            return self.metadata.get("error")
        return None

    # === Skill Information ===

    @property
    def skill_name(self) -> Optional[str]:
        """Get skill name if this is a skill message"""
        return self.metadata.get("skill_name")

    # === Intent Information ===

    @property
    def intent(self) -> Optional[str]:
        """Get classified intent if available"""
        return self.metadata.get("intent")

    @property
    def is_simple_intent(self) -> bool:
        """Is the classified intent simple (no complex processing needed)?"""
        return self.metadata.get("intent") == "simple"

    # === Checkpoint/HIL Response ===

    async def respond(self, response: Dict[str, Any]) -> None:
        """
        Respond to a checkpoint or HIL request.

        Args:
            response: Response data (e.g., {"continue": True} or {"authorized": True})

        Raises:
            ValueError: If this message doesn't support responses
        """
        if self._response_callback is None:
            raise ValueError(
                f"Cannot respond to message type {self.type}. "
                "Only checkpoint and hil_request messages support responses."
            )
        await self._response_callback(response)

    # === Node Information ===

    @property
    def node_name(self) -> Optional[str]:
        """Get node name for node enter/exit events"""
        if self.type in ("node_enter", "node_exit"):
            return self.metadata.get("node")
        return None

    # === Progress Information ===

    @property
    def progress_percent(self) -> Optional[float]:
        """Get progress percentage if this is a progress message"""
        if self.type == "progress":
            return self.metadata.get("percent") or self.metadata.get("progress")
        return None

    @property
    def progress_step(self) -> Optional[str]:
        """Get current step description if this is a progress message"""
        if self.type == "progress":
            return self.metadata.get("step") or self.content
        return None

    # === Structured Output Methods ===

    @property
    def has_structured_output(self) -> bool:
        """Check if this message has structured output data"""
        return self.structured_output is not None

    @property
    def is_structured_output_success(self) -> bool:
        """Check if structured output was generated successfully"""
        return self.subtype == ResultSubtype.SUCCESS.value or (
            self.has_structured_output and self.subtype is None
        )

    @property
    def is_structured_output_error(self) -> bool:
        """Check if structured output generation failed"""
        return self.subtype == ResultSubtype.ERROR_MAX_STRUCTURED_OUTPUT_RETRIES.value

    def parse(self, model: Type[T]) -> T:
        """
        Parse structured_output into a Pydantic model.

        Args:
            model: Pydantic BaseModel class to parse into

        Returns:
            Validated instance of the model

        Raises:
            ValueError: If no structured_output is available
            ValidationError: If data doesn't match the model schema

        Example:
            from pydantic import BaseModel

            class User(BaseModel):
                name: str
                age: int

            async for msg in query(prompt, options=ISAAgentOptions(
                output_format=OutputFormat.from_pydantic(User)
            )):
                if msg.has_structured_output:
                    user = msg.parse(User)
                    print(f"Name: {user.name}, Age: {user.age}")
        """
        if self.structured_output is None:
            raise ValueError(
                "No structured_output available. "
                "Ensure you're using output_format in your query options."
            )

        # Use Pydantic's model_validate for proper validation
        return model.model_validate(self.structured_output)

    def parse_or_none(self, model: Type[T]) -> Optional[T]:
        """
        Parse structured_output into a Pydantic model, returning None on failure.

        Args:
            model: Pydantic BaseModel class to parse into

        Returns:
            Validated instance of the model, or None if parsing fails

        Example:
            user = msg.parse_or_none(User)
            if user:
                print(f"Got user: {user.name}")
        """
        if self.structured_output is None:
            return None

        try:
            return model.model_validate(self.structured_output)
        except Exception:
            return None

    # === Factory Methods ===

    @classmethod
    def text(cls, content: str, session_id: Optional[str] = None) -> "AgentMessage":
        """Create a text message"""
        # Use isA's EventEmitter for consistency
        event = EventEmitter.content_complete(session_id or "unknown", content)
        msg = cls.from_event_data(event)
        msg.type = "text"  # Override to SDK type
        return msg

    @classmethod
    def thinking(cls, content: str, session_id: Optional[str] = None) -> "AgentMessage":
        """Create a thinking message"""
        event = EventEmitter.content_thinking(session_id or "unknown", content)
        msg = cls.from_event_data(event)
        msg.type = "thinking"
        return msg

    @classmethod
    def tool_use(
        cls,
        tool_name: str,
        args: Dict[str, Any],
        session_id: Optional[str] = None,
        tool_use_id: Optional[str] = None
    ) -> "AgentMessage":
        """Create a tool use message"""
        event = EventEmitter.tool_call(session_id or "unknown", tool_name, args)
        msg = cls.from_event_data(event)
        msg.type = "tool_use"
        msg.metadata["tool_use_id"] = tool_use_id
        return msg

    @classmethod
    def tool_result(
        cls,
        tool_name: str,
        result: Any,
        error: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_use_id: Optional[str] = None
    ) -> "AgentMessage":
        """Create a tool result message"""
        event = EventEmitter.tool_result(session_id or "unknown", tool_name, result)
        msg = cls.from_event_data(event)
        msg.type = "tool_result"
        msg.metadata["result"] = result
        msg.metadata["error"] = error
        msg.metadata["tool_use_id"] = tool_use_id
        return msg

    @classmethod
    def checkpoint(
        cls,
        content: str,
        session_id: Optional[str] = None,
        response_callback: Optional[Callable[[Dict], Awaitable[None]]] = None
    ) -> "AgentMessage":
        """Create a checkpoint message"""
        event = EventEmitter.hil_request(
            session_id or "unknown",
            request_type="checkpoint",
            question=content
        )
        msg = cls.from_event_data(event)
        msg.type = "checkpoint"
        msg._response_callback = response_callback
        return msg

    @classmethod
    def skill_activated(
        cls,
        skill_name: str,
        session_id: Optional[str] = None
    ) -> "AgentMessage":
        """Create a skill activated message"""
        return cls(
            type="skill_activated",
            content=f"Skill activated: {skill_name}",
            metadata={"skill_name": skill_name},
            session_id=session_id
        )

    @classmethod
    def intent_classification(
        cls,
        intent_type: str,
        confidence: float = 1.0,
        session_id: Optional[str] = None
    ) -> "AgentMessage":
        """Create an intent classification message"""
        return cls(
            type="intent",
            content=f"Intent: {intent_type}",
            metadata={"intent": intent_type, "confidence": confidence},
            session_id=session_id
        )

    @classmethod
    def progress(
        cls,
        step: str,
        percent: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> "AgentMessage":
        """Create a progress message"""
        event = EventEmitter.task_progress(session_id or "unknown", step, percent=percent)
        msg = cls.from_event_data(event)
        msg.type = "progress"
        return msg

    @classmethod
    def error(
        cls,
        content: str,
        error_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> "AgentMessage":
        """Create an error message"""
        event = EventEmitter.system_error(session_id or "unknown", content, error_type=error_type)
        msg = cls.from_event_data(event)
        msg.type = "error"
        return msg

    @classmethod
    def result(
        cls,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        structured_output: Optional[Dict[str, Any]] = None,
        subtype: Optional[str] = None,
        pydantic_model: Optional[Type["BaseModel"]] = None
    ) -> "AgentMessage":
        """
        Create a final result message.

        Args:
            content: Text content of the result
            session_id: Session ID
            metadata: Additional metadata
            structured_output: Validated structured output data (for json_schema format)
            subtype: Result subtype ('success', 'error_max_structured_output_retries', etc.)
            pydantic_model: Optional Pydantic model for parsing

        Returns:
            AgentMessage with type="result"
        """
        event = EventEmitter.content_complete(session_id or "unknown", content)
        msg = cls.from_event_data(event)
        msg.type = "result"
        msg.structured_output = structured_output
        msg.subtype = subtype or (ResultSubtype.SUCCESS.value if structured_output else None)
        msg._pydantic_model = pydantic_model
        if metadata:
            msg.metadata.update(metadata)
        return msg

    @classmethod
    def hil_request(
        cls,
        question: str,
        request_type: str,
        options: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        response_callback: Optional[Callable[[Dict], Awaitable[None]]] = None
    ) -> "AgentMessage":
        """Create a human-in-the-loop request message"""
        event = EventEmitter.hil_request(
            session_id or "unknown",
            request_type=request_type,
            question=question,
            options=options
        )
        msg = cls.from_event_data(event)
        msg.type = "hil_request"
        msg._response_callback = response_callback
        return msg

    @classmethod
    def session_start(cls, session_id: str, **metadata) -> "AgentMessage":
        """Create a session start message"""
        event = EventEmitter.session_start(session_id, **metadata)
        msg = cls.from_event_data(event)
        msg.type = "session_start"
        return msg

    @classmethod
    def session_end(cls, session_id: str, **metadata) -> "AgentMessage":
        """Create a session end message"""
        event = EventEmitter.session_end(session_id, **metadata)
        msg = cls.from_event_data(event)
        msg.type = "session_end"
        return msg

    @classmethod
    def from_langchain_message(
        cls,
        msg: Any,
        session_id: Optional[str] = None
    ) -> "AgentMessage":
        """
        Convert a LangChain message to AgentMessage.

        Args:
            msg: LangChain message (AIMessage, HumanMessage, ToolMessage, etc.)
            session_id: Session ID

        Returns:
            AgentMessage instance
        """
        from langchain_core.messages import AIMessage, ToolMessage

        if isinstance(msg, AIMessage):
            # Check for tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_call = msg.tool_calls[0]
                return cls.tool_use(
                    tool_name=tool_call.get('name', 'unknown'),
                    args=tool_call.get('args', {}),
                    session_id=session_id,
                    tool_use_id=tool_call.get('id')
                )
            # Regular text content
            return cls.text(content=msg.content, session_id=session_id)

        elif isinstance(msg, ToolMessage):
            return cls.tool_result(
                tool_name=msg.name if hasattr(msg, 'name') else 'unknown',
                result=msg.content,
                session_id=session_id,
                tool_use_id=msg.tool_call_id if hasattr(msg, 'tool_call_id') else None
            )

        else:
            # Generic message
            return cls.text(
                content=str(msg.content) if hasattr(msg, 'content') else str(msg),
                session_id=session_id
            )


@dataclass
class ConversationHistory:
    """Collection of messages in a conversation"""

    messages: List[AgentMessage] = field(default_factory=list)
    session_id: Optional[str] = None

    def add(self, message: AgentMessage) -> None:
        """Add a message to the history"""
        if message.session_id is None:
            message.session_id = self.session_id
        self.messages.append(message)

    def add_user_message(self, content: str) -> AgentMessage:
        """Add a user message to the history"""
        msg = AgentMessage(
            type="user",
            content=content,
            session_id=self.session_id,
            metadata={"role": "user"}
        )
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str) -> AgentMessage:
        """Add an assistant message to the history"""
        msg = AgentMessage.text(content, session_id=self.session_id)
        msg.metadata["role"] = "assistant"
        self.messages.append(msg)
        return msg

    def add_event(self, event: EventData) -> AgentMessage:
        """Add an isA EventData and return the converted AgentMessage"""
        msg = AgentMessage.from_event_data(event)
        self.add(msg)
        return msg

    def get_text_content(self) -> str:
        """Get all text content concatenated"""
        return "".join(
            msg.content for msg in self.messages
            if msg.is_text and msg.content
        )

    def get_tool_calls(self) -> List[AgentMessage]:
        """Get all tool use messages"""
        return [msg for msg in self.messages if msg.is_tool_use]

    def get_thinking(self) -> List[str]:
        """Get all thinking content"""
        return [
            msg.content for msg in self.messages
            if msg.is_thinking and msg.content
        ]

    def to_event_data_list(self) -> List[EventData]:
        """Convert all messages to isA EventData list"""
        return [msg.to_event_data() for msg in self.messages]

    @property
    def last_message(self) -> Optional[AgentMessage]:
        """Get the last message"""
        return self.messages[-1] if self.messages else None

    @property
    def is_complete(self) -> bool:
        """Check if conversation has a final result"""
        return any(msg.is_complete for msg in self.messages)


# =============================================================================
# Claude SDK Compatible Message Types
# =============================================================================

@dataclass
class TextBlock:
    """Text content block (Claude SDK compatible)"""
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    """Tool use block (Claude SDK compatible)"""
    id: str
    name: str
    input: Dict[str, Any]
    type: str = "tool_use"


@dataclass
class ToolResultBlock:
    """Tool result block (Claude SDK compatible)"""
    tool_use_id: str
    content: Any
    is_error: bool = False
    type: str = "tool_result"


@dataclass
class AssistantMessage:
    """Assistant message with content blocks (Claude SDK compatible)"""
    content: List[Union[TextBlock, ToolUseBlock]]
    model: Optional[str] = None
    role: str = "assistant"

    @classmethod
    def from_agent_message(cls, msg: AgentMessage) -> "AssistantMessage":
        """Convert AgentMessage to AssistantMessage"""
        blocks = []
        if msg.is_text or msg.is_thinking:
            blocks.append(TextBlock(text=msg.content or ""))
        elif msg.is_tool_use:
            blocks.append(ToolUseBlock(
                id=msg.metadata.get("tool_use_id", ""),
                name=msg.tool_name or "",
                input=msg.tool_args or {}
            ))
        return cls(content=blocks, model=msg.metadata.get("model"))


@dataclass
class ResultMessage:
    """
    Final result message (Claude SDK compatible).

    Contains the final result of agent execution, including structured output
    if output_format was specified.

    Attributes:
        subtype: Result status - 'success', 'error_max_structured_output_retries', etc.
        structured_output: Validated JSON data matching the requested schema
        result: Text content of the result
        duration_ms: Execution time in milliseconds
        num_turns: Number of agent turns taken
        total_cost_usd: Estimated cost in USD
        usage: Token usage breakdown
        is_error: Whether this is an error result
        session_id: Session identifier
    """
    subtype: str
    """Result status: 'success', 'error_max_turns', 'error_max_structured_output_retries', etc."""

    structured_output: Optional[Dict[str, Any]] = None
    """Validated structured output data when using output_format"""

    result: Optional[str] = None
    """Text content of the result"""

    duration_ms: Optional[float] = None
    """Execution time in milliseconds"""

    num_turns: Optional[int] = None
    """Number of agent turns taken"""

    total_cost_usd: Optional[float] = None
    """Estimated cost in USD"""

    usage: Optional[Dict[str, Any]] = None
    """Token usage breakdown"""

    is_error: bool = False
    """Whether this is an error result"""

    session_id: Optional[str] = None
    """Session identifier"""

    # Internal: Pydantic model reference for parsing
    _pydantic_model: Optional[Type["BaseModel"]] = field(default=None, repr=False)

    def parse(self, model: Type[T]) -> T:
        """
        Parse structured_output into a Pydantic model.

        Args:
            model: Pydantic BaseModel class to parse into

        Returns:
            Validated instance of the model

        Raises:
            ValueError: If no structured_output is available
        """
        if self.structured_output is None:
            raise ValueError("No structured_output available")
        return model.model_validate(self.structured_output)

    def parse_or_none(self, model: Type[T]) -> Optional[T]:
        """Parse structured_output, returning None on failure"""
        if self.structured_output is None:
            return None
        try:
            return model.model_validate(self.structured_output)
        except Exception:
            return None

    @property
    def is_success(self) -> bool:
        """Check if this is a successful result"""
        return self.subtype == ResultSubtype.SUCCESS.value

    @property
    def is_structured_output_error(self) -> bool:
        """Check if structured output validation failed"""
        return self.subtype == ResultSubtype.ERROR_MAX_STRUCTURED_OUTPUT_RETRIES.value


@dataclass
class SystemMessage:
    """System message (Claude SDK compatible)"""
    subtype: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class HookContext:
    """Context passed to hook callbacks (Claude SDK compatible)"""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_use_id: Optional[str] = None
    session_id: Optional[str] = None


# Re-export isA event types for convenience
__all__ = [
    # SDK types
    "AgentMessage",
    "ConversationHistory",
    "ResultSubtype",

    # Claude SDK compatible message types
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "AssistantMessage",
    "ResultMessage",
    "SystemMessage",
    "HookContext",

    # isA types (re-exported for convenience)
    "ISAEventType",
    "EventData",
    "EventEmitter",
    "EventCategory",

    # Mapping helpers
    "MESSAGE_TYPE_MAP",
    "REVERSE_TYPE_MAP",
]
