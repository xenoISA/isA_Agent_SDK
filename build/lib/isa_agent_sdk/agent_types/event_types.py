#!/usr/bin/env python3
"""
Unified Event Types for isA Agent

Event naming convention: category.action
Categories: session, node, content, tool, task, hil, system
"""

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel


class EventCategory(str, Enum):
    """Event categories"""
    SESSION = "session"      # Session lifecycle
    NODE = "node"            # Node execution
    CONTENT = "content"      # Content generation
    TOOL = "tool"           # Tool execution
    TASK = "task"           # Task management (autonomous)
    HIL = "hil"             # Human-in-the-loop
    CONTEXT = "context"     # Context preparation & updates
    MEMORY = "memory"       # Memory storage & curation
    ARTIFACT = "artifact"   # Artifact generation (Claude-style)
    SYSTEM = "system"       # System events


class EventType(str, Enum):
    """All event types following category.action pattern"""
    
    # Session lifecycle events
    SESSION_START = "session.start"          # Session started
    SESSION_END = "session.end"              # Session ended
    SESSION_PAUSED = "session.paused"        # Session paused (HIL)
    SESSION_RESUMED = "session.resumed"      # Session resumed
    SESSION_ERROR = "session.error"          # Session error
    
    # Node execution events
    NODE_ENTER = "node.enter"               # Entering a node
    NODE_EXIT = "node.exit"                 # Exiting a node
    NODE_ERROR = "node.error"               # Node execution error
    NODE_SKIP = "node.skip"                 # Node skipped
    
    # Content generation events
    CONTENT_THINKING = "content.thinking"    # AI thinking (streaming)
    CONTENT_TOKEN = "content.token"         # Response token (streaming)
    CONTENT_COMPLETE = "content.complete"   # Complete content (non-streaming)
    CONTENT_ERROR = "content.error"         # Content generation error
    
    # Tool execution events
    TOOL_CALL = "tool.call"                # Tool call requested
    TOOL_EXECUTING = "tool.executing"       # Tool executing (progress)
    TOOL_RESULT = "tool.result"            # Tool execution result
    TOOL_ERROR = "tool.error"              # Tool execution error
    TOOL_SKIP = "tool.skip"                # Tool skipped (cached/unnecessary)
    
    # Task management events (autonomous execution)
    TASK_PLAN = "task.plan"                # Task plan created
    TASK_START = "task.start"              # Task started
    TASK_PROGRESS = "task.progress"        # Task progress update
    TASK_COMPLETE = "task.complete"        # Task completed
    TASK_ERROR = "task.error"              # Task error
    TASK_APPROVED = "task.approved"        # Task plan approved
    TASK_REJECTED = "task.rejected"        # Task plan rejected
    TASK_MODIFIED = "task.modified"        # Task plan modified
    
    # Human-in-the-loop events
    HIL_REQUEST = "hil.request"            # HIL requested
    HIL_RESPONSE = "hil.response"          # HIL response received
    HIL_TIMEOUT = "hil.timeout"            # HIL timeout
    HIL_CANCEL = "hil.cancel"              # HIL cancelled
    
    # Context preparation events (5个核心组件)
    CONTEXT_LOADING = "context.loading"         # Context preparation started
    CONTEXT_TOOLS_READY = "context.tools_ready" # MCP Tools loaded (default + search)
    CONTEXT_PROMPTS_READY = "context.prompts_ready"  # MCP Prompts loaded
    CONTEXT_RESOURCES_READY = "context.resources_ready"  # MCP Resources loaded  
    CONTEXT_MEMORY_READY = "context.memory_ready"    # Memory context loaded
    CONTEXT_KNOWLEDGE_READY = "context.knowledge_ready"  # Knowledge/Files context loaded
    CONTEXT_COMPLETE = "context.complete"       # All context components ready
    CONTEXT_ERROR = "context.error"             # Context preparation error
    
    # Memory storage and curation events
    MEMORY_STORING = "memory.storing"      # Storing conversation memories
    MEMORY_STORED = "memory.stored"        # Memory successfully stored
    MEMORY_CURATING = "memory.curating"    # Memory curation in progress
    MEMORY_CURATED = "memory.curated"      # Memory curation completed
    MEMORY_ERROR = "memory.error"          # Memory operation error

    # Artifact generation events (Claude-style interactive content)
    ARTIFACT_GENERATING = "artifact.generating"  # Artifact generation in progress
    ARTIFACT_GENERATED = "artifact.generated"    # Artifact successfully generated
    ARTIFACT_UPDATED = "artifact.updated"        # Artifact updated
    ARTIFACT_ERROR = "artifact.error"            # Artifact generation error

    # System events
    SYSTEM_READY = "system.ready"          # System ready (context loaded)
    SYSTEM_BILLING = "system.billing"      # Billing event
    SYSTEM_ERROR = "system.error"          # System error
    SYSTEM_WARNING = "system.warning"      # System warning
    SYSTEM_INFO = "system.info"            # System info
    SYSTEM_METRICS = "system.metrics"      # Performance metrics

    # Debug events (state inspection)
    STATE_SNAPSHOT = "state.snapshot"      # Full state snapshot (for debugging)


class EventData(BaseModel):
    """Standard event data structure"""
    type: str                               # Event type (category.action)
    session_id: str                         # Session identifier
    timestamp: str                          # ISO format timestamp
    content: Optional[str] = None          # Main content/message
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    class Config:
        use_enum_values = True
    
    @classmethod
    def create(
        cls,
        event_type: EventType,
        session_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "EventData":
        """Factory method to create event data"""
        return cls(
            type=event_type.value,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            content=content,
            metadata=metadata or {}
        )
    
    def to_sse(self) -> str:
        """Convert to SSE format"""
        import json
        return f"data: {json.dumps(self.dict(exclude_none=True))}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict(exclude_none=True)


class EventEmitter:
    """Helper class for emitting events with consistent format"""
    
    @staticmethod
    def session_start(session_id: str, **metadata) -> EventData:
        """Emit session start event"""
        return EventData.create(
            EventType.SESSION_START,
            session_id,
            content="Session started",
            metadata=metadata
        )
    
    @staticmethod
    def session_end(session_id: str, **metadata) -> EventData:
        """Emit session end event"""
        return EventData.create(
            EventType.SESSION_END,
            session_id,
            content="Session ended",
            metadata=metadata
        )
    
    @staticmethod
    def node_enter(session_id: str, node_name: str, **metadata) -> EventData:
        """Emit node enter event"""
        return EventData.create(
            EventType.NODE_ENTER,
            session_id,
            content=f"Entering {node_name}",
            metadata={"node": node_name, **metadata}
        )
    
    @staticmethod
    def node_exit(session_id: str, node_name: str, **metadata) -> EventData:
        """Emit node exit event"""
        return EventData.create(
            EventType.NODE_EXIT,
            session_id,
            content=f"Exiting {node_name}",
            metadata={"node": node_name, **metadata}
        )
    
    @staticmethod
    def content_thinking(session_id: str, token: str) -> EventData:
        """Emit thinking token event"""
        return EventData.create(
            EventType.CONTENT_THINKING,
            session_id,
            content=token,
            metadata={"streaming": True}
        )
    
    @staticmethod
    def content_token(session_id: str, token: str) -> EventData:
        """Emit response token event"""
        return EventData.create(
            EventType.CONTENT_TOKEN,
            session_id,
            content=token,
            metadata={"streaming": True}
        )
    
    @staticmethod
    def content_complete(session_id: str, content: str, node: str = "unknown") -> EventData:
        """Emit complete content event"""
        return EventData.create(
            EventType.CONTENT_COMPLETE,
            session_id,
            content=content,
            metadata={"streaming": False, "node": node, "length": len(content)}
        )
    
    @staticmethod
    def tool_call(session_id: str, tool_name: str, args: Dict[str, Any]) -> EventData:
        """Emit tool call event"""
        return EventData.create(
            EventType.TOOL_CALL,
            session_id,
            content=f"Calling {tool_name}",
            metadata={"tool": tool_name, "args": args}
        )
    
    @staticmethod
    def tool_executing(session_id: str, tool_name: str, progress: str) -> EventData:
        """Emit tool executing event"""
        return EventData.create(
            EventType.TOOL_EXECUTING,
            session_id,
            content=progress,
            metadata={"tool": tool_name}
        )
    
    @staticmethod
    def tool_result(session_id: str, tool_name: str, result: Any) -> EventData:
        """Emit tool result event"""
        result_str = str(result)[:200] if result else "No result"
        return EventData.create(
            EventType.TOOL_RESULT,
            session_id,
            content=f"Tool {tool_name} completed",
            metadata={"tool": tool_name, "result_preview": result_str}
        )
    
    @staticmethod
    def task_plan(session_id: str, plan: Dict[str, Any]) -> EventData:
        """Emit task plan event"""
        task_count = len(plan.get("tasks", []))
        return EventData.create(
            EventType.TASK_PLAN,
            session_id,
            content=f"Created plan with {task_count} tasks",
            metadata={"plan": plan, "task_count": task_count}
        )
    
    @staticmethod
    def hil_request(session_id: str, request_type: str, question: str, **metadata) -> EventData:
        """Emit HIL request event"""
        return EventData.create(
            EventType.HIL_REQUEST,
            session_id,
            content=question,
            metadata={"request_type": request_type, **metadata}
        )
    
    @staticmethod
    def system_ready(session_id: str, tools_count: int = 0, **metadata) -> EventData:
        """Emit system ready event"""
        return EventData.create(
            EventType.SYSTEM_READY,
            session_id,
            content=f"System ready with {tools_count} tools",
            metadata={"tools_count": tools_count, **metadata}
        )
    
    @staticmethod
    def system_billing(session_id: str, credits: float, **metadata) -> EventData:
        """Emit billing event"""
        return EventData.create(
            EventType.SYSTEM_BILLING,
            session_id,
            content=f"Used {credits} credits",
            metadata={"credits": credits, **metadata}
        )
    
    @staticmethod
    def system_error(session_id: str, error: str, **metadata) -> EventData:
        """Emit system error event"""
        return EventData.create(
            EventType.SYSTEM_ERROR,
            session_id,
            content=error,
            metadata=metadata
        )
    
    @staticmethod
    def system_warning(session_id: str, warning: str, **metadata) -> EventData:
        """Emit system warning event"""
        return EventData.create(
            EventType.SYSTEM_WARNING,
            session_id,
            content=warning,
            metadata=metadata
        )
    
    @staticmethod
    def system_info(session_id: str, info: str, **metadata) -> EventData:
        """Emit system info event"""
        return EventData.create(
            EventType.SYSTEM_INFO,
            session_id,
            content=info,
            metadata=metadata
        )
    
    @staticmethod
    def task_start(session_id: str, task_info: str, **metadata) -> EventData:
        """Emit task start event"""
        return EventData.create(
            EventType.TASK_START,
            session_id,
            content=task_info,
            metadata=metadata
        )
    
    @staticmethod
    def task_progress(session_id: str, progress_info: str, **metadata) -> EventData:
        """Emit task progress event"""
        return EventData.create(
            EventType.TASK_PROGRESS,
            session_id,
            content=progress_info,
            metadata=metadata
        )
    
    @staticmethod
    def task_complete(session_id: str, completion_info: str, **metadata) -> EventData:
        """Emit task complete event"""
        return EventData.create(
            EventType.TASK_COMPLETE,
            session_id,
            content=completion_info,
            metadata=metadata
        )
    
    @staticmethod
    def session_paused(session_id: str, reason: str = "Human input required", **metadata) -> EventData:
        """Emit session paused event"""
        return EventData.create(
            EventType.SESSION_PAUSED,
            session_id,
            content=reason,
            metadata=metadata
        )
    
    # Context preparation event emitters
    @staticmethod
    def context_loading(session_id: str, **metadata) -> EventData:
        """Emit context loading started event"""
        return EventData.create(
            EventType.CONTEXT_LOADING,
            session_id,
            content="⚙️ Preparing context...",
            metadata=metadata
        )
    
    @staticmethod
    def context_tools_ready(session_id: str, tools_count: int = 0, **metadata) -> EventData:
        """Emit tools ready event"""
        return EventData.create(
            EventType.CONTEXT_TOOLS_READY,
            session_id,
            content=f"🔧 Tools ready ({tools_count})",
            metadata={"tools_count": tools_count, **metadata}
        )
    
    @staticmethod
    def context_prompts_ready(session_id: str, prompts_count: int = 0, **metadata) -> EventData:
        """Emit prompts ready event"""
        return EventData.create(
            EventType.CONTEXT_PROMPTS_READY,
            session_id,
            content=f"📝 Prompts ready ({prompts_count})",
            metadata={"prompts_count": prompts_count, **metadata}
        )
    
    @staticmethod
    def context_resources_ready(session_id: str, resources_count: int = 0, **metadata) -> EventData:
        """Emit resources ready event"""
        return EventData.create(
            EventType.CONTEXT_RESOURCES_READY,
            session_id,
            content=f"📦 Resources ready ({resources_count})",
            metadata={"resources_count": resources_count, **metadata}
        )
    
    @staticmethod
    def context_memory_ready(session_id: str, memory_length: int = 0, **metadata) -> EventData:
        """Emit memory ready event"""
        return EventData.create(
            EventType.CONTEXT_MEMORY_READY,
            session_id,
            content=f"🧠 Memory ready ({memory_length} chars)",
            metadata={"memory_length": memory_length, **metadata}
        )
    
    @staticmethod
    def context_knowledge_ready(session_id: str, files_count: int = 0, **metadata) -> EventData:
        """Emit knowledge ready event"""
        return EventData.create(
            EventType.CONTEXT_KNOWLEDGE_READY,
            session_id,
            content=f"📚 Knowledge ready ({files_count} files)",
            metadata={"files_count": files_count, **metadata}
        )
    
    @staticmethod
    def context_complete(session_id: str, duration_ms: int = 0, **metadata) -> EventData:
        """Emit context complete event"""
        return EventData.create(
            EventType.CONTEXT_COMPLETE,
            session_id,
            content=f"✅ Context ready ({duration_ms}ms)",
            metadata={"duration_ms": duration_ms, **metadata}
        )
    
    @staticmethod
    def context_error(session_id: str, error: str, **metadata) -> EventData:
        """Emit context error event"""
        return EventData.create(
            EventType.CONTEXT_ERROR,
            session_id,
            content=f"❌ Context error: {error}",
            metadata=metadata
        )
    
    # Memory event emitters
    @staticmethod
    def memory_storing(session_id: str, **metadata) -> EventData:
        """Emit memory storing event"""
        return EventData.create(
            EventType.MEMORY_STORING,
            session_id,
            content="💾 Storing memories...",
            metadata=metadata
        )
    
    @staticmethod
    def memory_stored(session_id: str, memories_count: int = 0, **metadata) -> EventData:
        """Emit memory stored event"""
        return EventData.create(
            EventType.MEMORY_STORED,
            session_id,
            content=f"💾 Stored {memories_count} memories",
            metadata={"memories_count": memories_count, **metadata}
        )
    
    @staticmethod
    def memory_curating(session_id: str, curation_type: str = "auto", **metadata) -> EventData:
        """Emit memory curating event"""
        return EventData.create(
            EventType.MEMORY_CURATING,
            session_id,
            content=f"🎨 Curating memories ({curation_type})",
            metadata={"curation_type": curation_type, **metadata}
        )
    
    @staticmethod
    def memory_curated(session_id: str, curation_type: str = "auto", **metadata) -> EventData:
        """Emit memory curated event"""
        return EventData.create(
            EventType.MEMORY_CURATED,
            session_id,
            content=f"✨ Memory curated ({curation_type})",
            metadata={"curation_type": curation_type, **metadata}
        )
    
    @staticmethod
    def memory_error(session_id: str, error: str, **metadata) -> EventData:
        """Emit memory error event"""
        return EventData.create(
            EventType.MEMORY_ERROR,
            session_id,
            content=f"❌ Memory error: {error}",
            metadata=metadata
        )

    # Debug event emitters
    @staticmethod
    def state_snapshot(
        session_id: str,
        messages_count: int = 0,
        messages: list = None,
        **metadata
    ) -> EventData:
        """Emit state snapshot event for debugging"""
        return EventData.create(
            EventType.STATE_SNAPSHOT,
            session_id,
            content=f"📸 State snapshot: {messages_count} messages",
            metadata={
                "messages_count": messages_count,
                "messages": messages or [],
                **metadata
            }
        )


# Backward compatibility mapping (old -> new)
EVENT_MAPPING = {
    # Lifecycle
    "start": EventType.SESSION_START,
    "end": EventType.SESSION_END,
    "paused": EventType.SESSION_PAUSED,
    
    # Content
    "thinking": EventType.CONTENT_THINKING,
    "thinking_complete": EventType.CONTENT_COMPLETE,
    "token": EventType.CONTENT_TOKEN,
    "content": EventType.CONTENT_COMPLETE,
    "content.thinking": EventType.CONTENT_THINKING,
    "content.token": EventType.CONTENT_TOKEN,
    "content.complete": EventType.CONTENT_COMPLETE,
    
    # Tools
    "tool_calls": EventType.TOOL_CALL,
    "tool_result_msg": EventType.TOOL_RESULT,
    "progress": EventType.TOOL_EXECUTING,
    "tool_start": EventType.TOOL_EXECUTING,
    "tool_executing": EventType.TOOL_EXECUTING,
    "tool_completed": EventType.TOOL_RESULT,
    
    # Tasks
    "task_planning": EventType.TASK_PLAN,
    "task_start": EventType.TASK_START,
    "task_complete": EventType.TASK_COMPLETE,
    "task_status": EventType.TASK_PROGRESS,
    "task_progress": EventType.TASK_PROGRESS,
    "task_status_update": EventType.TASK_PROGRESS,
    
    # Context
    "context_progress": EventType.CONTEXT_LOADING,
    "context_ready": EventType.CONTEXT_COMPLETE,
    "context_loading": EventType.CONTEXT_LOADING,
    "context_tools_ready": EventType.CONTEXT_TOOLS_READY,
    "context_prompts_ready": EventType.CONTEXT_PROMPTS_READY,
    "context_resources_ready": EventType.CONTEXT_RESOURCES_READY,
    "context_memory_ready": EventType.CONTEXT_MEMORY_READY,
    "context_knowledge_ready": EventType.CONTEXT_KNOWLEDGE_READY,
    "context_complete": EventType.CONTEXT_COMPLETE,
    
    # Memory
    "memory_storing": EventType.MEMORY_STORING,
    "memory_stored": EventType.MEMORY_STORED,
    "memory_curating": EventType.MEMORY_CURATING,
    "memory_curated": EventType.MEMORY_CURATED,
    
    # Nodes
    "node_update": EventType.NODE_EXIT,
    "node.exit": EventType.NODE_EXIT,
    "node.enter": EventType.NODE_ENTER,
    
    # HIL
    "interrupt": EventType.HIL_REQUEST,
    
    # Billing
    "credits": EventType.SYSTEM_BILLING,
    "billing": EventType.SYSTEM_BILLING,
    "system.billing": EventType.SYSTEM_BILLING,
    
    # System
    "system.error": EventType.SYSTEM_ERROR,
    "system.ready": EventType.SYSTEM_READY,
    "system.info": EventType.SYSTEM_INFO,
}


def get_event_type(old_type: str) -> EventType:
    """Get new event type from old type name"""
    return EVENT_MAPPING.get(old_type, EventType.SYSTEM_INFO)


def is_streaming_event(event_type: EventType) -> bool:
    """Check if event type is for streaming content"""
    return event_type in [EventType.CONTENT_THINKING, EventType.CONTENT_TOKEN]


def is_system_event(event_type: EventType) -> bool:
    """Check if event is a system event"""
    return event_type.value.startswith("system.")


def is_content_event(event_type: EventType) -> bool:
    """Check if event is a content event"""
    return event_type.value.startswith("content.")