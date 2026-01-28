"""
Unified Type Definitions for SmartAgent v3.0
统一的类型定义，确保整个应用程序的类型安全和一致性
"""

# ==================== Internal State Types ====================
from .agent_state import AgentState

# ==================== Common Types ====================
from .common_types import (
    # Enums
    ExecutionStrategy, InteractionLevel, EventType, ResponseMode,
    GuardrailMode, TaskStatus, FileType,
    
    # Base Models
    BaseTimestamped, BaseIdentified,
    
    # Data Models  
    FileInfo, AudioInfo, ToolInfo, TaskInfo, SessionMemory,
    BillingInfo, ProgressInfo, MessageContent, CapabilitiesInfo,
    ConfigInfo,
    
    # Validation Functions
    validate_session_id, validate_user_id, create_event_id,
    
    # Type Aliases
    ContentType, SessionIdType, UserIdType, MetadataType,
    ContextType, EventDataType
)

# ==================== Request Models ====================
from .request_models import (
    # Base Request Models
    BaseRequest, BaseConfigurableRequest,
    
    # Chat Request Models
    ChatRequest, MultimodalChatRequest, StreamingChatRequest,
    
    # Session Request Models
    SessionRequest, SessionHistoryRequest, SessionClearRequest,
    
    # Configuration Request Models
    ConfigRequest, CapabilitiesRequest,
    
    # Status Request Models
    HealthCheckRequest, StatusRequest,
    
    # Tool and Resource Request Models
    ToolExecutionRequest, ResourceSearchRequest,
    
    # Billing Request Models
    BillingRequest, UsageRequest,
    
    # Validation Functions
    validate_chat_request, validate_multimodal_request,
    
    # Factory
    RequestFactory
)

# ==================== Response Models ====================
from .response_models import (
    # Base Response Models
    BaseResponse, ErrorResponse, DataResponse,
    
    # Chat Response Models
    ChatResponse, StreamingEvent,
    
    # Specific Event Models
    TokenEvent, NodeExecutionEvent, ToolExecutionEvent,
    ToolResultEvent, TaskCreatedEvent, AutonomousPlanningEvent,
    AutonomousPlanningCompleteEvent,
    
    # Session Response Models
    SessionResponse, SessionHistoryResponse,
    
    # Configuration Response Models
    CapabilitiesResponse, ConfigResponse, HealthResponse,
    
    # Billing Response Models
    BillingResponse,
    
    # Factories
    EventFactory, ResponseFactory
)

# ==================== Event Types ====================
from .event_types import (
    EventCategory, EventType as EventTypeEnum, EventData, EventEmitter,
    get_event_type, is_streaming_event, is_system_event, is_content_event
)

# ==================== Legacy Agentic Events (Backward Compatibility) ====================
# Note: agentic_events.py has been removed as it was redundant with event_types.py
# All event functionality is now provided by event_types.py and response_models.py

# ==================== Export All Public Types ====================

__all__ = [
    # ==================== Internal State ====================
    "AgentState",
    
    # ==================== Common Types ====================
    # Enums
    "ExecutionStrategy", "InteractionLevel", "EventType", "ResponseMode",
    "GuardrailMode", "TaskStatus", "FileType",
    
    # Base Models
    "BaseTimestamped", "BaseIdentified",
    
    # Data Models
    "FileInfo", "AudioInfo", "ToolInfo", "TaskInfo", "SessionMemory",
    "BillingInfo", "ProgressInfo", "MessageContent", "CapabilitiesInfo",
    "ConfigInfo",
    
    # Validation Functions
    "validate_session_id", "validate_user_id", "create_event_id",
    
    # Type Aliases
    "ContentType", "SessionIdType", "UserIdType", "MetadataType",
    "ContextType", "EventDataType",
    
    # ==================== Request Models ====================
    # Base Request Models
    "BaseRequest", "BaseConfigurableRequest",
    
    # Chat Request Models
    "ChatRequest", "MultimodalChatRequest", "StreamingChatRequest",
    
    # Session Request Models
    "SessionRequest", "SessionHistoryRequest", "SessionClearRequest",
    
    # Configuration Request Models
    "ConfigRequest", "CapabilitiesRequest",
    
    # Status Request Models
    "HealthCheckRequest", "StatusRequest",
    
    # Tool and Resource Request Models
    "ToolExecutionRequest", "ResourceSearchRequest",
    
    # Billing Request Models
    "BillingRequest", "UsageRequest",
    
    # Validation Functions
    "validate_chat_request", "validate_multimodal_request",
    
    # Factory
    "RequestFactory",
    
    # ==================== Response Models ====================
    # Base Response Models
    "BaseResponse", "ErrorResponse", "DataResponse",
    
    # Chat Response Models
    "ChatResponse", "StreamingEvent",
    
    # Specific Event Models
    "TokenEvent", "NodeExecutionEvent", "ToolExecutionEvent",
    "ToolResultEvent", "TaskCreatedEvent", "AutonomousPlanningEvent",
    "AutonomousPlanningCompleteEvent",
    
    # Session Response Models
    "SessionResponse", "SessionHistoryResponse",
    
    # Configuration Response Models
    "CapabilitiesResponse", "ConfigResponse", "HealthResponse",
    
    # Billing Response Models
    "BillingResponse",
    
    # Factories
    "EventFactory", "ResponseFactory",
    
    # ==================== Event Types ====================
    # Event system (from event_types.py)
    "EventCategory", "EventTypeEnum", "EventData", "EventEmitter",
    "get_event_type", "is_streaming_event", "is_system_event", "is_content_event"
]

# ==================== Version Information ====================
__version__ = "3.0.0"
__author__ = "SmartAgent Team"
__description__ = "Unified type definitions for SmartAgent v3.0 with session memory and MCP integration"