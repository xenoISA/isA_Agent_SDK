#!/usr/bin/env python3
"""
Common types and enums for SmartAgent v3.0
统一的通用类型定义，用于整个应用程序
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


# ==================== Enums ====================

class ExecutionStrategy(str, Enum):
    """执行策略枚举"""
    DIRECT = "direct"
    TOOL_CALL = "tool_call"
    AUTONOMOUS = "autonomous"
    AUTONOMOUS_PLANNING = "autonomous_planning"


class InteractionLevel(str, Enum):
    """用户交互级别"""
    PASSIVE = "passive"      # 用户只需观察
    ACTIVE = "active"        # 用户可以中断或调整
    REQUIRED = "required"    # 需要用户输入


class EventType(str, Enum):
    """事件类型枚举"""
    # 系统状态
    START = "start"
    END = "end"
    ERROR = "error"
    
    # 内容流
    TOKEN = "token"
    CONTENT = "content"
    PARTIAL_RESULT = "partial_result"
    
    # 会话和记忆
    SESSION_RESTORED = "session_restored"
    MEMORY_CONTEXT = "memory_context"
    MEMORY_REVISION = "memory_revision"
    
    # 任务管理
    TASK_UNDERSTANDING = "task_understanding"
    TASK_PLANNING = "task_planning"
    TASK_STEP_START = "task_step_start"
    TASK_STEP_COMPLETE = "task_step_complete"
    TASK_COMPLETE = "task_complete"
    
    # 自主模式
    AUTONOMOUS_PLANNING = "autonomous_planning"
    AUTONOMOUS_PLANNING_COMPLETE = "autonomous_planning_complete"
    AUTONOMOUS_PAUSE = "autonomous_pause"
    TASK_CREATED = "task_created"
    
    # 工具和资源
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    RESOURCE_LOADING = "resource_loading"
    
    # 节点执行
    NODE_EXECUTION = "node_execution"
    
    # AI推理过程
    AI_THINKING = "ai_thinking"
    DECISION_MAKING = "decision_making"
    CONTENT_GENERATION = "content_generation"
    
    # 用户交互
    CLARIFICATION_NEEDED = "clarification_needed"
    
    # 完成状态
    SESSION_COMPLETE = "session_complete"
    
    # 错误和异常
    USER_ERROR = "user_error"
    RECOVERY_SUGGESTION = "recovery_suggestion"
    
    # 计费和信用
    CREDITS = "credits"
    BILLING_INFO = "billing_info"


class ResponseMode(str, Enum):
    """响应模式"""
    NATIVE = "native"
    ENHANCED = "enhanced"
    DEBUG = "debug"


class GuardrailMode(str, Enum):
    """安全检查模式"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"
    CANCELLED = "cancelled"


class FileType(str, Enum):
    """文件类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    OTHER = "other"


# ==================== Base Models ====================

class BaseTimestamped(BaseModel):
    """带时间戳的基础模型"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BaseIdentified(BaseTimestamped):
    """带ID和时间戳的基础模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ==================== Data Models ====================

class FileInfo(BaseModel):
    """文件信息"""
    filename: str
    content_type: str
    size: int
    file_type: FileType
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('file_type', pre=True, always=True)
    def determine_file_type(cls, v, values):
        if v:
            return v
        
        content_type = values.get('content_type', '')
        if content_type.startswith('image/'):
            return FileType.IMAGE
        elif content_type.startswith('audio/'):
            return FileType.AUDIO
        elif content_type.startswith('video/'):
            return FileType.VIDEO
        elif content_type.startswith('text/'):
            return FileType.TEXT
        elif content_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return FileType.DOCUMENT
        else:
            return FileType.OTHER


class AudioInfo(BaseModel):
    """音频信息"""
    filename: str
    content_type: str
    size: int
    duration: Optional[float] = None
    transcription: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolInfo(BaseModel):
    """工具信息"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    can_interrupt: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    """任务信息"""
    id: Union[str, int]
    title: str
    description: str = ""
    tools: List[str] = Field(default_factory=list)
    dependencies: List[Union[str, int]] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: str = "medium"
    estimated_duration: str = "Unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionMemory(BaseModel):
    """会话记忆信息"""
    session_id: str
    conversation_summary: str = ""
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    ongoing_tasks: List[TaskInfo] = Field(default_factory=list)
    total_messages: int = 0
    messages_since_last_summary: int = 0
    last_summary_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BillingInfo(BaseModel):
    """计费信息"""
    credits_used: float = 0.0
    cost: float = 0.0
    currency: str = "USD"
    breakdown: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressInfo(BaseModel):
    """进度信息"""
    current: int = Field(ge=0, le=100)
    total: int = 100
    step: Optional[str] = None
    stage: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def percentage(self) -> float:
        return (self.current / self.total) * 100 if self.total > 0 else 0.0


# ==================== Utility Classes ====================

class MessageContent(BaseModel):
    """消息内容统一格式"""
    text: Optional[str] = None
    files: List[FileInfo] = Field(default_factory=list)
    audio: Optional[AudioInfo] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('text', pre=True)
    def validate_text(cls, v):
        if v is not None and not isinstance(v, str):
            return str(v)
        return v
    
    def has_content(self) -> bool:
        """检查是否有有效内容"""
        return bool(self.text or self.files or self.audio)


class CapabilitiesInfo(BaseModel):
    """能力信息"""
    tools: List[ToolInfo] = Field(default_factory=list)
    prompts: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    formatted_tools: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConfigInfo(BaseModel):
    """配置信息"""
    guardrail_enabled: bool = False
    guardrail_mode: GuardrailMode = GuardrailMode.MODERATE
    response_mode: ResponseMode = ResponseMode.NATIVE
    streaming_enabled: bool = True
    session_memory_enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== Validation Helpers ====================

def validate_session_id(session_id: Optional[str]) -> str:
    """验证并标准化会话ID"""
    if not session_id:
        return f"session_{uuid.uuid4().hex[:8]}"
    return session_id


def validate_user_id(user_id: Optional[str]) -> str:
    """验证并标准化用户ID"""
    if not user_id:
        return "anonymous"
    return user_id


def create_event_id(event_type: EventType) -> str:
    """创建事件ID"""
    prefix_map = {
        EventType.MEMORY_CONTEXT: "mem",
        EventType.TASK_PLANNING: "plan",
        EventType.TOOL_EXECUTION: "tool",
        EventType.PARTIAL_RESULT: "result",
        EventType.AI_THINKING: "think",
        EventType.DECISION_MAKING: "decide",
        EventType.AUTONOMOUS_PAUSE: "pause",
        EventType.TASK_CREATED: "task",
        EventType.NODE_EXECUTION: "node",
        EventType.AUTONOMOUS_PLANNING: "auto",
        EventType.AUTONOMOUS_PLANNING_COMPLETE: "auto_done",
        EventType.TOOL_RESULT: "tool_ok",
        EventType.TOOL_ERROR: "tool_err"
    }
    
    prefix = prefix_map.get(event_type, "evt")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return f"{prefix}_{timestamp}"


# ==================== Type Aliases ====================

# 常用的联合类型
ContentType = Union[str, MessageContent]
SessionIdType = Union[str, None]
UserIdType = Union[str, None]
MetadataType = Dict[str, Any]
ContextType = Dict[str, Any]

# 事件数据类型
EventDataType = Union[
    TaskInfo, ToolInfo, SessionMemory, BillingInfo,
    ProgressInfo, CapabilitiesInfo, Dict[str, Any]
]