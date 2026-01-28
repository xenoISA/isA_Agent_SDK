#!/usr/bin/env python3
"""
Response Models for SmartAgent v3.0 API
统一的响应模型定义，确保API响应的一致性和类型安全
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import Field

from .common_types import (
    BaseTimestamped, EventType, InteractionLevel,
    TaskInfo, SessionMemory, BillingInfo,
    CapabilitiesInfo, ConfigInfo, MetadataType, ContextType
)


# ==================== Base Response Models ====================

class BaseResponse(BaseTimestamped):
    """统一的响应基类"""
    success: bool = Field(default=True, description="请求是否成功")
    message: Optional[str] = Field(default=None, description="响应消息")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    trace_id: Optional[str] = Field(default=None, description="追踪ID")
    metadata: MetadataType = Field(default_factory=dict, description="元数据")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = Field(default=False, description="请求失败")
    error_code: Optional[str] = Field(default=None, description="错误代码")
    error_details: Optional[str] = Field(default=None, description="错误详情")
    suggestions: List[str] = Field(default_factory=list, description="解决建议")
    
    @classmethod
    def create(
        cls,
        message: str,
        error_code: Optional[str] = None,
        error_details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        **kwargs
    ) -> 'ErrorResponse':
        """创建错误响应"""
        return cls(
            message=message,
            error_code=error_code,
            error_details=error_details,
            suggestions=suggestions or [],
            **kwargs
        )


class DataResponse(BaseResponse):
    """数据响应模型"""
    data: Any = Field(..., description="响应数据")
    
    @classmethod
    def create(cls, data: Any, message: Optional[str] = None, **kwargs) -> 'DataResponse':
        """创建数据响应"""
        return cls(data=data, message=message, **kwargs)


# ==================== Chat Response Models ====================

class ChatResponse(BaseResponse):
    """聊天响应模型"""
    content: str = Field(..., description="响应内容")
    billing: Optional[BillingInfo] = Field(default=None, description="计费信息")
    context: ContextType = Field(default_factory=dict, description="上下文信息")
    execution_strategy: Optional[str] = Field(default=None, description="执行策略")
    
    # Backward compatibility fields
    response: Optional[str] = Field(default=None, description="响应内容（兼容字段）")
    thread_id: Optional[str] = Field(default=None, description="线程ID（兼容字段）")
    credits_used: float = Field(default=0.0, description="使用的积分")
    
    @classmethod
    def create(
        cls,
        content: str,
        session_id: str,
        billing: Optional[BillingInfo] = None,
        **kwargs
    ) -> 'ChatResponse':
        """创建聊天响应"""
        return cls(
            content=content,
            response=content,  # Backward compatibility
            session_id=session_id,
            thread_id=session_id,  # Backward compatibility
            billing=billing,
            **kwargs
        )


class StreamingEvent(BaseTimestamped):
    """流式事件模型"""
    event_id: str = Field(..., description="事件ID")
    event_type: EventType = Field(..., description="事件类型")
    session_id: str = Field(..., description="会话ID")
    content: str = Field(default="", description="事件内容")
    
    # 进度和状态
    stage: Optional[str] = Field(default=None, description="当前阶段")
    progress: Optional[int] = Field(default=None, ge=0, le=100, description="进度百分比")
    
    # 交互控制
    interaction_level: InteractionLevel = Field(default=InteractionLevel.PASSIVE, description="交互级别")
    user_can_interrupt: bool = Field(default=False, description="用户是否可以中断")
    requires_response: bool = Field(default=False, description="是否需要用户响应")
    
    # 数据载荷
    data: Optional[Any] = Field(default=None, description="事件数据")
    context: ContextType = Field(default_factory=dict, description="上下文信息")
    metadata: MetadataType = Field(default_factory=dict, description="元数据")
    
    # 业务逻辑
    next_actions: List[str] = Field(default_factory=list, description="后续动作")
    alternatives: List[str] = Field(default_factory=list, description="替代选项")
    
    def to_sse_format(self) -> str:
        """转换为Server-Sent Events格式"""
        import json
        data_dict = self.model_dump()
        data_dict['event_type'] = self.event_type.value
        data_dict['interaction_level'] = self.interaction_level.value
        data_dict['timestamp'] = self.timestamp.isoformat()
        return f"data: {json.dumps(data_dict, ensure_ascii=False)}\n\n"


# ==================== Specific Event Models ====================

class TokenEvent(StreamingEvent):
    """Token事件模型"""
    event_type: EventType = EventType.TOKEN
    
    @classmethod
    def create(cls, session_id: str, content: str, **kwargs) -> 'TokenEvent':
        from .common_types import create_event_id
        return cls(
            event_id=create_event_id(EventType.TOKEN),
            session_id=session_id,
            content=content,
            **kwargs
        )


class NodeExecutionEvent(StreamingEvent):
    """节点执行事件模型"""
    event_type: EventType = EventType.NODE_EXECUTION
    node_name: Optional[str] = Field(default=None, description="节点名称")
    
    @classmethod
    def create(
        cls,
        session_id: str,
        content: str,
        node_name: str,
        progress: Optional[int] = None,
        **kwargs
    ) -> 'NodeExecutionEvent':
        from .common_types import create_event_id
        return cls(
            event_id=create_event_id(EventType.NODE_EXECUTION),
            session_id=session_id,
            content=content,
            node_name=node_name,
            progress=progress,
            metadata={"node_name": node_name, "progress": progress or 0},
            **kwargs
        )


class ToolExecutionEvent(StreamingEvent):
    """工具执行事件模型"""
    event_type: EventType = EventType.TOOL_EXECUTION
    tool_name: Optional[str] = Field(default=None, description="工具名称")
    
    @classmethod
    def create(
        cls,
        session_id: str,
        content: str,
        tool_info: Dict[str, Any],
        **kwargs
    ) -> 'ToolExecutionEvent':
        from .common_types import create_event_id
        return cls(
            event_id=create_event_id(EventType.TOOL_EXECUTION),
            session_id=session_id,
            content=content,
            tool_name=tool_info.get("name"),
            context={"tool_info": tool_info},
            user_can_interrupt=tool_info.get("can_interrupt", True),
            **kwargs
        )


class ToolResultEvent(StreamingEvent):
    """工具结果事件模型"""
    event_type: EventType = EventType.TOOL_RESULT
    tool_name: Optional[str] = Field(default=None, description="工具名称")
    
    @classmethod
    def create(
        cls,
        session_id: str,
        content: str,
        tool_name: str,
        result_preview: str = "",
        billing_cost: float = 0.0,
        **kwargs
    ) -> 'ToolResultEvent':
        from .common_types import create_event_id
        return cls(
            event_id=create_event_id(EventType.TOOL_RESULT),
            session_id=session_id,
            content=content,
            tool_name=tool_name,
            metadata={
                "tool_name": tool_name,
                "result_preview": result_preview,
                "billing_cost": billing_cost
            },
            **kwargs
        )


class TaskCreatedEvent(StreamingEvent):
    """任务创建事件模型"""
    event_type: EventType = EventType.TASK_CREATED
    task_id: Optional[str] = Field(default=None, description="任务ID")
    task_title: Optional[str] = Field(default=None, description="任务标题")
    
    @classmethod
    def create(
        cls,
        session_id: str,
        content: str,
        task_id: str,
        task_title: str,
        task_info: Optional[TaskInfo] = None,
        **kwargs
    ) -> 'TaskCreatedEvent':
        from .common_types import create_event_id
        metadata = {
            "task_id": task_id,
            "task_title": task_title
        }
        
        if task_info:
            additional_metadata = {
                "dependencies": task_info.dependencies,
                "estimated_duration": task_info.estimated_duration,
                "tools_required": task_info.tools,
                "priority": task_info.priority,
                "status": task_info.status.value if hasattr(task_info.status, 'value') else str(task_info.status)
            }
            metadata.update(additional_metadata)
        
        return cls(
            event_id=create_event_id(EventType.TASK_CREATED),
            session_id=session_id,
            content=content,
            task_id=task_id,
            task_title=task_title,
            metadata=metadata,
            **kwargs
        )


class AutonomousPlanningEvent(StreamingEvent):
    """自主规划事件模型"""
    event_type: EventType = EventType.AUTONOMOUS_PLANNING
    
    @classmethod
    def create(
        cls,
        session_id: str,
        content: str = "🎨 开始自主任务规划",
        strategy: str = "autonomous_planning",
        **kwargs
    ) -> 'AutonomousPlanningEvent':
        from .common_types import create_event_id
        return cls(
            event_id=create_event_id(EventType.AUTONOMOUS_PLANNING),
            session_id=session_id,
            content=content,
            stage="自主规划",
            metadata={
                "mode": "autonomous_planning",
                "strategy": strategy
            },
            **kwargs
        )


class AutonomousPlanningCompleteEvent(StreamingEvent):
    """自主规划完成事件模型"""
    event_type: EventType = EventType.AUTONOMOUS_PLANNING_COMPLETE
    
    @classmethod
    def create(
        cls,
        session_id: str,
        content: str = "✅ 自主任务规划完成",
        tool_name: str = "plan_autonomous_task",
        result_preview: str = "",
        billing_cost: float = 0.0,
        **kwargs
    ) -> 'AutonomousPlanningCompleteEvent':
        from .common_types import create_event_id
        return cls(
            event_id=create_event_id(EventType.AUTONOMOUS_PLANNING_COMPLETE),
            session_id=session_id,
            content=content,
            metadata={
                "tool_name": tool_name,
                "result_preview": result_preview,
                "billing_cost": billing_cost
            },
            **kwargs
        )


# ==================== Session Response Models ====================

class SessionResponse(BaseResponse):
    """会话响应模型"""
    session_info: Dict[str, Any] = Field(default_factory=dict, description="会话信息")
    
    @classmethod
    def create(cls, session_id: str, session_info: Dict[str, Any], **kwargs) -> 'SessionResponse':
        return cls(
            session_id=session_id,
            session_info=session_info,
            **kwargs
        )


class SessionHistoryResponse(BaseResponse):
    """会话历史响应模型"""
    history: Dict[str, Any] = Field(default_factory=dict, description="会话历史")
    session_memory: Optional[SessionMemory] = Field(default=None, description="会话记忆")
    recent_messages: List[Dict[str, Any]] = Field(default_factory=list, description="最近消息")
    
    @classmethod
    def create(
        cls,
        session_id: str,
        history: Dict[str, Any],
        **kwargs
    ) -> 'SessionHistoryResponse':
        return cls(
            session_id=session_id,
            history=history,
            session_memory=history.get("session_memory"),
            recent_messages=history.get("recent_messages", []),
            **kwargs
        )


# ==================== Configuration Response Models ====================

class CapabilitiesResponse(BaseResponse):
    """能力响应模型"""
    capabilities: CapabilitiesInfo = Field(..., description="系统能力")
    version: str = Field(default="3.0.0", description="版本信息")
    features: List[str] = Field(default_factory=list, description="功能列表")
    
    @classmethod
    def create(cls, capabilities: CapabilitiesInfo, **kwargs) -> 'CapabilitiesResponse':
        return cls(capabilities=capabilities, **kwargs)


class ConfigResponse(BaseResponse):
    """配置响应模型"""
    config: ConfigInfo = Field(..., description="配置信息")
    
    @classmethod
    def create(cls, config: ConfigInfo, **kwargs) -> 'ConfigResponse':
        return cls(config=config, **kwargs)


class HealthResponse(BaseResponse):
    """健康检查响应模型"""
    status: str = Field(default="healthy", description="健康状态")
    version: str = Field(default="3.0.0", description="版本信息")
    uptime: Optional[float] = Field(default=None, description="运行时间（秒）")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")
    
    @classmethod
    def create(cls, status: str = "healthy", **kwargs) -> 'HealthResponse':
        return cls(status=status, **kwargs)


# ==================== Billing Response Models ====================

class BillingResponse(BaseResponse):
    """计费响应模型"""
    billing: BillingInfo = Field(..., description="计费信息")
    usage_summary: Dict[str, Any] = Field(default_factory=dict, description="使用摘要")
    
    @classmethod
    def create(cls, billing: BillingInfo, **kwargs) -> 'BillingResponse':
        return cls(billing=billing, **kwargs)


# ==================== Event Factory ====================

class EventFactory:
    """事件工厂类"""
    
    @staticmethod
    def create_token_event(session_id: str, content: str) -> TokenEvent:
        return TokenEvent.create(session_id, content)
    
    @staticmethod
    def create_node_execution_event(
        session_id: str,
        content: str,
        node_name: str,
        progress: Optional[int] = None
    ) -> NodeExecutionEvent:
        return NodeExecutionEvent.create(session_id, content, node_name, progress)
    
    @staticmethod
    def create_tool_execution_event(
        session_id: str,
        content: str,
        tool_info: Dict[str, Any]
    ) -> ToolExecutionEvent:
        return ToolExecutionEvent.create(session_id, content, tool_info)
    
    @staticmethod
    def create_tool_result_event(
        session_id: str,
        content: str,
        tool_name: str,
        result_preview: str = "",
        billing_cost: float = 0.0
    ) -> ToolResultEvent:
        return ToolResultEvent.create(
            session_id, content, tool_name, result_preview, billing_cost
        )
    
    @staticmethod
    def create_task_created_event(
        session_id: str,
        content: str,
        task_id: str,
        task_title: str,
        task_info: Optional[TaskInfo] = None
    ) -> TaskCreatedEvent:
        return TaskCreatedEvent.create(session_id, content, task_id, task_title, task_info)
    
    @staticmethod
    def create_autonomous_planning_event(session_id: str) -> AutonomousPlanningEvent:
        return AutonomousPlanningEvent.create(session_id)
    
    @staticmethod
    def create_autonomous_planning_complete_event(
        session_id: str,
        tool_name: str = "plan_autonomous_task",
        result_preview: str = "",
        billing_cost: float = 0.0
    ) -> AutonomousPlanningCompleteEvent:
        return AutonomousPlanningCompleteEvent.create(
            session_id, tool_name=tool_name, result_preview=result_preview, billing_cost=billing_cost
        )
    
    @staticmethod
    def create_generic_event(
        event_type: EventType,
        session_id: str,
        content: str,
        **kwargs
    ) -> StreamingEvent:
        """创建通用事件"""
        from .common_types import create_event_id
        return StreamingEvent(
            event_id=create_event_id(event_type),
            event_type=event_type,
            session_id=session_id,
            content=content,
            **kwargs
        )


# ==================== Response Factory ====================

class ResponseFactory:
    """响应工厂类"""
    
    @staticmethod
    def create_success_response(
        message: str = "Success",
        data: Any = None,
        **kwargs
    ) -> Union[BaseResponse, DataResponse]:
        """创建成功响应"""
        if data is not None:
            return DataResponse.create(data, message, **kwargs)
        return BaseResponse(message=message, **kwargs)
    
    @staticmethod
    def create_error_response(
        message: str,
        error_code: Optional[str] = None,
        error_details: Optional[str] = None,
        **kwargs
    ) -> ErrorResponse:
        """创建错误响应"""
        return ErrorResponse.create(message, error_code, error_details, **kwargs)
    
    @staticmethod
    def create_chat_response(
        content: str,
        session_id: str,
        billing: Optional[BillingInfo] = None,
        **kwargs
    ) -> ChatResponse:
        """创建聊天响应"""
        return ChatResponse.create(content, session_id, billing, **kwargs)