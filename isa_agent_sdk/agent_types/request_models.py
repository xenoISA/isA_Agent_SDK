#!/usr/bin/env python3
"""
Request Models for SmartAgent v3.0 API
统一的请求模型定义，确保API接口的一致性和类型安全
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from fastapi import UploadFile

from .common_types import (
    BaseTimestamped, MessageContent, FileInfo, AudioInfo, 
    ExecutionStrategy, ResponseMode, GuardrailMode,
    validate_session_id, validate_user_id, MetadataType
)


# ==================== Base Request Models ====================

class BaseRequest(BaseTimestamped):
    """统一的请求基类"""
    user_id: str = Field(default="anonymous", description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    metadata: MetadataType = Field(default_factory=dict, description="元数据")
    
    @validator('user_id', pre=True)
    def validate_user_id_field(cls, v):
        return validate_user_id(v)
    
    @validator('session_id', pre=True)
    def validate_session_id_field(cls, v):
        return validate_session_id(v) if v else None


class BaseConfigurableRequest(BaseRequest):
    """可配置的请求基类"""
    response_mode: ResponseMode = Field(default=ResponseMode.NATIVE, description="响应模式")
    guardrail_enabled: bool = Field(default=False, description="是否启用安全检查")
    guardrail_mode: GuardrailMode = Field(default=GuardrailMode.MODERATE, description="安全检查模式")
    streaming: bool = Field(default=True, description="是否启用流式响应")


# ==================== Chat Request Models ====================

class ChatRequest(BaseConfigurableRequest):
    """聊天请求模型"""
    message: Union[str, MessageContent] = Field(..., description="消息内容")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文信息")
    execution_strategy: Optional[ExecutionStrategy] = Field(default=None, description="执行策略")
    
    @validator('message', pre=True)
    def validate_message(cls, v):
        if isinstance(v, str):
            return MessageContent(text=v)
        elif isinstance(v, dict):
            return MessageContent(**v)
        elif isinstance(v, MessageContent):
            return v
        else:
            return MessageContent(text=str(v))
    
    def has_multimodal_content(self) -> bool:
        """检查是否包含多模态内容"""
        if isinstance(self.message, MessageContent):
            return bool(self.message.files or self.message.audio)
        return False


class MultimodalChatRequest(BaseConfigurableRequest):
    """多模态聊天请求模型"""
    text: Optional[str] = Field(default=None, description="文本消息")
    files: List[FileInfo] = Field(default_factory=list, description="文件信息列表")
    audio: Optional[AudioInfo] = Field(default=None, description="音频信息")
    context: Optional[Dict[str, Any]] = Field(default=None, description="上下文信息")
    
    @validator('files', pre=True)
    def validate_files(cls, v):
        if not v:
            return []
        
        result = []
        for item in v:
            if isinstance(item, FileInfo):
                result.append(item)
            elif isinstance(item, dict):
                result.append(FileInfo(**item))
            # 忽略其他类型
        return result
    
    def to_message_content(self) -> MessageContent:
        """转换为MessageContent格式"""
        return MessageContent(
            text=self.text,
            files=self.files,
            audio=self.audio,
            metadata=self.metadata
        )
    
    def has_content(self) -> bool:
        """检查是否有有效内容"""
        return bool(self.text or self.files or self.audio)


class StreamingChatRequest(MultimodalChatRequest):
    """流式聊天请求模型（表单数据格式）"""
    # 这个模型主要用于处理表单数据，实际字段通过表单解析填充
    
    @classmethod
    def from_form_data(
        cls,
        message: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        audio_file: Optional[UploadFile] = None,
        files: List[UploadFile] = None,
        **kwargs
    ) -> 'StreamingChatRequest':
        """从表单数据创建请求"""
        import json
        
        # 解析消息数据
        text_input = ""
        metadata = {}
        
        if message:
            try:
                # 尝试解析JSON格式的消息
                message_data = json.loads(message)
                text_input = message_data.get('text', '') or message_data.get('message', '')
                metadata = message_data
            except (json.JSONDecodeError, TypeError):
                # 如果不是JSON，直接使用字符串
                text_input = message
                metadata = {}
        
        # 处理文件信息
        file_infos = []
        if files:
            for file in files:
                if file and file.filename:
                    file_infos.append(FileInfo(
                        filename=file.filename,
                        content_type=file.content_type or "application/octet-stream",
                        size=0  # 实际大小需要在处理时计算
                    ))
        
        # 处理音频信息
        audio_info = None
        if audio_file and audio_file.filename:
            audio_info = AudioInfo(
                filename=audio_file.filename,
                content_type=audio_file.content_type or "audio/mpeg",
                size=0  # 实际大小需要在处理时计算
            )
        
        return cls(
            text=text_input,
            files=file_infos,
            audio=audio_info,
            session_id=thread_id,
            user_id=user_id or "anonymous",
            metadata=metadata,
            **kwargs
        )


# ==================== Session Request Models ====================

class SessionRequest(BaseRequest):
    """会话请求模型"""
    action: str = Field(..., description="操作类型: create, get, update, delete, clear")
    
    @validator('action')
    def validate_action(cls, v):
        allowed_actions = ['create', 'get', 'update', 'delete', 'clear', 'list']
        if v not in allowed_actions:
            raise ValueError(f"Action must be one of {allowed_actions}")
        return v


class SessionHistoryRequest(BaseRequest):
    """会话历史请求模型"""
    limit: int = Field(default=20, ge=1, le=100, description="返回记录数量")
    offset: int = Field(default=0, ge=0, description="偏移量")
    include_summary: bool = Field(default=True, description="是否包含摘要")


class SessionClearRequest(BaseRequest):
    """会话清理请求模型"""
    clear_memory: bool = Field(default=True, description="是否清理会话记忆")
    clear_messages: bool = Field(default=True, description="是否清理消息历史")


# ==================== Configuration Request Models ====================

class ConfigRequest(BaseRequest):
    """配置请求模型"""
    guardrail_enabled: Optional[bool] = Field(default=None, description="是否启用安全检查")
    guardrail_mode: Optional[GuardrailMode] = Field(default=None, description="安全检查模式")
    response_mode: Optional[ResponseMode] = Field(default=None, description="响应模式")
    settings: Optional[Dict[str, Any]] = Field(default=None, description="其他设置")


class CapabilitiesRequest(BaseRequest):
    """能力查询请求模型"""
    include_tools: bool = Field(default=True, description="是否包含工具信息")
    include_prompts: bool = Field(default=True, description="是否包含提示信息")
    include_resources: bool = Field(default=True, description="是否包含资源信息")


# ==================== Health and Status Request Models ====================

class HealthCheckRequest(BaseModel):
    """健康检查请求模型"""
    include_details: bool = Field(default=False, description="是否包含详细信息")


class StatusRequest(BaseRequest):
    """状态查询请求模型"""
    include_session_info: bool = Field(default=True, description="是否包含会话信息")
    include_memory_stats: bool = Field(default=False, description="是否包含内存统计")
    include_performance_metrics: bool = Field(default=False, description="是否包含性能指标")


# ==================== Tool and Resource Request Models ====================

class ToolExecutionRequest(BaseRequest):
    """工具执行请求模型"""
    tool_name: str = Field(..., description="工具名称")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    async_execution: bool = Field(default=False, description="是否异步执行")


class ResourceSearchRequest(BaseRequest):
    """资源搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    resource_types: List[str] = Field(default_factory=list, description="资源类型过滤")
    limit: int = Field(default=10, ge=1, le=50, description="返回结果数量")


# ==================== Billing Request Models ====================

class BillingRequest(BaseRequest):
    """计费请求模型"""
    start_date: Optional[datetime] = Field(default=None, description="开始日期")
    end_date: Optional[datetime] = Field(default=None, description="结束日期")
    include_details: bool = Field(default=False, description="是否包含详细信息")


class UsageRequest(BaseRequest):
    """使用量查询请求模型"""
    period: str = Field(default="daily", description="统计周期: daily, weekly, monthly")
    limit: int = Field(default=30, ge=1, le=365, description="返回天数")


# ==================== Validation and Utility Functions ====================

def validate_chat_request(request: ChatRequest) -> ChatRequest:
    """验证并标准化聊天请求"""
    # 确保消息有内容
    if isinstance(request.message, MessageContent):
        if not request.message.has_content():
            raise ValueError("Message must contain text, files, or audio")
    
    return request


def validate_multimodal_request(request: MultimodalChatRequest) -> MultimodalChatRequest:
    """验证并标准化多模态请求"""
    if not request.has_content():
        raise ValueError("Request must contain text, files, or audio")
    
    return request


# ==================== Request Factory ====================

class RequestFactory:
    """请求工厂类"""
    
    @staticmethod
    def create_chat_request(
        message: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        **kwargs
    ) -> ChatRequest:
        """创建聊天请求"""
        return ChatRequest(
            message=message,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )
    
    @staticmethod
    def create_multimodal_request(
        text: Optional[str] = None,
        files: List[FileInfo] = None,
        audio: Optional[AudioInfo] = None,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        **kwargs
    ) -> MultimodalChatRequest:
        """创建多模态请求"""
        return MultimodalChatRequest(
            text=text,
            files=files or [],
            audio=audio,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )
    
    @staticmethod
    def create_session_request(
        action: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        **kwargs
    ) -> SessionRequest:
        """创建会话请求"""
        return SessionRequest(
            action=action,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )