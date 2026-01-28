#!/usr/bin/env python3
"""
Task Data Models - Pydantic models for background tasks

Defines task structure, status, and progress tracking
"""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolCallInfo(BaseModel):
    """Individual tool call information"""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str


class TaskDefinition(BaseModel):
    """Background task definition"""
    job_id: str
    session_id: str
    user_id: Optional[str] = None
    tools: List[ToolCallInfo]
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    priority: Literal["low", "normal", "high"] = "normal"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ToolExecutionResult(BaseModel):
    """Result of a single tool execution"""
    tool_name: str
    tool_call_id: str
    status: Literal["success", "error"]
    result: Optional[Any] = None
    error: Optional[str] = None
    result_length: int = 0
    execution_time_ms: float = 0


class TaskProgress(BaseModel):
    """Task execution progress"""
    job_id: str
    status: TaskStatus
    total_tools: int
    completed_tools: int = 0
    failed_tools: int = 0
    progress_percent: float = 0.0
    current_tool: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskResult(BaseModel):
    """Final task execution result"""
    job_id: str
    session_id: str
    status: TaskStatus
    total_tools: int
    successful_tools: int
    failed_tools: int
    results: List[ToolExecutionResult]
    started_at: datetime
    completed_at: datetime
    execution_time_seconds: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProgressEvent(BaseModel):
    """Progress event for streaming updates"""
    type: Literal["task_start", "tool_start", "tool_complete", "tool_error", "task_complete", "task_error"]
    job_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
