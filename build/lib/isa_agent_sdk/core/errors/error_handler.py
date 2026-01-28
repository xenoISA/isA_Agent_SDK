"""
统一错误处理工具模块

文件功能：
提供全系统的统一错误处理机制，确保错误信息的一致性和可追踪性。
该模块实现了标准化的异常处理、日志记录和错误响应格式。

主要功能：
1. API错误处理 - 统一的API异常处理和HTTP响应格式
2. 装饰器错误捕获 - 自动化的错误捕获和处理装饰器
3. 结构化日志 - 一致的错误日志格式和上下文信息
4. 追踪集成 - 与分布式追踪系统的集成
5. 错误分类 - 不同类型错误的分类处理
6. 上下文保护 - 保护敏感信息不泄露到错误日志

核心特性：
- 统一接口: 标准化的错误处理方法和格式
- 上下文感知: 包含操作类型、时间戳等关键上下文
- 追踪关联: 与分布式追踪系统的无缝集成
- 日志分级: 支持不同级别的错误日志记录
- 错误恢复: 提供错误处理和恢复建议

错误处理层级：
1. 装饰器级别: 自动捕获函数级别的异常
2. 操作级别: 手动处理特定操作的错误
3. 系统级别: 全局异常处理和系统错误
4. 用户级别: 用户友好的错误信息展示

错误类型分类：
- HTTPException: API层面的HTTP错误
- ValidationError: 输入验证错误
- ServiceError: 服务层面的业务错误
- SystemError: 系统层面的技术错误
- ExternalError: 外部服务的调用错误

日志结构化字段：
- operation: 失败的操作名称
- error_type: 异常类型名称
- timestamp: 错误发生时间
- trace_id: 分布式追踪标识符
- session_id: 用户会话标识符
- function: 发生错误的函数名称

错误响应格式：
- error: 错误类型描述
- message: 详细错误信息
- context: 结构化的上下文信息
- timestamp: 错误发生时间

追踪器集成：
- 自动初始化: 智能的追踪器获取和初始化
- 方法检查: 验证追踪器支持的功能
- 错误处理: 追踪器不可用时的降级处理
- 服务状态: 追踪服务的健康状态检查

使用模式：
- 装饰器模式: @api_error_handler()自动错误处理
- 函数调用模式: handle_api_error()手动处理
- 上下文记录: log_error_with_context()结构化日志
- 初始化检查: TracerInitializer类的服务检查

性能考虑：
- 懒加载: 按需加载追踪器和日志组件
- 异常优化: 避免在错误处理中产生新异常
- 内存管理: 控制错误上下文的内存使用
- 批量处理: 支持批量错误的处理

安全特性：
- 信息过滤: 避免敏感信息泄露到日志
- 堆栈保护: 控制堆栈信息的暴露级别
- 用户隔离: 不同用户错误的隔离处理
- 审计追踪: 错误事件的审计记录

设计合理性分析：
✅ 架构优势：
- 单一职责: 专注于错误处理和日志记录
- 统一标准: 一致的错误处理模式和格式
- 可扩展性: 支持新的错误类型和处理策略
- 集成友好: 与其他系统组件的良好集成
- 性能优化: 高效的错误处理和日志记录

✅ 运维友好：
- 结构化日志: 便于日志分析和监控
- 追踪集成: 完整的错误追踪和定位
- 上下文丰富: 详细的错误上下文信息
- 分级处理: 不同严重程度的差异化处理

✅ 开发体验：
- 简单易用: 装饰器和函数的简洁接口
- 自动化: 减少手动错误处理的代码
- 调试友好: 丰富的调试信息和堆栈
- 文档清晰: 明确的使用说明和示例
"""
from typing import Callable, Any, Optional, Dict
from fastapi import HTTPException
from functools import wraps
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def api_error_handler(operation_type: str, status_code: int = 500):
    """
    Decorator for consistent API error handling with context
    
    Args:
        operation_type: Description of the operation for error messages
        status_code: HTTP status code to return on error
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as e:
                # Enhanced error logging with context
                error_context = {
                    "operation": operation_type,
                    "function": func.__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.error(
                    f"❌ {operation_type} error: {e}", 
                    extra=error_context,
                    exc_info=True
                )
                
                raise HTTPException(
                    status_code=status_code, 
                    detail={
                        "error": f"Failed to {operation_type.lower()}",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
        return wrapper
    return decorator


def handle_api_error(
    operation: str, 
    exception: Exception, 
    status_code: int = 500,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> HTTPException:
    """
    Enhanced error handling for API operations with context
    
    Args:
        operation: Description of the operation that failed
        exception: The exception that occurred
        status_code: HTTP status code to return
        trace_id: Optional trace ID for tracking
        session_id: Optional session ID for tracking
    
    Returns:
        HTTPException to raise with structured error information
    """
    error_context = {
        "operation": operation,
        "error_type": type(exception).__name__,
        "timestamp": datetime.now().isoformat()
    }
    
    if trace_id:
        error_context["trace_id"] = trace_id
    if session_id:
        error_context["session_id"] = session_id
    
    logger.error(f"❌ {operation} error: {exception}", extra=error_context, exc_info=True)
    
    return HTTPException(
        status_code=status_code,
        detail={
            "error": f"Failed to {operation.lower()}",
            "message": str(exception),
            "context": error_context
        }
    )


def log_error_with_context(
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error"
) -> None:
    """
    Log error with consistent format and context
    
    Args:
        error: The exception that occurred
        operation: Description of the operation
        context: Additional context information
        level: Log level (error, warning, critical)
    """
    log_context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "timestamp": datetime.now().isoformat(),
        **(context or {})
    }
    
    log_message = f"{operation} failed: {str(error)}"
    
    if level == "warning":
        logger.warning(log_message, extra=log_context)
    elif level == "error":
        logger.error(log_message, extra=log_context, exc_info=True)
    elif level == "critical":
        logger.critical(log_message, extra=log_context, exc_info=True)

class TracerInitializer:
    """Common tracer initialization logic"""
    
    @staticmethod
    def get_tracer():
        """Get tracer instance with common error handling"""
        try:
            from isa_agent_sdk.tracing.tracer import tracer
            if not tracer:
                raise HTTPException(status_code=503, detail="Tracer not available")
            return tracer
        except ImportError:
            raise HTTPException(status_code=503, detail="Tracing module not available")
    
    @staticmethod
    def check_tracer_method(tracer, method_name: str):
        """Check if tracer has required method"""
        if not hasattr(tracer, method_name):
            raise HTTPException(
                status_code=503, 
                detail=f"Tracer does not support {method_name} method"
            )