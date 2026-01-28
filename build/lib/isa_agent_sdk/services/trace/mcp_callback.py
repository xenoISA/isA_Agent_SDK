#!/usr/bin/env python3
"""
MCP Callback - Capture MCP operation data and persist to database
统一的 MCP 操作追踪，自动识别操作类型，写入数据库
"""
import time
import uuid
import json
from typing import Dict, Any, Optional
from functools import wraps

from isa_agent_sdk.utils.logger import agent_logger
from .trace_writer import get_trace_writer

logger = agent_logger


def log_mcp_call(
    span_id: str,
    session_id: str,
    node_name: str,
    operation_type: str,
    tool_name: Optional[str] = None,
    tool_arguments: Optional[Dict] = None,
    prompt_name: Optional[str] = None,
    prompt_arguments: Optional[Dict] = None,
    resource_uri: Optional[str] = None
):
    """记录 MCP 调用（开始） + 写入数据库"""
    try:
        logger.info(f"""
╔════════════════════════════════════════════════════════════════
║ [MCP_CALL_START] Span: {span_id[:8]}
╠════════════════════════════════════════════════════════════════
║ Session: {session_id}
║ Node: {node_name}
║ Operation: {operation_type}
""")

        if tool_name:
            logger.info(f"║ Tool: {tool_name}")
            if tool_arguments:
                logger.info(f"║ Arguments: {json.dumps(tool_arguments, indent=2)}")

        if prompt_name:
            logger.info(f"║ Prompt: {prompt_name}")
            if prompt_arguments:
                logger.info(f"║ Arguments: {json.dumps(prompt_arguments, indent=2)}")

        if resource_uri:
            logger.info(f"║ Resource URI: {resource_uri}")

        logger.info("╚════════════════════════════════════════════════════════════════")

        # Write to database
        writer = get_trace_writer()

        # Build input data based on operation type
        input_data = {'operation_type': operation_type}
        if tool_name:
            input_data['tool_name'] = tool_name
            if tool_arguments:
                input_data['arguments'] = tool_arguments
        if prompt_name:
            input_data['prompt_name'] = prompt_name
            if prompt_arguments:
                input_data['arguments'] = prompt_arguments
        if resource_uri:
            input_data['resource_uri'] = resource_uri

        # Name for the span
        if tool_name:
            span_name = f"{node_name}.mcp_call_tool({tool_name})"
        elif prompt_name:
            span_name = f"{node_name}.mcp_get_prompt({prompt_name})"
        elif resource_uri:
            span_name = f"{node_name}.mcp_get_resource"
        else:
            span_name = f"{node_name}.mcp_{operation_type}"

        writer.write_span_start(
            span_id=span_id,
            trace_id=session_id,
            session_id=session_id,
            span_type='mcp_call',
            name=span_name,
            input_data=input_data
        )

    except Exception as e:
        logger.error(f"Failed to log MCP call: {e}")


def log_mcp_response(
    span_id: str,
    output_content: str,
    duration_ms: int,
    status: str = 'success',
    error_message: Optional[str] = None
):
    """记录 MCP 响应（完成） + 写入数据库"""
    try:
        logger.info(f"""
╔════════════════════════════════════════════════════════════════
║ [MCP_CALL_END] Span: {span_id[:8]}
╠════════════════════════════════════════════════════════════════
║ Duration: {duration_ms}ms
║ Status: {status}
║ Output Length: {len(output_content)} chars
""")

        if error_message:
            logger.info(f"║ Error: {error_message}")

        if output_content:
            preview = output_content[:200]
            logger.info(f"║ Output Preview:")
            logger.info(f"║   {preview}...")

        logger.info("╚════════════════════════════════════════════════════════════════")

        # Write to database
        writer = get_trace_writer()
        writer.write_span_end(
            span_id=span_id,
            duration_ms=duration_ms,
            output_data={'content': output_content, 'length': len(output_content)},
            status=status,
            error=error_message
        )

    except Exception as e:
        logger.error(f"Failed to log MCP response: {e}")


def trace_mcp_operation(node_name: str):
    """
    统一的 MCP 操作追踪装饰器
    自动从函数名识别操作类型：
    - mcp_call_tool → call_tool
    - mcp_get_prompt → get_prompt
    - mcp_get_resource → get_resource

    Usage:
        @trace_mcp_operation("ToolNode")
        async def mcp_call_tool(self, tool_name, arguments, config):
            ...

        @trace_mcp_operation("ReasonNode")
        async def mcp_get_prompt(self, prompt_name, arguments, config):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # 生成 span_id
            span_id = str(uuid.uuid4())

            # 获取 session_id from config
            session_id = 'unknown'
            if hasattr(self, 'config') and self.config:
                configurable = self.config.get('configurable', {})
                session_id = configurable.get('thread_id', session_id)

            # Fallback: try from self.session_id attribute
            if session_id == 'unknown':
                session_id = getattr(self, 'session_id', 'unknown')

            # 从函数名推断操作类型
            func_name = func.__name__
            if 'call_tool' in func_name:
                operation_type = 'call_tool'
                tool_name = args[0] if args else kwargs.get('tool_name')
                tool_arguments = args[1] if len(args) > 1 else kwargs.get('arguments', {})
                log_mcp_call(
                    span_id=span_id,
                    session_id=session_id,
                    node_name=node_name,
                    operation_type=operation_type,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments
                )
            elif 'get_prompt' in func_name:
                operation_type = 'get_prompt'
                prompt_name = args[0] if args else kwargs.get('prompt_name')
                prompt_arguments = args[1] if len(args) > 1 else kwargs.get('arguments', {})
                log_mcp_call(
                    span_id=span_id,
                    session_id=session_id,
                    node_name=node_name,
                    operation_type=operation_type,
                    prompt_name=prompt_name,
                    prompt_arguments=prompt_arguments
                )
            elif 'get_resource' in func_name:
                operation_type = 'get_resource'
                resource_uri = args[0] if args else kwargs.get('uri')
                log_mcp_call(
                    span_id=span_id,
                    session_id=session_id,
                    node_name=node_name,
                    operation_type=operation_type,
                    resource_uri=resource_uri
                )
            else:
                operation_type = 'unknown'
                log_mcp_call(
                    span_id=span_id,
                    session_id=session_id,
                    node_name=node_name,
                    operation_type=operation_type
                )

            # 执行原函数
            start_time = time.time()
            try:
                result = await func(self, *args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                # 记录响应
                if result is not None:
                    if isinstance(result, dict):
                        output_content = json.dumps(result)
                    else:
                        output_content = str(result)
                else:
                    output_content = ''

                log_mcp_response(
                    span_id=span_id,
                    output_content=output_content,
                    duration_ms=duration_ms,
                    status='success'
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                # 记录错误
                log_mcp_response(
                    span_id=span_id,
                    output_content='',
                    duration_ms=duration_ms,
                    status='error',
                    error_message=str(e)
                )
                raise

        return wrapper
    return decorator
