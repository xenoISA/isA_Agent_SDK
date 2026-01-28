#!/usr/bin/env python3
"""
Model Callback - Capture model call data and persist to database
记录模型调用的输入和输出，写入数据库
"""
import time
import uuid
import json
from typing import Dict, List, Any, Optional
from functools import wraps

from isa_agent_sdk.utils.logger import agent_logger
from .trace_writer import get_trace_writer

logger = agent_logger


def serialize_message(msg) -> dict:
    """序列化 LangChain 消息"""
    msg_type = type(msg).__name__

    if hasattr(msg, 'content'):
        content = str(msg.content)
    elif isinstance(msg, dict):
        content = str(msg.get('content', ''))
    else:
        content = str(msg)

    result = {
        "type": msg_type,
        "content": content,
        "length": len(content)
    }

    # 额外字段
    if hasattr(msg, 'name'):
        result["name"] = msg.name
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.get("id"),
                "name": tc.get("name"),
                "args": tc.get("args")
            }
            for tc in msg.tool_calls
        ]

    return result


def log_model_call(
    span_id: str,
    session_id: str,
    node_name: str,
    model: str,
    provider: str,
    messages: List[Any],
    tools: Optional[List[Dict]] = None,
    stream_mode: bool = True,
    output_format: Optional[str] = None
):
    """
    记录模型调用开始
    Phase 2: 打印日志 + 写入数据库
    """
    logger.info(f"🔍 [LOG_MODEL_CALL_ENTRY] span={span_id[:8]}, session={session_id}")
    try:
        serialized_messages = [serialize_message(m) for m in messages]
        total_chars = sum(m.get('length', 0) for m in serialized_messages)

        logger.info(f"""
╔════════════════════════════════════════════════════════════════
║ [MODEL_CALL_START] Span: {span_id[:8]}
╠════════════════════════════════════════════════════════════════
║ Session: {session_id}
║ Node: {node_name}
║ Model: {model} ({provider})
║ Stream: {stream_mode} | Format: {output_format}
║ Messages: {len(messages)} ({total_chars} chars)
║ Tools: {len(tools) if tools else 0}
╠════════════════════════════════════════════════════════════════
║ Input Messages:
""")

        for i, msg in enumerate(serialized_messages):
            logger.info(f"║   [{i+1}] {msg['type']}: {msg['length']} chars")
            if msg.get('tool_calls'):
                logger.info(f"║       Tool calls: {len(msg['tool_calls'])}")

        logger.info("╚════════════════════════════════════════════════════════════════")

        # Write to database
        writer = get_trace_writer()
        writer.write_span_start(
            span_id=span_id,
            trace_id=session_id,
            session_id=session_id,
            span_type='model_call',
            name=f"{node_name}.call_model",
            input_data={
                'messages': serialized_messages,
                'tools_count': len(tools) if tools else 0,
                'stream_mode': stream_mode,
                'output_format': output_format
            },
            model=model,
            provider=provider,
            metadata={
                'total_chars': total_chars,
                'message_count': len(messages)
            }
        )
        logger.info(f"🔍 [LOG_MODEL_CALL_SUCCESS] DB write attempted for span {span_id[:8]}")

    except Exception as e:
        logger.error(f"❌ [LOG_MODEL_CALL_ERROR] Failed to log model call: {e}", exc_info=True)


def log_model_response(
    span_id: str,
    output_content: str,
    output_type: str,
    duration_ms: int,
    has_tool_calls: bool = False,
    tool_calls: Optional[List[Dict]] = None,
    tokens_used: Optional[Dict] = None,
    status: str = 'success',
    error_message: Optional[str] = None
):
    """
    记录模型响应完成
    Phase 2: 打印日志 + 写入数据库
    """
    try:
        logger.info(f"""
╔════════════════════════════════════════════════════════════════
║ [MODEL_CALL_END] Span: {span_id[:8]}
╠════════════════════════════════════════════════════════════════
║ Duration: {duration_ms}ms
║ Status: {status}
║ Output Type: {output_type}
║ Content Length: {len(output_content)} chars
║ Has Tool Calls: {has_tool_calls}
""")

        if tool_calls:
            logger.info(f"║ Tool Calls: {len(tool_calls)}")
            for tc in tool_calls:
                logger.info(f"║   - {tc.get('name')}")

        if tokens_used:
            logger.info(f"║ Tokens: {json.dumps(tokens_used)}")

        if error_message:
            logger.info(f"║ Error: {error_message}")

        if output_content:
            preview = output_content[:200]
            logger.info(f"║ Content Preview:")
            logger.info(f"║   {preview}...")

        logger.info("╚════════════════════════════════════════════════════════════════")

        # Write to database
        writer = get_trace_writer()
        output_data = {
            'content': output_content,
            'output_type': output_type,
            'has_tool_calls': has_tool_calls
        }
        if tool_calls:
            output_data['tool_calls'] = tool_calls

        writer.write_span_end(
            span_id=span_id,
            duration_ms=duration_ms,
            output_data=output_data,
            status=status,
            error=error_message,
            tokens_used=tokens_used
        )

    except Exception as e:
        logger.error(f"Failed to log model response: {e}")


def trace_model_call(node_name: str):
    """
    Decorator 装饰器记录模型调用
    Phase 1: 只捕获和打印数据，不写数据库

    Usage:
        @trace_model_call("ReasonNode")
        async def call_model(self, messages, tools=None, ...):
            ...
    """
    print(f"🔧 [DECORATOR_INIT] trace_model_call decorator initialized for node={node_name}", flush=True)
    def decorator(func):
        print(f"🔧 [DECORATOR_APPLIED] Applying trace_model_call to func={func.__name__}", flush=True)
        @wraps(func)
        async def wrapper(self, messages, tools=None, model=None, provider=None,
                         stream_tokens=True, output_format=None, *args, **kwargs):

            # DEBUG: Write directly to file to bypass any logging issues
            try:
                with open('/tmp/trace_decorator_direct.log', 'a') as f:
                    f.write(f"[WRAPPER_EXEC] node={node_name}, func={func.__name__}, time={time.time()}\n")
                    f.flush()
            except:
                pass

            # DEBUG: Confirm decorator is being called
            logger.info(f"🔍 [TRACE_DECORATOR_CALLED] node={node_name}, func={func.__name__}")

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

            # 记录调用开始
            log_model_call(
                span_id=span_id,
                session_id=session_id,
                node_name=node_name,
                model=model or 'unknown',
                provider=provider or 'unknown',
                messages=messages,
                tools=tools,
                stream_mode=stream_tokens,
                output_format=output_format
            )

            # 执行原函数
            start_time = time.time()
            try:
                result = await func(self, messages, tools=tools, model=model,
                                  provider=provider, stream_tokens=stream_tokens,
                                  output_format=output_format, *args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                # 解析响应
                if isinstance(result, tuple):
                    response, _ = result
                else:
                    response = result

                output_content = getattr(response, 'content', '')
                output_type = type(response).__name__
                has_tool_calls = hasattr(response, 'tool_calls') and bool(getattr(response, 'tool_calls', []))
                tool_calls = getattr(response, 'tool_calls', None) if has_tool_calls else None

                # 记录响应
                log_model_response(
                    span_id=span_id,
                    output_content=output_content,
                    output_type=output_type,
                    duration_ms=duration_ms,
                    has_tool_calls=has_tool_calls,
                    tool_calls=tool_calls,
                    status='success'
                )

                return result

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                # 记录错误
                log_model_response(
                    span_id=span_id,
                    output_content='',
                    output_type='error',
                    duration_ms=duration_ms,
                    status='error',
                    error_message=str(e)
                )
                raise

        return wrapper
    return decorator
