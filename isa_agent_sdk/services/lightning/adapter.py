#!/usr/bin/env python3
"""
Lightning Adapter - Bridge between existing Tracing and Agent Lightning

将现有的 tracing 数据自动同步到 Agent Lightning 用于训练
"""

from typing import Any, Dict, List, Optional

from isa_agent_sdk.utils.logger import agent_logger
from .service import get_lightning_service

logger = agent_logger


class LightningAdapter:
    """
    将现有 tracing 数据适配到 Agent Lightning 格式
    """

    def __init__(self):
        self.lightning = get_lightning_service()

    async def on_model_call_start(
        self,
        span_id: str,
        session_id: str,
        node_name: str,
        model: str,
        messages: List[Dict],
        **kwargs,
    ):
        """
        当模型调用开始时，同步到 Lightning

        从 trace/model_callback.py 的 log_model_call 调用
        """
        if not self.lightning.is_enabled:
            return

        try:
            # 提取 prompt (最后一条用户消息)
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    prompt_parts.append(
                        f"{msg.get('type', 'unknown')}: {msg.get('content', '')[:100]}"
                    )

            prompt = "\n".join(prompt_parts)

            # 记录状态
            await self.lightning.emit_state(
                thread_id=session_id,
                state_data={
                    "span_id": span_id,
                    "node_name": node_name,
                    "model": model,
                    "message_count": len(messages),
                },
                node_name=node_name,
            )

        except Exception as e:
            logger.warning(f"Lightning adapter failed to emit model call start: {e}")

    async def on_model_call_end(
        self,
        span_id: str,
        session_id: str,
        node_name: str,
        model: str,
        prompt: str,
        response: str,
        duration_ms: int,
        has_tool_calls: bool = False,
        tool_calls: Optional[List[Dict]] = None,
        **kwargs,
    ):
        """
        当模型调用完成时，同步到 Lightning

        从 trace/model_callback.py 的 log_model_response 调用
        """
        if not self.lightning.is_enabled:
            return

        try:
            # 记录 LLM 调用
            await self.lightning.emit_llm_call(
                thread_id=session_id,
                prompt=prompt[:500],  # 限制长度
                response=response[:500],
                model=model,
                node_name=node_name,
                metadata={
                    "span_id": span_id,
                    "duration_ms": duration_ms,
                    "has_tool_calls": has_tool_calls,
                },
            )

            # 如果有工具调用，也记录
            if has_tool_calls and tool_calls:
                for tc in tool_calls:
                    await self.lightning.emit_action(
                        thread_id=session_id,
                        action_type="tool_call",
                        action_data={
                            "tool_name": tc.get("name"),
                            "tool_args": tc.get("args", {}),
                        },
                        node_name=node_name,
                    )

        except Exception as e:
            logger.warning(f"Lightning adapter failed to emit model call end: {e}")

    async def on_mcp_operation(
        self,
        span_id: str,
        session_id: str,
        operation_type: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any,
        duration_ms: int,
        success: bool,
        **kwargs,
    ):
        """
        当 MCP 工具调用时，同步到 Lightning

        从 trace/mcp_callback.py 调用
        """
        if not self.lightning.is_enabled:
            return

        try:
            await self.lightning.emit_tool_call(
                thread_id=session_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=result,
                success=success,
                node_name="mcp",
                execution_time=duration_ms / 1000.0,
            )

        except Exception as e:
            logger.warning(f"Lightning adapter failed to emit MCP operation: {e}")


# Global singleton
_adapter: Optional[LightningAdapter] = None


def get_lightning_adapter() -> LightningAdapter:
    """Get or create Lightning adapter singleton"""
    global _adapter
    if _adapter is None:
        _adapter = LightningAdapter()
    return _adapter
