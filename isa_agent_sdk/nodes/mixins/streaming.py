"""
Streaming Mixin - Methods for streaming data to clients
"""
from typing import Dict, Any

from langgraph.config import get_stream_writer


class StreamingMixin:
    """Mixin providing streaming methods for BaseNode"""

    def stream_custom(self, data: Dict[str, Any]) -> None:
        """
        Unified streaming method for both LLM tokens and tool progress

        Support two types of streaming:

        1. LLM Token Streaming (based on LangGraph example):
           writer({"custom_llm_chunk": chunk})

        2. Tool Progress Streaming (based on LangGraph example):
           writer({"data": "Retrieved 0/100 records", "type": "progress"})

        Args:
            data: Streaming data - can be:
                  - {"custom_llm_chunk": "token"} for LLM tokens
                  - {"data": "progress info", "type": "progress"} for tool progress
                  - Any other custom streaming data
        """
        writer = get_stream_writer()
        if writer:
            writer(data)
        else:
            self.logger.debug(f"stream_custom | no_writer | data_keys={list(data.keys())}")

    def stream_tool(self, tool_name: str, progress_info: str):
        """
        Stream tool progress using unified tool_execution format

        Args:
            tool_name: Name of tool being executed
            progress_info: Progress information
        """
        # Use unified tool_execution format (consistent with other tool events)
        self.stream_custom({
            "tool_execution": {
                "status": "executing",
                "tool_name": tool_name,
                "progress": progress_info
            }
        })
