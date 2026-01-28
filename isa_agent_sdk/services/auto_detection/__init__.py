"""
Auto-Detection Service - Tool Profiler

Records and estimates tool execution times using Redis sorted sets.
"""

from .tool_profiler import ToolProfiler, get_tool_profiler

__all__ = ["ToolProfiler", "get_tool_profiler"]
