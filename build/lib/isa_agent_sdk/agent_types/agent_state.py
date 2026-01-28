"""
Agent State definition for LangGraph
"""
from typing import Annotated, Optional, Dict, Any, List, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps


def preserve_latest(existing, new):
    """Reducer that preserves the latest non-empty value"""
    return new if new is not None and new != "" else existing


def append_if_not_none(existing, new):
    """Reducer that appends new items to existing list if new is not None"""
    if existing is None:
        existing = []
    if new is None:
        return existing
    if isinstance(new, list):
        return existing + new
    return existing + [new]


def merge_dicts(existing, new):
    """Reducer that merges dictionaries, with new values overwriting existing"""
    if existing is None:
        existing = {}
    if new is None:
        return existing
    return {**existing, **new}


def sum_numeric(existing, new):
    """Reducer that sums numeric values"""
    if existing is None:
        existing = 0
    if new is None:
        return existing
    return existing + new


class AgentState(TypedDict):
    """
    Comprehensive LangGraph state for complex agent workflows with LangGraph loop patterns
    """

    # Messages with proper reducer
    messages: Annotated[List[AnyMessage], add_messages]

    # Conversation summarization (Official LangGraph pattern)
    # Compresses old messages to manage context window
    summary: Annotated[Optional[str], preserve_latest]

    # Core workflow control
    next_action: Annotated[Optional[str], preserve_latest]

    # LangGraph loop control - official RemainingSteps for recursion management
    remaining_steps: RemainingSteps

    # Task execution state for autonomous agent loops
    autonomous_tasks: Annotated[Optional[List[Dict[str, Any]]], preserve_latest]
    task_list: Annotated[Optional[List[Dict[str, Any]]], preserve_latest]
    current_task_index: Annotated[Optional[int], preserve_latest]
    execution_mode: Annotated[Optional[str], preserve_latest]  # "sequential" or "parallel"

    # Task execution tracking
    completed_task_count: Annotated[Optional[int], sum_numeric]
    failed_task_count: Annotated[Optional[int], sum_numeric]

    # Execution plan from planning tools
    execution_plan: Annotated[Optional[Dict[str, Any]], preserve_latest]


class MCPResourcesSchema(TypedDict):
    """
    MCP Resources schema for type safety
    """
    mcp_tools: Optional[List[Dict[str, Any]]]
    mcp_prompts: Optional[List[Dict[str, Any]]]
    mcp_resources: Optional[List[Dict[str, Any]]]


class GuardrailResult(TypedDict):
    """
    Guardrail check result schema
    """
    action: str  # "allow", "block", "modify"
    violations: Optional[List[str]]
    modified_content: Optional[str]