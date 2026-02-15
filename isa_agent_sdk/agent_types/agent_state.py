"""
Agent State definition for LangGraph

This module defines the state schema and reducer functions for LangGraph agent workflows.
All reducers are fully typed to ensure type safety across the codebase.
"""
from typing import Annotated, Optional, Dict, Any, List, TypedDict, TypeVar, Union
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

T = TypeVar('T')


def preserve_latest(existing: Optional[T], new: Optional[T]) -> Optional[T]:
    """
    Reducer that preserves the latest non-empty value.

    Args:
        existing: The current value in state
        new: The new value to potentially apply

    Returns:
        The new value if it is not None and not empty string, otherwise existing
    """
    return new if new is not None and new != "" else existing


def append_if_not_none(existing: Optional[List[Any]], new: Optional[Union[Any, List[Any]]]) -> List[Any]:
    """
    Reducer that appends new items to existing list if new is not None.

    Args:
        existing: The current list in state (or None)
        new: The new item(s) to append (single item or list)

    Returns:
        Updated list with new items appended
    """
    if existing is None:
        existing = []
    if new is None:
        return existing
    if isinstance(new, list):
        return existing + new
    return existing + [new]


def merge_dicts(existing: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reducer that merges dictionaries, with new values overwriting existing.

    Args:
        existing: The current dictionary in state (or None)
        new: The new dictionary to merge

    Returns:
        Merged dictionary with new values taking precedence
    """
    if existing is None:
        existing = {}
    if new is None:
        return existing
    return {**existing, **new}


def sum_numeric(existing: Optional[Union[int, float]], new: Optional[Union[int, float]]) -> Union[int, float]:
    """
    Reducer that sums numeric values.

    Args:
        existing: The current numeric value in state (or None)
        new: The new numeric value to add

    Returns:
        Sum of existing and new values
    """
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
    next_agent: Annotated[Optional[str], preserve_latest]
    active_agent: Annotated[Optional[str], preserve_latest]

    # LangGraph loop control - official RemainingSteps for recursion management
    remaining_steps: RemainingSteps

    # Task execution state for autonomous agent loops
    autonomous_tasks: Annotated[Optional[List[Dict[str, Any]]], preserve_latest]
    task_list: Annotated[Optional[List[Dict[str, Any]]], preserve_latest]
    current_task_index: Annotated[Optional[int], preserve_latest]
    execution_mode: Annotated[Optional[str], preserve_latest]  # "sequential" or "parallel"

    # Task execution tracking
    completed_task_count: Annotated[Optional[int], preserve_latest]
    failed_task_count: Annotated[Optional[int], preserve_latest]

    # Execution plan from planning tools
    execution_plan: Annotated[Optional[Dict[str, Any]], preserve_latest]

    # Shared state across multi-agent graphs
    shared_state: Annotated[Optional[Dict[str, Any]], merge_dicts]
    agent_outputs: Annotated[Optional[Dict[str, Any]], merge_dicts]

    # Auto-approve settings (similar to Claude Code's -y flag)
    # When True, plan reviews are auto-approved without HIL interruption
    auto_approve_plans: Annotated[Optional[bool], preserve_latest]
    auto_approve_code: Annotated[Optional[bool], preserve_latest]

    # Autonomous execution flag (set by ToolNode when plan is detected)
    is_autonomous: Annotated[Optional[bool], preserve_latest]
    execution_strategy: Annotated[Optional[str], preserve_latest]  # "autonomous_planning", etc.

    # Task DAG state (optional - when present, enables DAG-aware execution)
    task_dag: Annotated[Optional[Dict[str, Any]], preserve_latest]


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
