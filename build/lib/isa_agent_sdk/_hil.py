#!/usr/bin/env python3
"""
isA Agent SDK - Human-in-the-Loop Integration
==============================================

SDK wrapper around the existing HIL service.
Provides Claude SDK-compatible interface while using isA's powerful
durable execution HIL system underneath.

isA's HIL is MORE powerful than Claude SDK hooks:
- Durable execution (survives restarts via LangGraph checkpointing)
- Async human responses (can wait hours/days)
- Full scenarios: authorization, review, input validation with retry

Example:
    from isa_agent_sdk import query, ISAAgentOptions
    from isa_agent_sdk.hil import request_tool_permission, collect_input

    # Request permission before dangerous operation
    authorized = await request_tool_permission(
        tool_name="delete_file",
        tool_args={"path": "/important/data.txt"},
        reason="User requested file deletion"
    )

    if authorized:
        # Proceed with operation
        ...
"""

from typing import Dict, Any, Optional, List

# Import the existing HIL service - REUSE, don't duplicate!
from .services.human_in_the_loop import get_hil_service, HILService
from .services.human_in_the_loop.models import InterruptStats


def get_hil() -> HILService:
    """
    Get the HIL service instance.

    Returns:
        HILService singleton instance
    """
    return get_hil_service()


# === Authorization Methods ===

async def request_tool_permission(
    tool_name: str,
    tool_args: Dict[str, Any],
    security_level: str = "HIGH",
    reason: Optional[str] = None,
    user_id: str = "default"
) -> bool:
    """
    Request permission to execute a tool.

    This pauses graph execution until human responds.
    Uses durable execution - survives process restarts.

    Args:
        tool_name: Name of tool requesting permission
        tool_args: Arguments for the tool
        security_level: Security level (LOW, MEDIUM, HIGH, CRITICAL)
        reason: Reason for needing this tool
        user_id: User identifier

    Returns:
        True if authorized, False if rejected

    Example:
        authorized = await request_tool_permission(
            tool_name="web_crawl",
            tool_args={"url": "https://example.com"},
            security_level="HIGH",
            reason="Need to fetch page content"
        )
    """
    hil = get_hil_service()
    return await hil.request_authorization(
        scenario="tool_authorization",
        data={
            "action": f"Execute tool: {tool_name}",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": reason or f"Tool {tool_name} requires authorization",
            "risk_level": security_level.lower(),
            "context": {"tool_name": tool_name, "args": tool_args}
        },
        user_id=user_id,
        node_source="sdk"
    )


async def request_authorization(
    action: str,
    reason: str,
    context: Optional[Dict[str, Any]] = None,
    risk_level: str = "medium",
    user_id: str = "default"
) -> bool:
    """
    Request generic authorization for an action.

    Args:
        action: Description of the action
        reason: Why authorization is needed
        context: Additional context
        risk_level: Risk level (low, medium, high, critical)
        user_id: User identifier

    Returns:
        True if authorized, False if rejected

    Example:
        authorized = await request_authorization(
            action="Delete all temporary files",
            reason="Cleanup requested by user",
            risk_level="high"
        )
    """
    hil = get_hil_service()
    return await hil.request_authorization(
        scenario="generic",
        data={
            "action": action,
            "reason": reason,
            "risk_level": risk_level,
            "context": context or {}
        },
        user_id=user_id,
        node_source="sdk"
    )


# === Input Collection Methods ===

async def collect_input(
    prompt: str,
    description: Optional[str] = None,
    input_type: str = "text",
    schema: Optional[Dict[str, Any]] = None,
    user_id: str = "default"
) -> Any:
    """
    Collect input from the user.

    Pauses execution until user provides input.

    Args:
        prompt: Question/prompt for the user
        description: Additional description
        input_type: Type of input (text, number, boolean, selection)
        schema: JSON schema for validation
        user_id: User identifier

    Returns:
        User's input (validated if schema provided)

    Example:
        email = await collect_input(
            prompt="What is your email address?",
            input_type="text",
            schema={"type": "string", "format": "email"}
        )
    """
    hil = get_hil_service()
    return await hil.request_input(
        scenario="generic",
        data={
            "prompt": prompt,
            "description": description or prompt,
            "input_type": input_type,
            "schema": schema
        },
        user_id=user_id,
        node_source="sdk"
    )


async def collect_credentials(
    prompt: str,
    credential_type: str = "api_key",
    description: Optional[str] = None,
    user_id: str = "default"
) -> str:
    """
    Collect credentials from the user.

    Args:
        prompt: Prompt for credentials
        credential_type: Type (api_key, password, token)
        description: Additional description
        user_id: User identifier

    Returns:
        Credential string

    Example:
        api_key = await collect_credentials(
            prompt="Enter your OpenAI API key",
            credential_type="api_key"
        )
    """
    hil = get_hil_service()
    return await hil.request_input(
        scenario="credentials",
        data={
            "input_type": "credentials",
            "prompt": prompt,
            "description": description or prompt,
            "credential_type": credential_type
        },
        user_id=user_id,
        node_source="sdk"
    )


async def collect_selection(
    prompt: str,
    options: List[str],
    description: Optional[str] = None,
    user_id: str = "default"
) -> str:
    """
    Collect a selection from predefined options.

    Args:
        prompt: Question for the user
        options: List of options to choose from
        description: Additional description
        user_id: User identifier

    Returns:
        Selected option

    Example:
        choice = await collect_selection(
            prompt="Which database should I use?",
            options=["PostgreSQL", "MySQL", "SQLite"]
        )
    """
    hil = get_hil_service()
    return await hil.request_input(
        scenario="selection",
        data={
            "input_type": "selection",
            "prompt": prompt,
            "description": description or prompt,
            "options": options
        },
        user_id=user_id,
        node_source="sdk"
    )


# === Review Methods ===

async def request_review(
    content: str,
    content_type: str = "text",
    instructions: Optional[str] = None,
    editable: bool = True,
    user_id: str = "default"
) -> Dict[str, Any]:
    """
    Request human review of content.

    Args:
        content: Content to review
        content_type: Type (text, code, plan, config)
        instructions: Review instructions
        editable: Whether user can edit
        user_id: User identifier

    Returns:
        Dict with 'approved' (bool), 'edited_content' (if edited), 'action' (str)

    Example:
        result = await request_review(
            content=generated_code,
            content_type="code",
            instructions="Review this code before execution",
            editable=True
        )
        if result['approved']:
            code = result.get('edited_content', generated_code)
    """
    hil = get_hil_service()
    return await hil.request_review(
        scenario=content_type if content_type in ["code", "config"] else "generic",
        data={
            "content": content,
            "content_type": content_type,
            "instructions": instructions or f"Please review this {content_type}",
            "editable": editable
        },
        user_id=user_id,
        node_source="sdk"
    )


async def request_plan_approval(
    plan: str,
    plan_title: Optional[str] = None,
    user_id: str = "default"
) -> Dict[str, Any]:
    """
    Request approval for an execution plan.

    Args:
        plan: The plan content (text or JSON)
        plan_title: Title for the plan
        user_id: User identifier

    Returns:
        Dict with 'approved' (bool), 'edited_content' (if edited)

    Example:
        result = await request_plan_approval(
            plan="1. Read file\\n2. Parse data\\n3. Update database",
            plan_title="Data Migration Plan"
        )
    """
    hil = get_hil_service()
    return await hil.request_review(
        scenario="execution_plan",
        data={
            "content": plan,
            "content_type": "execution_plan",
            "instructions": f"Review execution plan: {plan_title or 'Untitled'}",
            "editable": True
        },
        user_id=user_id,
        node_source="sdk"
    )


# === Execution Choice ===

async def request_execution_choice(
    prompt: str,
    options: List[Dict[str, str]],
    estimated_time_seconds: Optional[float] = None,
    recommendation: Optional[str] = None,
    user_id: str = "default"
) -> str:
    """
    Request user choice for execution strategy.

    Useful for long-running tasks where user can choose
    between quick/comprehensive/background execution.

    Args:
        prompt: Description of the situation
        options: List of options with 'value', 'label', 'description'
        estimated_time_seconds: Estimated execution time
        recommendation: Recommended option value
        user_id: User identifier

    Returns:
        Selected option value

    Example:
        choice = await request_execution_choice(
            prompt="This task will take ~60 seconds",
            options=[
                {"value": "quick", "label": "Quick", "description": "Fast (~30s)"},
                {"value": "full", "label": "Full", "description": "Complete (~60s)"},
                {"value": "background", "label": "Background", "description": "Run async"}
            ],
            recommendation="background"
        )
    """
    hil = get_hil_service()
    return await hil.request_execution_choice(
        scenario="long_running_task",
        data={
            "prompt": prompt,
            "options": options,
            "estimated_time_seconds": estimated_time_seconds,
            "recommendation": recommendation
        },
        user_id=user_id,
        node_source="sdk"
    )


# === Checkpoint (Collaborative Mode) ===

async def checkpoint(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    require_approval: bool = True,
    user_id: str = "default"
) -> bool:
    """
    Create a checkpoint in collaborative mode.

    Pauses execution for human review/approval before continuing.

    Args:
        message: Checkpoint message/question
        context: Additional context to show
        require_approval: Whether to require explicit approval
        user_id: User identifier

    Returns:
        True if approved to continue, False otherwise

    Example:
        # In collaborative mode, checkpoint every N steps
        approved = await checkpoint(
            message="Completed step 3. Ready to proceed?",
            context={"completed": ["step1", "step2", "step3"]}
        )
    """
    if not require_approval:
        return True

    hil = get_hil_service()
    return await hil.request_authorization(
        scenario="generic",
        data={
            "action": "Continue execution",
            "reason": message,
            "risk_level": "low",
            "context": context or {}
        },
        user_id=user_id,
        node_source="sdk_checkpoint"
    )


# === Statistics ===

def get_hil_stats() -> InterruptStats:
    """
    Get HIL interrupt statistics.

    Returns:
        InterruptStats with total, by_type, by_node, latest
    """
    hil = get_hil_service()
    return hil.get_interrupt_stats()


def clear_hil_history() -> None:
    """Clear HIL interrupt history"""
    hil = get_hil_service()
    hil.clear_interrupt_history()


__all__ = [
    # Service access
    "get_hil",

    # Authorization
    "request_tool_permission",
    "request_authorization",

    # Input collection
    "collect_input",
    "collect_credentials",
    "collect_selection",

    # Review
    "request_review",
    "request_plan_approval",

    # Execution choice
    "request_execution_choice",

    # Collaborative mode
    "checkpoint",

    # Statistics
    "get_hil_stats",
    "clear_hil_history",
]
