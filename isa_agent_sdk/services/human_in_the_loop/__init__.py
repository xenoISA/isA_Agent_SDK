"""
Human-in-the-Loop Service Package (Refactored)
File: app/services/human_in_the_loop/__init__.py

This package provides a modular, professional implementation of HIL functionality.

Example Usage:
    # Import the main service
    from isa_agent_sdk.services.human_in_the_loop import get_hil_service, HILService

    # Get service instance
    hil = get_hil_service()

    # Use 5 core HIL methods
    # 1. Authorization
    approved = await hil.request_authorization(
        scenario="payment",
        data={"action": "Process payment", "reason": "...", "risk_level": "high"}
    )

    # 2. Input
    api_key = await hil.request_input(
        scenario="credentials",
        data={"input_type": "credentials", "prompt": "Enter API key", "description": "..."}
    )

    # 3. Review
    result = await hil.request_review(
        scenario="execution_plan",
        data={"content": plan, "content_type": "execution_plan", "instructions": "..."}
    )

    # 4. Input + Authorization
    result = await hil.request_input_with_authorization(
        scenario="payment_with_amount",
        data={"input_prompt": "Enter amount", "authorization_reason": "...", "risk_level": "high"}
    )

    # 5. Execution Choice (for long-running tasks)
    choice = await hil.request_execution_choice(
        scenario="long_running_web_crawl",
        data={"prompt": "...", "estimated_time_seconds": 60, "tool_count": 5, "task_type": "web_crawling", "recommendation": "background", "options": [...]}
    )

    # Import specific components if needed
    from isa_agent_sdk.services.human_in_the_loop.models import (
        AuthorizationRequest, InputRequest, ReviewRequest, InputWithAuthRequest
    )
"""

# Main service exports
from .hil_service import (
    HILService,
    get_hil_service,
    reset_hil_service
)

# Data models
from .models import (
    InterruptType,
    InterventionType,
    SecurityLevel,
    InterruptData,
    ApprovalInterruptData,
    ReviewEditInterruptData,
    ValidationInterruptData,
    ToolAuthorizationInterruptData,
    OAuthAuthorizationInterruptData,
    CredentialAuthorizationInterruptData,
    ManualInterventionInterruptData,
    InterruptStats,
    # 4 Core HIL Request Models (MCP-aligned)
    AuthorizationRequest,
    InputRequest,
    ReviewRequest,
    InputWithAuthRequest,
    # 5th HIL Method - Execution Choice
    ExecutionChoiceRequest
)

# Validators
from .validators import (
    HILValidator,
    ValidationRulesBuilder
)

# Core components (for advanced usage)
from .interrupt_manager import InterruptManager
from .scenario_handlers import ScenarioHandler

# Package metadata
__version__ = "2.0.0"
__author__ = "isA Agent Team"

# Public API
__all__ = [
    # Main service
    "HILService",
    "get_hil_service",
    "reset_hil_service",

    # Data models (legacy)
    "InterruptType",
    "InterventionType",
    "SecurityLevel",
    "InterruptData",
    "ApprovalInterruptData",
    "ReviewEditInterruptData",
    "ValidationInterruptData",
    "ToolAuthorizationInterruptData",
    "OAuthAuthorizationInterruptData",
    "CredentialAuthorizationInterruptData",
    "ManualInterventionInterruptData",
    "InterruptStats",

    # 4 Core HIL Request Models (MCP-aligned)
    "AuthorizationRequest",
    "InputRequest",
    "ReviewRequest",
    "InputWithAuthRequest",
    # 5th HIL Method - Execution Choice
    "ExecutionChoiceRequest",

    # Validators
    "HILValidator",
    "ValidationRulesBuilder",

    # Core components (advanced)
    "InterruptManager",
    "ScenarioHandler"
]

