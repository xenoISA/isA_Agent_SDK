"""
Scenario-based HIL handlers
File: app/services/human_in_the_loop/scenario_handlers.py

This module provides scenario-based HIL methods for common use cases.
"""

from typing import Dict, Any, Awaitable
from langgraph.types import interrupt
import uuid

from .models import (
    AuthorizationRequest,
    InputRequest,
    ReviewRequest,
    InputWithAuthRequest,
    ExecutionChoiceRequest
)
from .validators import HILValidator
from .timeout_handler import TimeoutHandler
from isa_agent_sdk.utils.logger import api_logger


class ScenarioHandler:
    """
    Handles scenario-based HIL interrupts with timeout support

    This class provides high-level methods for common HIL scenarios:
    - Collect user input
    - Request tool permission
    - Request OAuth authorization
    - Request credential usage
    - Request manual intervention

    All methods now support automatic timeout handling with sensible defaults.
    """

    def __init__(self, interrupt_manager, mcp_client=None):
        self.interrupt_manager = interrupt_manager
        self.mcp_client = mcp_client
        self.validator = HILValidator()
        self.timeout_handler = TimeoutHandler()


    async def _trigger_interrupt(self, interrupt_data: Dict[str, Any]) -> Any:
        """
        Trigger LangGraph interrupt (helper method for timeout wrapper)

        Args:
            interrupt_data: Interrupt data to send

        Returns:
            Human response
        """
        return interrupt(interrupt_data)

    async def handle_authorization_scenario(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> bool:
        """
        Core Handler 1: Authorization scenarios (with timeout)

        Routes to specific authorization handlers based on scenario type.

        Scenarios:
            - tool_authorization: Tool permission (maps to request_tool_permission)
            - payment: Financial authorization
            - deletion: Destructive operation authorization
            - deployment: Deployment authorization
            - generic: General authorization

        Args:
            scenario: Scenario identifier
            data: MCP HIL data (must contain action, reason, risk_level)
            user_id: User identifier
            node_source: Source node

        Returns:
            True if authorized, False if rejected (also False on timeout)
        """
        api_logger.info(f"HIL: Authorization scenario - {scenario}")

        # Extract data fields
        action = data.get("action", "Unknown action")
        reason = data.get("reason", "No reason provided")
        risk_level = data.get("risk_level", "high")
        context = data.get("context", {})

        # Create authorization request model
        auth_req = AuthorizationRequest(
            action=action,
            reason=reason,
            risk_level=risk_level,
            scenario=scenario,
            context=context
        )

        # Convert to interrupt data
        interrupt_data = auth_req.to_interrupt_data()
        interrupt_data["user_id"] = user_id
        interrupt_data["node_source"] = node_source
        interrupt_data["id"] = str(uuid.uuid4())

        # Log interrupt
        self.interrupt_manager._log_interrupt(interrupt_data)

        # Get timeout
        timeout = self.timeout_handler.get_timeout("authorization", data.get("timeout"))
        api_logger.info(f"🔔 [HIL] Authorization: {scenario} (timeout={timeout}s)")

        # Trigger interrupt with timeout
        human_response = await self.timeout_handler.with_timeout(
            coro=self._trigger_interrupt(interrupt_data),
            timeout=timeout,
            scenario=scenario,
            default_value={"approved": False}  # Timeout = reject (safety first)
        )

        # Validate response
        approved = self.validator.is_approved(human_response)

        api_logger.info(f"✅ [HIL] Authorization {scenario} - {'APPROVED' if approved else 'REJECTED'}")
        return approved

    async def handle_input_scenario(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> Any:
        """
        Core Handler 2: Input scenarios (with timeout)

        Routes to specific input handlers based on scenario type.

        Scenarios:
            - credentials: Credential input
            - selection: Selection from options
            - augmentation: Data augmentation
            - generic: General input

        Args:
            scenario: Scenario identifier
            data: MCP HIL data (must contain input_type, prompt, description)
            user_id: User identifier
            node_source: Source node

        Returns:
            User input (validated if schema provided), or default_value on timeout
        """
        api_logger.info(f"HIL: Input scenario - {scenario}")

        # Extract data fields
        input_type = data.get("input_type", "text")
        prompt = data.get("prompt", "Please provide input")
        description = data.get("description", "")
        schema = data.get("schema")
        current_data = data.get("current_data")
        suggestions = data.get("suggestions")
        default_value = data.get("default_value")

        # Create input request model
        input_req = InputRequest(
            input_type=input_type,
            prompt=prompt,
            description=description,
            scenario=scenario,
            schema=schema,
            current_data=current_data,
            suggestions=suggestions,
            default_value=default_value
        )

        # Convert to interrupt data
        interrupt_data = input_req.to_interrupt_data()
        interrupt_data["user_id"] = user_id
        interrupt_data["node_source"] = node_source
        interrupt_data["id"] = str(uuid.uuid4())

        # Log interrupt
        self.interrupt_manager._log_interrupt(interrupt_data)

        # Get timeout
        timeout = self.timeout_handler.get_timeout("input", data.get("timeout"))
        api_logger.info(f"🔔 [HIL] Input: {scenario} (timeout={timeout}s)")

        # Trigger interrupt with timeout
        human_response = await self.timeout_handler.with_timeout(
            coro=self._trigger_interrupt(interrupt_data),
            timeout=timeout,
            scenario=scenario,
            default_value=default_value  # Use default_value on timeout
        )

        # Validate if schema provided
        if schema and human_response is not None:
            validation_result = self.validator.validate_input(human_response, schema)
            if not validation_result["valid"]:
                api_logger.warning(f"HIL: Input validation failed - {validation_result['error']}")
            return validation_result.get("value", human_response)

        api_logger.info(f"✅ [HIL] Input {scenario} collected")
        return human_response

    async def handle_review_scenario(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> Dict[str, Any]:
        """
        Core Handler 3: Review scenarios (with timeout)

        Routes to specific review handlers based on scenario type.

        Scenarios:
            - execution_plan: Review execution plan
            - code: Review generated code
            - config: Review configuration
            - generic: General review

        Args:
            scenario: Scenario identifier
            data: MCP HIL data (must contain content, content_type, instructions)
            user_id: User identifier
            node_source: Source node

        Returns:
            Dict with 'approved' (bool), 'edited_content' (if edited), 'action' (str)
            On timeout: auto-approves with original content
        """
        api_logger.info(f"HIL: Review scenario - {scenario}")

        # Extract data fields
        content = data.get("content")
        content_type = data.get("content_type", "text")
        instructions = data.get("instructions", "Please review the content")
        editable = data.get("editable", True)

        # Create review request model
        review_req = ReviewRequest(
            content=content,
            content_type=content_type,
            instructions=instructions,
            scenario=scenario,
            editable=editable
        )

        # Convert to interrupt data
        interrupt_data = review_req.to_interrupt_data()
        interrupt_data["user_id"] = user_id
        interrupt_data["node_source"] = node_source
        interrupt_data["id"] = str(uuid.uuid4())

        # Log interrupt
        self.interrupt_manager._log_interrupt(interrupt_data)

        # Get timeout
        timeout = self.timeout_handler.get_timeout("review", data.get("timeout"))
        api_logger.info(f"🔔 [HIL] Review: {scenario} (timeout={timeout}s)")

        # Default behavior on timeout: approve with original content
        default_response = {
            "approved": True,
            "edited_content": content,
            "action": "approve_by_timeout",
            "original_content": content
        }

        # Trigger interrupt with timeout
        human_response = await self.timeout_handler.with_timeout(
            coro=self._trigger_interrupt(interrupt_data),
            timeout=timeout,
            scenario=scenario,
            default_value=default_response
        )

        # Parse response
        if isinstance(human_response, dict):
            action = human_response.get("action", "approve")
            approved = action == "approve" or action == "approve_by_timeout"
            edited_content = human_response.get("edited_content", content)
        else:
            # Simple response - treat as approval
            approved = self.validator.is_approved(human_response)
            action = "approve" if approved else "reject"
            edited_content = content

        api_logger.info(f"✅ [HIL] Review {scenario} - {action}")

        return {
            "approved": approved,
            "edited_content": edited_content,
            "action": action,
            "original_content": content
        }

    async def handle_input_with_auth_scenario(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> Dict[str, Any]:
        """
        Core Handler 4: Input + Authorization scenarios (with timeout)

        Routes to specific input+auth handlers based on scenario type.

        Scenarios:
            - payment_with_amount: Input payment amount + authorize
            - deploy_with_config: Input deployment config + authorize
            - generic: General input + authorization

        Args:
            scenario: Scenario identifier
            data: MCP HIL data (must contain input_prompt, authorization_reason, risk_level)
            user_id: User identifier
            node_source: Source node

        Returns:
            Dict with 'user_input' (Any), 'approved' (bool), 'action' (str)
            On timeout: returns rejected (approved=False)
        """
        api_logger.info(f"HIL: Input+Auth scenario - {scenario}")

        # Extract data fields
        input_prompt = data.get("input_prompt", "Please provide input")
        input_description = data.get("input_description", "")
        authorization_reason = data.get("authorization_reason", "Authorization required")
        input_type = data.get("input_type", "text")
        risk_level = data.get("risk_level", "high")
        schema = data.get("schema")
        context = data.get("context", {})

        # Create combined request model
        combined_req = InputWithAuthRequest(
            input_prompt=input_prompt,
            input_description=input_description,
            authorization_reason=authorization_reason,
            input_type=input_type,
            risk_level=risk_level,
            scenario=scenario,
            schema=schema,
            context=context
        )

        # Convert to interrupt data
        interrupt_data = combined_req.to_interrupt_data()
        interrupt_data["user_id"] = user_id
        interrupt_data["node_source"] = node_source
        interrupt_data["id"] = str(uuid.uuid4())

        # Log interrupt
        self.interrupt_manager._log_interrupt(interrupt_data)

        # Get timeout
        timeout = self.timeout_handler.get_timeout("input_with_auth", data.get("timeout"))
        api_logger.info(f"🔔 [HIL] Input+Auth: {scenario} (timeout={timeout}s)")

        # Default behavior on timeout: reject (safety first)
        default_response = {
            "user_input": None,
            "approved": False,
            "action": "timeout_rejected",
            "scenario": scenario
        }

        # Trigger interrupt with timeout
        human_response = await self.timeout_handler.with_timeout(
            coro=self._trigger_interrupt(interrupt_data),
            timeout=timeout,
            scenario=scenario,
            default_value=default_response
        )

        # Parse response
        if isinstance(human_response, dict):
            user_input = human_response.get("user_input", human_response.get("input"))
            approved = human_response.get("approved", False)
            action = human_response.get("action", "approve_with_input" if approved else "cancel")
        else:
            # Simple response - extract input and assume approved
            user_input = human_response
            approved = True
            action = "approve_with_input"

        # Validate input if schema provided
        if schema and user_input:
            validation_result = self.validator.validate_input(user_input, schema)
            if not validation_result["valid"]:
                api_logger.warning(f"HIL: Input validation failed - {validation_result['error']}")
                approved = False
                action = "validation_failed"
            else:
                user_input = validation_result.get("value", user_input)

        api_logger.info(f"✅ [HIL] Input+Auth {scenario} - {action}")

        return {
            "user_input": user_input,
            "approved": approved,
            "action": action,
            "scenario": scenario
        }

    async def handle_execution_choice_scenario(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> str:
        """
        Core Handler 5: Execution Choice scenarios (with timeout)

        NEW: Handle long-running task execution mode selection.

        This is triggered by ToolNode (not MCP tools) when it detects
        a task that will take significant time to execute.

        Scenarios:
            - long_running_web_crawl: Multiple web pages
            - long_running_web_search: Multiple searches
            - long_running_mixed: Mixed operations

        Args:
            scenario: Scenario identifier
            data: Execution choice data (from BackgroundHILDetector)
            user_id: User identifier
            node_source: Source node (should be "tool_node")

        Returns:
            User's choice: "quick" | "comprehensive" | "background"
            On timeout: returns recommendation
        """
        api_logger.info(f"🔔 [HIL] Execution choice scenario - {scenario}")

        # Extract data fields (from BackgroundHILDetector)
        prompt = data.get("prompt", "How do you want to execute this task?")
        estimated_time = data.get("estimated_time_seconds", 0)
        tool_count = data.get("tool_count", 0)
        task_type = data.get("task_type", "unknown")
        recommendation = data.get("recommendation", "comprehensive")
        options = data.get("options", [])
        context = data.get("context", {})

        api_logger.info(
            f"📋 [HIL] Creating ExecutionChoiceRequest: "
            f"time={estimated_time:.1f}s, tools={tool_count}, "
            f"type={task_type}, rec={recommendation}"
        )

        # Create execution choice request model
        exec_req = ExecutionChoiceRequest(
            prompt=prompt,
            estimated_time_seconds=estimated_time,
            tool_count=tool_count,
            task_type=task_type,
            recommendation=recommendation,
            options=options,
            scenario=scenario,
            context=context
        )

        # Convert to interrupt data
        interrupt_data = exec_req.to_interrupt_data()
        interrupt_data["user_id"] = user_id
        interrupt_data["node_source"] = node_source
        interrupt_data["id"] = str(uuid.uuid4())

        api_logger.info(f"🚨 [HIL] Triggering interrupt with type='{interrupt_data['type']}'")

        # Log interrupt
        self.interrupt_manager._log_interrupt(interrupt_data)

        # Get timeout (30 seconds for quick choice)
        timeout = self.timeout_handler.get_timeout("execution_choice", data.get("timeout"))
        api_logger.info(f"🔔 [HIL] Execution choice: {scenario} (timeout={timeout}s)")

        # Trigger interrupt with timeout (default to recommendation)
        human_response = await self.timeout_handler.with_timeout(
            coro=self._trigger_interrupt(interrupt_data),
            timeout=timeout,
            scenario=scenario,
            default_value=recommendation  # Use recommendation on timeout
        )

        # Parse response
        if isinstance(human_response, dict):
            choice = human_response.get("choice", human_response.get("value", recommendation))
        else:
            # Simple string response
            choice = str(human_response).lower().strip()

        # Validate choice
        valid_choices = ["quick", "comprehensive", "background", "q", "c", "b", "bg"]
        if choice in ["q", "quick"]:
            choice = "quick"
        elif choice in ["c", "comprehensive"]:
            choice = "comprehensive"
        elif choice in ["b", "bg", "background"]:
            choice = "background"
        elif choice not in valid_choices:
            # Default to recommendation if invalid
            api_logger.warning(f"Invalid choice '{choice}', using recommendation '{recommendation}'")
            choice = recommendation

        api_logger.info(f"✅ [HIL] Execution choice {scenario} - {choice} (estimated {estimated_time:.0f}s for {tool_count} tools)")

        return choice

