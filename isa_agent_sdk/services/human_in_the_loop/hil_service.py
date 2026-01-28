"""
Human-in-the-Loop Service (Refactored)
File: app/services/human_in_the_loop/hil_service_refactored.py

Professional implementation based on LangGraph v1.0+ best practices.
This is the main orchestrating service that brings together all HIL components.

Example Usage:
    from isa_agent_sdk.services.human_in_the_loop import get_hil_service
    
    # Get service instance
    hil = get_hil_service()
    
    # Collect user input
    email = await hil.collect_user_input("What is your email?")
    
    # Request tool permission
    authorized = await hil.request_tool_permission(
        tool_name="web_crawl",
        tool_args={"url": "https://example.com"},
        security_level="HIGH",
        reason="Need to scrape data"
    )
    
    # Get interrupt statistics
    stats = hil.get_interrupt_stats()
"""

from typing import Dict, Any, Optional, List
from langgraph.types import Command

from .interrupt_manager import InterruptManager
from .scenario_handlers import ScenarioHandler
from .validators import HILValidator
from .models import (
    InterruptStats,
    AuthorizationRequest,
    InputRequest,
    ReviewRequest,
    InputWithAuthRequest
)


class HILService:
    """
    Human-in-the-Loop service with durable execution support.
    
    This service provides a clean interface for all HIL operations:
    
    CORE INTERRUPT PATTERNS:
    =======================
    - approve_or_reject()         : Approval/rejection workflow
    - review_and_edit()            : Review and edit content
    - validate_input_with_retry()  : Input validation with retry
    - simple_interrupt()           : Generic interrupt
    
    SCENARIO-BASED METHODS (Recommended):
    ====================================
    - collect_user_input()           : Collect information from user
    - request_tool_permission()      : Request tool execution authorization
    - request_oauth_authorization()  : Request OAuth app authorization (Composio)
    - request_credential_usage()     : Request to use stored credentials (Vault)
    - request_manual_intervention()  : Request manual user action (CAPTCHA, login, etc.)
    
    LEGACY METHODS (Deprecated):
    ===========================
    - ask_human_via_mcp_with_interrupt() -> use collect_user_input()
    - request_tool_authorization()       -> use request_tool_permission()
    """
    
    def __init__(self, mcp_client=None):
        """
        Initialize HIL service
        
        Args:
            mcp_client: Optional MCP client for logging/tracking
            
        Example:
            # Without MCP
            hil = HILService()
            
            # With MCP
            from isa_agent_sdk.clients.mcp_client import mcp_client
            hil = HILService(mcp_client=mcp_client)
        """
        self.mcp_client = mcp_client
        self.interrupt_manager = InterruptManager()
        self.scenario_handler = ScenarioHandler(self.interrupt_manager, mcp_client)
        self.validator = HILValidator()
    
    # ========== Core Interrupt Patterns ==========
    
    def approve_or_reject(
        self,
        question: str,
        context: Dict[str, Any],
        node_source: str = "unknown",
        approval_options: Optional[List[str]] = None
    ) -> Command:
        """
        Approve/reject pattern - returns Command for routing
        
        See interrupt_manager.approve_or_reject() for details
        """
        return self.interrupt_manager.approve_or_reject(
            question,
            context,
            node_source,
            approval_options
        )
    
    def review_and_edit(
        self,
        content_to_review: str,
        task_description: str,
        node_source: str = "unknown",
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Review and edit pattern - returns edited content
        
        See interrupt_manager.review_and_edit() for details
        """
        return self.interrupt_manager.review_and_edit(
            content_to_review,
            task_description,
            node_source,
            required_fields
        )
    
    def validate_input_with_retry(
        self,
        initial_question: str,
        validation_rules: Dict[str, Any],
        max_retries: int = 3,
        node_source: str = "unknown"
    ) -> Any:
        """
        Input validation with retry pattern
        
        See interrupt_manager.validate_input_with_retry() for details
        """
        return self.interrupt_manager.validate_input_with_retry(
            initial_question,
            validation_rules,
            max_retries,
            node_source
        )
    
    def simple_interrupt(
        self,
        question: str,
        context: str = "",
        user_id: str = "default",
        node_source: str = "unknown"
    ) -> Any:
        """
        Simple interrupt that pauses execution

        See interrupt_manager.simple_interrupt() for details
        """
        return self.interrupt_manager.simple_interrupt(
            question,
            context,
            user_id,
            node_source
        )

    # ========== 4 Core HIL Methods (MCP-Aligned) ==========

    async def request_authorization(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> bool:
        """
        Core HIL Method 1: Request Authorization

        Routes authorization requests to appropriate scenario handlers based on scenario type.

        Supported Scenarios:
            - tool_authorization: High/Critical security tools
            - payment: Financial transactions
            - deletion: Destructive operations
            - deployment: Production deployments
            - generic: General authorization

        Args:
            scenario: Scenario type (tool_authorization, payment, deletion, deployment, generic)
            data: MCP HIL data containing authorization details
            user_id: User identifier
            node_source: Source node

        Returns:
            True if authorized, False if rejected

        Example:
            # From MCP tool response
            authorized = await hil_service.request_authorization(
                scenario="payment",
                data={
                    "action": "Process $5000 payment",
                    "reason": "Complete vendor payment",
                    "risk_level": "high",
                    "context": {"vendor": "Acme", "amount": 5000}
                }
            )
        """
        return await self.scenario_handler.handle_authorization_scenario(
            scenario, data, user_id, node_source
        )

    async def request_input(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> Any:
        """
        Core HIL Method 2: Request Input

        Routes input requests to appropriate scenario handlers based on scenario type.

        Supported Scenarios:
            - credentials: API keys, passwords
            - selection: Choose from options
            - augmentation: Add missing details
            - generic: General input

        Args:
            scenario: Scenario type (credentials, selection, augmentation, generic)
            data: MCP HIL data containing input request details
            user_id: User identifier
            node_source: Source node

        Returns:
            User input (validated if schema provided)

        Example:
            # From MCP tool response
            api_key = await hil_service.request_input(
                scenario="credentials",
                data={
                    "input_type": "credentials",
                    "prompt": "Enter OpenAI API Key",
                    "description": "Provide your API key",
                    "schema": {"type": "string", "pattern": "^sk-"}
                }
            )
        """
        return await self.scenario_handler.handle_input_scenario(
            scenario, data, user_id, node_source
        )

    async def request_review(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> Dict[str, Any]:
        """
        Core HIL Method 3: Request Review

        Routes review requests to appropriate scenario handlers based on scenario type.

        Supported Scenarios:
            - execution_plan: Review task execution plan
            - code: Review generated code
            - config: Review configuration
            - generic: General review

        Args:
            scenario: Scenario type (execution_plan, code, config, generic)
            data: MCP HIL data containing review details
            user_id: User identifier
            node_source: Source node

        Returns:
            Dict with 'approved' (bool), 'edited_content' (if edited), 'action' (str)

        Example:
            # From MCP tool response
            result = await hil_service.request_review(
                scenario="execution_plan",
                data={
                    "content": execution_plan,
                    "content_type": "execution_plan",
                    "instructions": "Review before execution",
                    "editable": True
                }
            )
        """
        return await self.scenario_handler.handle_review_scenario(
            scenario, data, user_id, node_source
        )

    async def request_input_with_authorization(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> Dict[str, Any]:
        """
        Core HIL Method 4: Request Input + Authorization

        Routes combined input+authorization requests to appropriate scenario handlers.

        Supported Scenarios:
            - payment_with_amount: Enter amount + authorize payment
            - deploy_with_config: Enter config + authorize deployment
            - generic: General input + authorization

        Args:
            scenario: Scenario type (payment_with_amount, deploy_with_config, generic)
            data: MCP HIL data containing input and authorization details
            user_id: User identifier
            node_source: Source node

        Returns:
            Dict with 'user_input' (Any), 'approved' (bool), 'action' (str)

        Example:
            # From MCP tool response
            result = await hil_service.request_input_with_authorization(
                scenario="payment_with_amount",
                data={
                    "input_prompt": "Enter payment amount",
                    "input_description": "Specify amount",
                    "authorization_reason": "Authorize payment",
                    "input_type": "number",
                    "risk_level": "high",
                    "schema": {"type": "number", "minimum": 0}
                }
            )
        """
        return await self.scenario_handler.handle_input_with_auth_scenario(
            scenario, data, user_id, node_source
        )

    async def request_execution_choice(
        self,
        scenario: str,
        data: Dict[str, Any],
        user_id: str = "default",
        node_source: str = "tool_node"
    ) -> str:
        """
        Core HIL Method 5: Request Execution Choice

        Routes execution choice requests to appropriate scenario handlers.
        This is triggered by ToolNode when it detects long-running tasks.

        Supported Scenarios:
            - long_running_task: Generic long-running task
            - long_running_web_crawl: Multiple web pages to crawl
            - long_running_web_search: Multiple searches
            - long_running_mixed: Mixed operations

        Args:
            scenario: Scenario type (long_running_task, long_running_web_crawl, etc.)
            data: Execution choice data from BackgroundHILDetector containing:
                - prompt: User-facing prompt string
                - estimated_time_seconds: Estimated execution time
                - tool_count: Number of tools to execute
                - task_type: Type of task (web_crawling, web_searching, mixed)
                - recommendation: Recommended choice (quick, comprehensive, background)
                - options: List of available options with descriptions
                - context: Additional context data
            user_id: User identifier
            node_source: Source node (should be "tool_node")

        Returns:
            User's choice: "quick" | "comprehensive" | "background"

        Example:
            # From ToolNode when long-running task detected
            choice = await hil_service.request_execution_choice(
                scenario="long_running_web_crawl",
                data={
                    "prompt": "Long-running task detected: crawling 5 web pages (~60s total)",
                    "estimated_time_seconds": 60.0,
                    "tool_count": 5,
                    "task_type": "web_crawling",
                    "recommendation": "background",
                    "options": [
                        {"value": "quick", "label": "Quick", "description": "Fast response (~30s, 3 sources)"},
                        {"value": "comprehensive", "label": "Comprehensive", "description": "Wait for all 5 sources (~60s)"},
                        {"value": "background", "label": "Background", "description": "Run in background, get job_id immediately"}
                    ],
                    "context": {"tool_breakdown": {"web_crawl": 5}}
                },
                user_id="user_123",
                node_source="tool_node"
            )
            # Returns: "quick" | "comprehensive" | "background"
        """
        return await self.scenario_handler.handle_execution_choice_scenario(
            scenario, data, user_id, node_source
        )

    # ========== Utility Methods ==========
    
    def resume_multiple_interrupts(
        self,
        interrupt_responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resume multiple interrupts with single invocation
        
        See interrupt_manager.resume_multiple_interrupts() for details
        """
        return self.interrupt_manager.resume_multiple_interrupts(interrupt_responses)
    
    def get_interrupt_stats(self) -> InterruptStats:
        """
        Get statistics about interrupts
        
        Returns:
            InterruptStats object with total, by_type, by_node, latest
            
        Example:
            stats = hil.get_interrupt_stats()
            print(f"Total interrupts: {stats.total}")
            print(f"By type: {stats.by_type}")
            print(f"By node: {stats.by_node}")
            print(f"Latest: {stats.latest}")
        """
        return self.interrupt_manager.get_interrupt_stats()
    
    def clear_interrupt_history(self):
        """Clear interrupt history"""
        self.interrupt_manager.clear_history()
    
    @property
    def interrupt_history(self) -> List[Dict[str, Any]]:
        """Get interrupt history"""
        return self.interrupt_manager.interrupt_history


# ========== Global Service Instance ==========

_hil_service_instance: Optional[HILService] = None


def get_hil_service(mcp_client=None) -> HILService:
    """
    Get or create HIL service singleton instance
    
    Args:
        mcp_client: Optional MCP client for logging/tracking
        
    Returns:
        HILService instance
        
    Example:
        # Get default instance
        hil = get_hil_service()
        
        # Get instance with MCP client
        from isa_agent_sdk.clients.mcp_client import mcp_client
        hil = get_hil_service(mcp_client=mcp_client)
    """
    global _hil_service_instance
    
    if _hil_service_instance is None:
        _hil_service_instance = HILService(mcp_client)
    elif mcp_client and not _hil_service_instance.mcp_client:
        _hil_service_instance.mcp_client = mcp_client
        _hil_service_instance.scenario_handler.mcp_client = mcp_client
    
    return _hil_service_instance


def reset_hil_service():
    """
    Reset the global HIL service instance (useful for testing)
    
    Example:
        # In test teardown
        reset_hil_service()
    """
    global _hil_service_instance
    _hil_service_instance = None

