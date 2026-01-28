#!/usr/bin/env python3
"""
Tool HIL Router - 路由HIL请求到服务层

简单路由器，将检测到的HIL请求路由到对应的HIL服务4个核心方法。

Auto-Approve Feature:
    Similar to Claude Code's `-y` flag, supports auto-approving certain HIL types:
    - Environment variable: ISA_AUTO_APPROVE_PLANS=true
    - Supported auto-approve scenarios: execution_plan reviews
"""

import os
from typing import Any, Dict, Optional
from datetime import datetime
from isa_agent_sdk.services.human_in_the_loop import get_hil_service
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger

# Auto-approve configuration from environment
AUTO_APPROVE_PLANS = os.getenv("ISA_AUTO_APPROVE_PLANS", "false").lower() == "true"
AUTO_APPROVE_CODE = os.getenv("ISA_AUTO_APPROVE_CODE", "false").lower() == "true"


class ToolHILRouter:
    """
    路由HIL请求到HIL服务的4个核心方法

    Auto-Approve Support:
        Set ISA_AUTO_APPROVE_PLANS=true to automatically approve execution plans.
        This is similar to Claude Code's --dangerously-skip-permissions flag.

        Example:
            export ISA_AUTO_APPROVE_PLANS=true
            # Or in Python before agent runs:
            os.environ["ISA_AUTO_APPROVE_PLANS"] = "true"
    """

    def __init__(self, auto_approve_plans: Optional[bool] = None):
        """
        Initialize HIL Router

        Args:
            auto_approve_plans: Override for auto-approve plans setting.
                              If None, uses ISA_AUTO_APPROVE_PLANS env var.
        """
        self.hil_service = get_hil_service()

        # Allow programmatic override of auto-approve settings
        self._auto_approve_plans = auto_approve_plans if auto_approve_plans is not None else AUTO_APPROVE_PLANS
        self._auto_approve_code = AUTO_APPROVE_CODE

        if self._auto_approve_plans:
            logger.warning("⚠️ [HILRouter] AUTO-APPROVE PLANS ENABLED - Plans will execute without human review!")

    async def route(self, hil_type: str, hil_data: Dict[str, Any], tool_name: str, tool_args: dict) -> Any:
        """
        根据HIL类型路由到对应的4个核心服务方法

        Args:
            hil_type: HIL类型 (authorization/input/review/input_with_authorization)
            hil_data: 提取的HIL数据
            tool_name: 触发HIL的工具名称
            tool_args: 工具参数

        Returns:
            用户响应
        """
        logger.info(f"[HILRouter] Routing HIL type: {hil_type} from tool: {tool_name}")

        try:
            # 识别具体场景
            scenario = self._identify_scenario(hil_type, hil_data, tool_name)
            user_id = hil_data.get("context", {}).get("user_id", "default")

            # Check for auto-approve before routing to HIL service
            if self._should_auto_approve(hil_type, scenario, hil_data):
                logger.info(f"🚀 [HILRouter] AUTO-APPROVING {hil_type}/{scenario} for {tool_name}")
                return self._create_auto_approve_response(hil_type, scenario, hil_data)

            # 路由到4个核心方法
            if hil_type == "authorization":
                return await self.hil_service.request_authorization(
                    scenario=scenario,
                    data=hil_data.get("data", {}),
                    user_id=user_id,
                    node_source="tool_node"
                )

            elif hil_type == "input":
                return await self.hil_service.request_input(
                    scenario=scenario,
                    data=hil_data.get("data", {}),
                    user_id=user_id,
                    node_source="tool_node"
                )

            elif hil_type == "review":
                return await self.hil_service.request_review(
                    scenario=scenario,
                    data=hil_data.get("data", {}),
                    user_id=user_id,
                    node_source="tool_node"
                )

            elif hil_type == "input_with_authorization":
                return await self.hil_service.request_input_with_authorization(
                    scenario=scenario,
                    data=hil_data.get("data", {}),
                    user_id=user_id,
                    node_source="tool_node"
                )

            else:
                logger.warning(f"[HILRouter] Unknown HIL type: {hil_type}, falling back to generic")
                # Fallback to simple interrupt
                return await self.hil_service.simple_interrupt(
                    question=hil_data.get("question", "Human input required"),
                    context=hil_data.get("message", ""),
                    user_id=user_id,
                    node_source="tool_node"
                )

        except Exception as e:
            logger.error(f"[HILRouter] Route failed: {e}", exc_info=True)
            raise

    def _identify_scenario(self, hil_type: str, hil_data: Dict, tool_name: str) -> str:
        """
        从MCP数据识别具体场景

        Args:
            hil_type: HIL类型 (authorization/input/review/input_with_authorization)
            hil_data: MCP HIL数据
            tool_name: 工具名称

        Returns:
            场景标识符

        场景映射规则：
            Authorization:
                - request_type="authorization" -> "tool_authorization"
                - request_type="oauth_authorization" -> "oauth_authorization"
                - 其他 -> "generic"

            Input:
                - input_type="credentials" -> "credentials"
                - input_type="selection" -> "selection"
                - input_type="augmentation" -> "augmentation"
                - 其他 -> "generic"

            Review:
                - content_type="execution_plan" -> "execution_plan"
                - content_type="code" -> "code"
                - content_type="config" -> "config"
                - 其他 -> "generic"

            Input + Authorization:
                - Based on context or default to "generic"
        """
        data = hil_data.get("data", {})

        # Authorization scenarios
        if hil_type == "authorization":
            request_type = data.get("request_type")

            if request_type == "authorization":
                # Tool authorization scenario
                return "tool_authorization"
            elif request_type == "oauth_authorization":
                # OAuth authorization scenario
                return "oauth_authorization"
            else:
                # Generic authorization
                return "generic"

        # Input scenarios
        elif hil_type == "input":
            input_type = data.get("input_type", "text")

            if input_type == "credentials":
                return "credentials"
            elif input_type == "selection":
                return "selection"
            elif input_type == "augmentation":
                return "augmentation"
            else:
                return "generic"

        # Review scenarios
        elif hil_type == "review":
            content_type = data.get("content_type", "text")

            if content_type == "execution_plan":
                return "execution_plan"
            elif content_type == "code":
                return "code"
            elif content_type == "config":
                return "config"
            else:
                return "generic"

        # Input + Authorization scenarios
        elif hil_type == "input_with_authorization":
            # Check context for scenario hints
            context = data.get("context", {})

            # Payment scenario
            if "payment" in str(context).lower() or "amount" in data:
                return "payment_with_amount"

            # Deployment scenario
            if "deploy" in str(context).lower() or "config" in data:
                return "deploy_with_config"

            # Generic
            return "generic"

        # Unknown HIL type
        else:
            logger.warning(f"[HILRouter] Unknown HIL type for scenario identification: {hil_type}")
            return "generic"

    def _should_auto_approve(self, hil_type: str, scenario: str, hil_data: Dict[str, Any]) -> bool:
        """
        Check if this HIL request should be auto-approved

        Auto-approve is supported for:
        - execution_plan reviews (when ISA_AUTO_APPROVE_PLANS=true)
        - code reviews (when ISA_AUTO_APPROVE_CODE=true)

        Args:
            hil_type: HIL type (review, authorization, etc.)
            scenario: Identified scenario (execution_plan, code, etc.)
            hil_data: HIL data from MCP tool

        Returns:
            True if should auto-approve, False otherwise
        """
        # Only auto-approve review types for now
        if hil_type != "review":
            return False

        # Auto-approve execution plans
        if scenario == "execution_plan" and self._auto_approve_plans:
            logger.info(f"[HILRouter] Auto-approve enabled for execution_plan reviews")
            return True

        # Auto-approve code reviews
        if scenario == "code" and self._auto_approve_code:
            logger.info(f"[HILRouter] Auto-approve enabled for code reviews")
            return True

        return False

    def _create_auto_approve_response(self, hil_type: str, scenario: str, hil_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an auto-approve response that mimics human approval

        This response is what would be returned if a human clicked "approve".

        Args:
            hil_type: HIL type
            scenario: Identified scenario
            hil_data: Original HIL data from MCP tool

        Returns:
            Auto-approval response dict
        """
        data = hil_data.get("data", {})

        if scenario == "execution_plan":
            # For execution plans, return the plan content as approved
            plan_content = data.get("content", {})

            response = {
                "action": "approve",
                "approved": True,
                "auto_approved": True,
                "scenario": scenario,
                "timestamp": datetime.now().isoformat(),
                "content": plan_content,
                "message": "Auto-approved: ISA_AUTO_APPROVE_PLANS=true"
            }

            # Log auto-approval details
            plan_id = plan_content.get("plan_id", "unknown") if isinstance(plan_content, dict) else "unknown"
            task_count = plan_content.get("total_tasks", 0) if isinstance(plan_content, dict) else 0
            logger.info(
                f"✅ [HILRouter] AUTO-APPROVED execution plan | "
                f"plan_id={plan_id} | "
                f"task_count={task_count}"
            )

            return response

        elif scenario == "code":
            # For code reviews, return the code as approved
            code_content = data.get("content", "")

            response = {
                "action": "approve",
                "approved": True,
                "auto_approved": True,
                "scenario": scenario,
                "timestamp": datetime.now().isoformat(),
                "content": code_content,
                "message": "Auto-approved: ISA_AUTO_APPROVE_CODE=true"
            }

            logger.info(f"✅ [HILRouter] AUTO-APPROVED code review")
            return response

        else:
            # Generic auto-approve response
            return {
                "action": "approve",
                "approved": True,
                "auto_approved": True,
                "scenario": scenario,
                "timestamp": datetime.now().isoformat(),
                "content": data.get("content"),
                "message": f"Auto-approved: {scenario}"
            }

    def set_auto_approve_plans(self, enabled: bool):
        """
        Programmatically enable/disable auto-approve for plans

        Args:
            enabled: True to enable auto-approve, False to disable

        Example:
            router = ToolHILRouter()
            router.set_auto_approve_plans(True)  # Enable auto-approve
        """
        self._auto_approve_plans = enabled
        if enabled:
            logger.warning("⚠️ [HILRouter] AUTO-APPROVE PLANS ENABLED")
        else:
            logger.info("✅ [HILRouter] AUTO-APPROVE PLANS DISABLED")
