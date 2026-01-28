#!/usr/bin/env python3
"""
Tool HIL Router - 路由HIL请求到服务层

简单路由器，将检测到的HIL请求路由到对应的HIL服务4个核心方法。
"""

from typing import Any, Dict
from isa_agent_sdk.services.human_in_the_loop import get_hil_service
from isa_agent_sdk.utils.logger import agent_logger

logger = agent_logger


class ToolHILRouter:
    """路由HIL请求到HIL服务的4个核心方法"""

    def __init__(self):
        self.hil_service = get_hil_service()

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
