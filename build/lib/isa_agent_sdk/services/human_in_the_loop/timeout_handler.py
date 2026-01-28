"""
Timeout handler for HIL interrupts
File: app/services/human_in_the_loop/timeout_handler.py

Simple timeout handling with default behaviors for Human-in-the-Loop scenarios.

Example:
    handler = TimeoutHandler()

    # Get timeout for a scenario
    timeout = handler.get_timeout("authorization")

    # Execute with timeout
    result = await handler.with_timeout(
        coro=some_async_function(),
        timeout=timeout,
        scenario="authorization",
        default_value=False
    )
"""

import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable
from datetime import datetime
from isa_agent_sdk.utils.logger import api_logger


# 基础超时配置（秒）
# 用户在等待，不应该等太久
TIMEOUT_CONFIG = {
    "execution_choice": 30,    # 30秒 - 快速选择执行模式
    "input": 60,               # 1分钟 - 输入信息（可能需要查找）
    "authorization": 120,      # 2分钟 - 授权审批（需要思考）
    "review": 120,             # 2分钟 - 审查内容
    "input_with_auth": 120,    # 2分钟 - 输入+授权
}


class TimeoutHandler:
    """
    简单的超时处理器

    Features:
    - 为不同 HIL 场景提供合理的超时时间
    - 超时后提供默认行为（安全优先）
    - 记录超时事件用于监控和调试

    Example:
        handler = TimeoutHandler()

        # 获取超时时间
        timeout = handler.get_timeout("authorization", custom_timeout=180)

        # 带超时执行
        result = await handler.with_timeout(
            coro=interrupt_func(),
            timeout=timeout,
            scenario="authorization",
            default_value=False
        )

        # 查看超时统计
        stats = handler.get_timeout_stats()
    """

    def __init__(self):
        """初始化超时处理器"""
        self.timeout_events = []  # 记录超时事件（用于监控）

    def get_timeout(
        self,
        scenario_type: str,
        custom_timeout: Optional[int] = None
    ) -> int:
        """
        获取场景的超时时间

        Args:
            scenario_type: 场景类型（execution_choice, input, authorization, review, input_with_auth）
            custom_timeout: 用户自定义超时（优先使用）

        Returns:
            超时秒数

        Example:
            timeout = handler.get_timeout("authorization")  # 返回 120
            timeout = handler.get_timeout("input", 90)      # 返回 90（使用自定义）
        """
        # 用户自定义超时优先
        if custom_timeout and custom_timeout > 0:
            api_logger.debug(f"Using custom timeout: {custom_timeout}s for {scenario_type}")
            return custom_timeout

        # 使用默认配置
        default_timeout = TIMEOUT_CONFIG.get(scenario_type, 60)  # 默认60秒
        api_logger.debug(f"Using default timeout: {default_timeout}s for {scenario_type}")
        return default_timeout

    async def with_timeout(
        self,
        coro: Awaitable[Any],
        timeout: int,
        scenario: str,
        default_value: Any = None
    ) -> Any:
        """
        执行带超时的协程

        Args:
            coro: 要执行的协程（例如 interrupt()）
            timeout: 超时时间（秒）
            scenario: 场景名称（用于日志）
            default_value: 超时后的默认返回值

        Returns:
            协程结果或默认值（如果超时）

        Example:
            result = await handler.with_timeout(
                coro=interrupt(interrupt_data),
                timeout=120,
                scenario="authorization",
                default_value=False
            )
        """
        try:
            # 执行带超时的协程
            result = await asyncio.wait_for(coro, timeout=timeout)
            api_logger.debug(f"✅ [HIL] {scenario} completed within {timeout}s")
            return result

        except asyncio.TimeoutError:
            # 记录超时事件
            self._log_timeout(scenario, timeout)

            api_logger.warning(
                f"⏱️ [HIL] Timeout after {timeout}s for scenario '{scenario}' "
                f"- using default behavior"
            )

            return default_value

    def get_default_behavior(
        self,
        scenario_type: str,
        data: Dict[str, Any]
    ) -> Any:
        """
        获取超时后的默认行为

        根据不同场景类型，返回合理的默认值：
        - execution_choice: 使用推荐选项
        - input: 使用默认值或 None
        - authorization: 拒绝（安全优先）
        - review: 批准原内容
        - input_with_auth: 拒绝（安全优先）

        Args:
            scenario_type: 场景类型
            data: 场景数据（包含默认值等信息）

        Returns:
            默认行为结果

        Example:
            default = handler.get_default_behavior("authorization", data)
            # 返回 False（拒绝）
        """
        if scenario_type == "execution_choice":
            # 使用推荐选项（quick/comprehensive/background）
            recommendation = data.get("recommendation", "comprehensive")
            api_logger.info(f"Timeout default: using recommendation '{recommendation}'")
            return recommendation

        elif scenario_type == "input":
            # 使用默认值或返回 None
            default_value = data.get("default_value")
            if default_value is not None:
                api_logger.info(f"Timeout default: using default_value '{default_value}'")
            else:
                api_logger.warning("Timeout default: no default_value provided, returning None")
            return default_value

        elif scenario_type in ["authorization", "input_with_auth"]:
            # 授权超时 = 拒绝（安全优先）
            api_logger.warning("Timeout default: REJECTED (safety first for authorization)")
            return False

        elif scenario_type == "review":
            # 审查超时 = 批准原内容
            original_content = data.get("content")
            api_logger.info("Timeout default: APPROVED with original content")
            return {
                "approved": True,
                "edited_content": original_content,
                "action": "approve_by_timeout",
                "original_content": original_content
            }

        # 未知场景：返回 None
        api_logger.warning(f"Unknown scenario type '{scenario_type}', returning None")
        return None

    def _log_timeout(self, scenario: str, timeout: int):
        """
        记录超时事件（内部方法）

        Args:
            scenario: 场景名称
            timeout: 超时时间
        """
        event = {
            "scenario": scenario,
            "timeout": timeout,
            "timestamp": datetime.now().isoformat()
        }
        self.timeout_events.append(event)

        # 保留最近100条记录（避免内存泄漏）
        if len(self.timeout_events) > 100:
            self.timeout_events = self.timeout_events[-100:]

    def get_timeout_stats(self) -> Dict[str, Any]:
        """
        获取超时统计信息

        Returns:
            超时统计字典，包含：
            - total: 总超时次数
            - by_scenario: 按场景分组的超时次数
            - recent_events: 最近的超时事件

        Example:
            stats = handler.get_timeout_stats()
            # {
            #     "total": 5,
            #     "by_scenario": {"authorization": 3, "input": 2},
            #     "recent_events": [...]
            # }
        """
        by_scenario = {}
        for event in self.timeout_events:
            scenario = event["scenario"]
            by_scenario[scenario] = by_scenario.get(scenario, 0) + 1

        return {
            "total": len(self.timeout_events),
            "by_scenario": by_scenario,
            "recent_events": self.timeout_events[-10:]  # 最近10条
        }

    def clear_timeout_history(self):
        """清空超时历史（用于测试）"""
        self.timeout_events.clear()
        api_logger.debug("Timeout history cleared")


# 全局单例实例（可选）
_timeout_handler_instance: Optional[TimeoutHandler] = None


def get_timeout_handler() -> TimeoutHandler:
    """
    获取全局超时处理器实例（单例模式）

    Returns:
        TimeoutHandler 实例

    Example:
        handler = get_timeout_handler()
        timeout = handler.get_timeout("authorization")
    """
    global _timeout_handler_instance

    if _timeout_handler_instance is None:
        _timeout_handler_instance = TimeoutHandler()

    return _timeout_handler_instance


def reset_timeout_handler():
    """
    重置全局超时处理器（用于测试）

    Example:
        # 在测试中
        reset_timeout_handler()
    """
    global _timeout_handler_instance
    _timeout_handler_instance = None
