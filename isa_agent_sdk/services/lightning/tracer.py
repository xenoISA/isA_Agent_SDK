"""
Agent Lightning Tracer 集成

使用官方的 Tracer 来自动收集 Agent 执行轨迹
"""
from typing import Optional, Any, Dict
from contextlib import contextmanager

try:
    from agentlightning import Tracer
    TRACER_AVAILABLE = True
except ImportError:
    TRACER_AVAILABLE = False
    Tracer = None

from .lightning_store import get_lightning_store, is_lightning_enabled


class LightningTracer:
    """Lightning Tracer 包装器"""

    def __init__(self):
        self.tracer: Optional[Tracer] = None
        self._current_rollout_id: Optional[str] = None

    def initialize(self) -> bool:
        """初始化 Tracer"""
        if not TRACER_AVAILABLE:
            print("⚠️  Agent Lightning Tracer not available")
            return False

        if not is_lightning_enabled():
            return False

        try:
            store = get_lightning_store()
            if store is None:
                return False

            # 创建 Tracer 实例
            self.tracer = Tracer(store)
            print("✅ Agent Lightning Tracer initialized")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize Lightning Tracer: {e}")
            import traceback
            traceback.print_exc()
            return False

    @contextmanager
    def trace_rollout(self, input_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """
        追踪一个完整的 rollout (对话会话)

        使用方式:
            with tracer.trace_rollout({"user_message": "Hello"}) as rollout_id:
                # 执行 agent
                response = agent.run(message)
        """
        if not self.tracer or not is_lightning_enabled():
            # Lightning 未启用，返回空上下文
            yield None
            return

        try:
            # 开始一个新的 rollout
            rollout_id = self.tracer.start_rollout(
                input=input_data,
                metadata=metadata or {}
            )
            self._current_rollout_id = rollout_id

            yield rollout_id

        except Exception as e:
            print(f"⚠️  Error in trace_rollout: {e}")
            yield None

        finally:
            # 结束 rollout
            if self._current_rollout_id:
                try:
                    self.tracer.end_rollout(self._current_rollout_id)
                except Exception as e:
                    print(f"⚠️  Error ending rollout: {e}")

                self._current_rollout_id = None

    def add_span(self, span_type: str, input_data: Dict[str, Any], output_data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        手动添加一个 span (如果需要)

        Args:
            span_type: 类型，如 'llm_call', 'tool_call', 'action' 等
            input_data: 输入数据
            output_data: 输出数据
            metadata: 元数据
        """
        if not self.tracer or not is_lightning_enabled():
            return

        if not self._current_rollout_id:
            print("⚠️  No active rollout for adding span")
            return

        try:
            self.tracer.add_span(
                rollout_id=self._current_rollout_id,
                span_type=span_type,
                input=input_data,
                output=output_data or {},
                metadata=metadata or {}
            )
        except Exception as e:
            print(f"⚠️  Error adding span: {e}")

    def add_reward(self, rollout_id: str, reward: float, metadata: Optional[Dict[str, Any]] = None):
        """
        为 rollout 添加奖励信号

        Args:
            rollout_id: Rollout ID
            reward: 奖励值 (通常 -1.0 到 1.0)
            metadata: 奖励元数据（如来源、类型等）
        """
        if not self.tracer or not is_lightning_enabled():
            return

        try:
            self.tracer.add_reward(
                rollout_id=rollout_id,
                reward=reward,
                metadata=metadata or {}
            )
            print(f"✅ Added reward {reward} to rollout {rollout_id}")
        except Exception as e:
            print(f"⚠️  Error adding reward: {e}")

    def get_rollout_data(self, rollout_id: str) -> Optional[Dict[str, Any]]:
        """
        获取 rollout 的完整数据

        Args:
            rollout_id: Rollout ID

        Returns:
            Rollout 数据字典，如果不存在返回 None
        """
        if not is_lightning_enabled():
            return None

        try:
            store = get_lightning_store()
            if store:
                return store.get_rollout(rollout_id)
        except Exception as e:
            print(f"⚠️  Error getting rollout data: {e}")

        return None


# 全局单例
_tracer: Optional[LightningTracer] = None


def get_lightning_tracer() -> LightningTracer:
    """获取全局 Tracer 单例"""
    global _tracer
    if _tracer is None:
        _tracer = LightningTracer()
    return _tracer


def initialize_lightning_tracer() -> bool:
    """初始化全局 Lightning Tracer"""
    tracer = get_lightning_tracer()
    return tracer.initialize()
