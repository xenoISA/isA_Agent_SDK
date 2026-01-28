"""
Agent Lightning Store 配置和初始化

使用官方的 SqliteLightningStore 来收集训练数据
"""
import os
from pathlib import Path
from typing import Optional

try:
    from agentlightning import SqliteLightningStore
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    SqliteLightningStore = None


class LightningStoreManager:
    """管理 Agent Lightning Store 的生命周期"""

    def __init__(self):
        self.store: Optional[SqliteLightningStore] = None
        self.enabled = False
        self.store_path = None

    def initialize(self) -> bool:
        """初始化 Lightning Store"""
        if not LIGHTNING_AVAILABLE:
            print("⚠️  agentlightning package not available")
            return False

        # 检查是否启用
        enabled = os.getenv("AGENT_LIGHTNING_ENABLED", "false").lower() == "true"
        if not enabled:
            print("ℹ️  Agent Lightning is disabled (AGENT_LIGHTNING_ENABLED=false)")
            return False

        try:
            # 设置 SQLite 数据库路径
            storage_dir = Path(os.getenv("LIGHTNING_STORAGE_DIR", "/app/storage/lightning"))
            storage_dir.mkdir(parents=True, exist_ok=True)

            self.store_path = storage_dir / "training_data.db"

            # 创建 SqliteLightningStore
            self.store = SqliteLightningStore(str(self.store_path))

            self.enabled = True
            print(f"✅ Agent Lightning Store initialized at {self.store_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize Agent Lightning Store: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_store(self) -> Optional[SqliteLightningStore]:
        """获取 Store 实例"""
        return self.store if self.enabled else None

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.enabled

    def close(self):
        """关闭 Store"""
        if self.store:
            try:
                # SqliteLightningStore 可能没有显式的 close 方法
                # 但我们可以清理引用
                self.store = None
                print("✅ Agent Lightning Store closed")
            except Exception as e:
                print(f"⚠️  Error closing Lightning Store: {e}")


# 全局单例
_store_manager: Optional[LightningStoreManager] = None


def get_lightning_store_manager() -> LightningStoreManager:
    """获取全局 Store Manager 单例"""
    global _store_manager
    if _store_manager is None:
        _store_manager = LightningStoreManager()
    return _store_manager


def initialize_lightning_store() -> bool:
    """初始化全局 Lightning Store"""
    manager = get_lightning_store_manager()
    return manager.initialize()


def get_lightning_store() -> Optional[SqliteLightningStore]:
    """获取 Lightning Store 实例"""
    manager = get_lightning_store_manager()
    return manager.get_store()


def is_lightning_enabled() -> bool:
    """检查 Lightning 是否启用"""
    manager = get_lightning_store_manager()
    return manager.is_enabled()


def close_lightning_store():
    """关闭 Lightning Store"""
    manager = get_lightning_store_manager()
    manager.close()
