#!/usr/bin/env python3
"""Agent Lightning configuration for RL-based training"""

import os
from dataclasses import dataclass


def _bool(val: str) -> bool:
    """Convert string to boolean"""
    return val.lower() in ("true", "1", "yes")


def _int(val: str, default: int) -> int:
    """Convert string to int with default"""
    try:
        return int(val) if val else default
    except ValueError:
        return default


@dataclass
class LightningConfig:
    """Agent Lightning training configuration"""

    # Enable/Disable Agent Lightning
    enabled: bool = False

    # Training mode: 'api' (lightweight) or 'full' (open-source models)
    mode: str = "api"

    # Storage backend type: 'redis', 'postgres', 'memory'
    store_type: str = "redis"
    store_url: str = "redis://localhost:6379/2"

    # Data collection toggles
    collect_prompts: bool = True
    collect_tool_calls: bool = True
    collect_rewards: bool = True

    # Training parameters
    batch_size: int = 32
    checkpoint_dir: str = "/app/storage/lightning_checkpoints"

    # Session/trajectory tracking
    enable_trajectory_tracking: bool = True

    @classmethod
    def from_env(cls) -> "LightningConfig":
        """Load Lightning configuration from environment variables"""
        # Get Redis configuration from standard infra config
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")

        # Build default Redis URL if not explicitly set
        default_redis_url = f"redis://{redis_host}:{redis_port}/2"

        return cls(
            enabled=_bool(os.getenv("AGENT_LIGHTNING_ENABLED", "false")),
            mode=os.getenv("AGENT_LIGHTNING_MODE", "api"),
            store_type=os.getenv("AGENT_LIGHTNING_STORE_TYPE", "redis"),
            store_url=os.getenv("AGENT_LIGHTNING_STORE_URL", default_redis_url),
            collect_prompts=_bool(os.getenv("AGENT_LIGHTNING_COLLECT_PROMPTS", "true")),
            collect_tool_calls=_bool(
                os.getenv("AGENT_LIGHTNING_COLLECT_TOOL_CALLS", "true")
            ),
            collect_rewards=_bool(os.getenv("AGENT_LIGHTNING_COLLECT_REWARDS", "true")),
            batch_size=_int(os.getenv("AGENT_LIGHTNING_BATCH_SIZE", "32"), 32),
            checkpoint_dir=os.getenv(
                "AGENT_LIGHTNING_CHECKPOINT_DIR", "/app/storage/lightning_checkpoints"
            ),
            enable_trajectory_tracking=_bool(
                os.getenv("AGENT_LIGHTNING_TRAJECTORY_TRACKING", "true")
            ),
        )

    @property
    def is_enabled(self) -> bool:
        """Check if Agent Lightning is enabled"""
        return self.enabled

    @property
    def is_api_mode(self) -> bool:
        """Check if running in API mode (lightweight)"""
        return self.mode == "api"

    @property
    def is_full_mode(self) -> bool:
        """Check if running in full training mode"""
        return self.mode == "full"
