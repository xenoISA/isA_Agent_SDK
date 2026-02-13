#!/usr/bin/env python3
"""Modular configuration system for Agent

Configuration hierarchy:
- infra_config: Infrastructure services from isA_Cloud (PostgreSQL, Redis, Qdrant, etc.)
- service_config: External business services from isA_User + peer ISA services
- model_config: LLM/embedding models from isA_Model
- agent_config: Agent-specific settings (server, auth, resources)
- consul_config: Service discovery settings
- logging_config: Logging configuration
- lightning_config: Agent Lightning (RL training) configuration

Environment loading is delegated to ConfigManager from isA_user/core,
which follows the platform-wide convention:
  1. Environment variables (highest priority)
  2. deployment/environments/{env}.env
  3. config/*.json files
  4. Defaults
"""
import os
import logging
from .logging_config import LoggingConfig
from .infra_config import InfraConfig
from .consul_config import consul_config, ConsulConfig
from .service_config import ServiceConfig
from .model_config import ModelConfig
from .lightning_config import LightningConfig
from .agent_config import (
    AgentConfig,
    AgentResourceConfig,
    AgentQdrantConfig,
    AgentMinIOConfig,
    AgentRedisConfig,
    AgentPostgresConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load environment using the platform-wide ConfigManager (isA_user/core)
# ---------------------------------------------------------------------------
try:
    from isA_user.core.config_manager import ConfigManager

    _config_manager = ConfigManager("agent_service")
    logger.info(
        "ConfigManager loaded env=%s POSTGRES_DB=%s",
        _config_manager.environment.value,
        os.getenv("POSTGRES_DB"),
    )
except ImportError:
    # Fallback: load env file directly (e.g. in SDK-only test environments)
    from dotenv import load_dotenv

    _env_files = {
        "dev": "deployment/environments/dev.env",
        "development": "deployment/environments/dev.env",
        "test": "deployment/environments/test.env",
        "staging": "deployment/environments/staging.env",
        "production": "deployment/environments/production.env",
    }
    _env = os.getenv("ENVIRONMENT", os.getenv("ENV", "dev")).lower()
    _env_file = _env_files.get(_env, "deployment/environments/dev.env")
    load_dotenv(_env_file, override=False)
    logger.info("ConfigManager unavailable, loaded %s directly", _env_file)
    _config_manager = None

# Create global settings instance
settings = AgentConfig.from_env()

def get_settings() -> AgentConfig:
    """Get global settings instance"""
    return settings

def get_config_manager():
    """Get the ConfigManager instance (None if unavailable)"""
    return _config_manager

def reload_settings() -> AgentConfig:
    """Reload settings from environment"""
    global settings
    settings = AgentConfig.from_env()
    return settings

# Backward compatibility
def get_openai_api_key():
    """Get OpenAI API key (backward compatibility)"""
    from pydantic import SecretStr
    return SecretStr(settings.openai_api_key) if settings.openai_api_key else None

__all__ = [
    # Main config
    'AgentConfig',
    'get_settings',
    'get_config_manager',
    'reload_settings',
    'settings',
    'get_openai_api_key',
    # Sub-configs
    'LoggingConfig',
    'InfraConfig',
    'ConsulConfig',
    'consul_config',
    'ServiceConfig',
    'ModelConfig',
    'LightningConfig',
    # Agent resource configs
    'AgentResourceConfig',
    'AgentQdrantConfig',
    'AgentMinIOConfig',
    'AgentRedisConfig',
    'AgentPostgresConfig',
]
