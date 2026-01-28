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
"""
import os
from dotenv import load_dotenv
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

# Load environment file based on ENVIRONMENT
env = os.getenv("ENVIRONMENT", "dev")
env_files = {
    "dev": "deployment/dev/config/.env.dev",
    "development": "deployment/dev/config/.env.dev",
    "test": "deployment/test/config/.env.test",
    "staging": "deployment/staging/config/.env.staging",
    "production": "deployment/production/.env.production"
}
env_file = env_files.get(env, "deployment/dev/config/.env.dev")
load_dotenv(env_file, override=False)

# Create global settings instance
settings = AgentConfig.from_env()

def get_settings() -> AgentConfig:
    """Get global settings instance"""
    return settings

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
