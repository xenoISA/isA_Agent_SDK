#!/usr/bin/env python3
"""Agent service main configuration"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

from .infra_config import InfraConfig
from .lightning_config import LightningConfig
from .logging_config import LoggingConfig
from .service_config import ServiceConfig
from .model_config import ModelConfig


def _bool(val: str) -> bool:
    return val.lower() == "true"


def _int(val: str, default: int) -> int:
    try:
        return int(val) if val else default
    except ValueError:
        return default


def _float(val: str, default: float) -> float:
    try:
        return float(val) if val else default
    except ValueError:
        return default


# ===========================================
# Agent-Specific Resource Configuration
# ===========================================

@dataclass
class AgentQdrantConfig:
    """Qdrant collection names for Agent"""
    conversation_collection: str = "agent_conversations"
    memory_collection: str = "agent_memories"
    document_collection: str = "agent_documents"

    @property
    def all_collections(self) -> List[str]:
        return [
            self.conversation_collection,
            self.memory_collection,
            self.document_collection,
        ]

    @classmethod
    def from_env(cls) -> 'AgentQdrantConfig':
        prefix = os.getenv("AGENT_QDRANT_PREFIX", "agent")
        return cls(
            conversation_collection=os.getenv("AGENT_QDRANT_CONVERSATIONS", f"{prefix}_conversations"),
            memory_collection=os.getenv("AGENT_QDRANT_MEMORIES", f"{prefix}_memories"),
            document_collection=os.getenv("AGENT_QDRANT_DOCUMENTS", f"{prefix}_documents"),
        )


@dataclass
class AgentMinIOConfig:
    """MinIO bucket names for Agent"""
    attachments_bucket: str = "agent-attachments"
    exports_bucket: str = "agent-exports"

    @property
    def all_buckets(self) -> List[str]:
        return [self.attachments_bucket, self.exports_bucket]

    @classmethod
    def from_env(cls) -> 'AgentMinIOConfig':
        prefix = os.getenv("AGENT_MINIO_PREFIX", "agent")
        return cls(
            attachments_bucket=os.getenv("AGENT_MINIO_ATTACHMENTS", f"{prefix}-attachments"),
            exports_bucket=os.getenv("AGENT_MINIO_EXPORTS", f"{prefix}-exports"),
        )


@dataclass
class AgentRedisConfig:
    """Redis key prefixes for Agent"""
    session_prefix: str = "agent:session:"
    cache_prefix: str = "agent:cache:"
    rate_prefix: str = "agent:rate:"
    lock_prefix: str = "agent:lock:"

    @classmethod
    def from_env(cls) -> 'AgentRedisConfig':
        prefix = os.getenv("AGENT_REDIS_PREFIX", "agent")
        return cls(
            session_prefix=f"{prefix}:session:",
            cache_prefix=f"{prefix}:cache:",
            rate_prefix=f"{prefix}:rate:",
            lock_prefix=f"{prefix}:lock:",
        )


@dataclass
class AgentPostgresConfig:
    """PostgreSQL schema and table names for Agent"""
    schema: str = "agent"
    checkpoints_table: str = "checkpoints"
    conversations_table: str = "conversations"
    messages_table: str = "messages"

    @classmethod
    def from_env(cls) -> 'AgentPostgresConfig':
        return cls(
            schema=os.getenv("AGENT_POSTGRES_SCHEMA", "agent"),
            checkpoints_table=os.getenv("AGENT_POSTGRES_CHECKPOINTS", "checkpoints"),
            conversations_table=os.getenv("AGENT_POSTGRES_CONVERSATIONS", "conversations"),
            messages_table=os.getenv("AGENT_POSTGRES_MESSAGES", "messages"),
        )


@dataclass
class AgentResourceConfig:
    """Combined Agent resource configuration"""
    qdrant: AgentQdrantConfig = field(default_factory=AgentQdrantConfig)
    minio: AgentMinIOConfig = field(default_factory=AgentMinIOConfig)
    redis: AgentRedisConfig = field(default_factory=AgentRedisConfig)
    postgres: AgentPostgresConfig = field(default_factory=AgentPostgresConfig)

    @classmethod
    def from_env(cls) -> 'AgentResourceConfig':
        return cls(
            qdrant=AgentQdrantConfig.from_env(),
            minio=AgentMinIOConfig.from_env(),
            redis=AgentRedisConfig.from_env(),
            postgres=AgentPostgresConfig.from_env(),
        )


# ===========================================
# Main Agent Configuration
# ===========================================

@dataclass
class AgentConfig:
    """Main Agent service configuration with all sub-configs"""

    # Environment
    environment: str = "dev"
    app_version: str = "1.0.0"

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_debug: bool = False
    api_master_key: Optional[str] = None

    # Execution Mode: "local" (run agent locally) or "proxy" (delegate to pool_manager VMs)
    execution_mode: str = "local"
    pool_manager_url: str = "http://localhost:8090"
    pool_acquire_timeout: int = 30  # seconds to wait for VM
    pool_request_timeout: int = 300  # seconds for query execution

    # CORS
    cors_origins: str = (
        "http://localhost:5173,http://localhost:3000,http://localhost:8080"
    )
    cors_credentials: bool = True

    # AI Models
    ai_provider: str = "openai"
    ai_model: str = "gpt-4.1-nano"
    ai_temperature: float = 0.0
    reason_model: str = (
        "deepseek-reasoner"  # Use DeepSeek Reasoner for reasoning with thinking process
    )
    reason_model_provider: str = "yyds"  # YYDS provider for DeepSeek R1
    response_model: str = "gpt-5-nano"
    response_model_provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"

    # Consul
    consul_host: str = "localhost"
    consul_port: int = 8500
    consul_enabled: bool = True

    # Database
    database_url: str = ""
    database_schema: str = "dev"

    # Sub-configurations
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    infrastructure: InfraConfig = field(default_factory=InfraConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)
    lightning: LightningConfig = field(default_factory=LightningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    resources: AgentResourceConfig = field(default_factory=AgentResourceConfig)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load complete configuration from environment"""
        return cls(
            # Environment
            environment=os.getenv("ENVIRONMENT", "dev"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            # Server
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=_int(os.getenv("PORT") or os.getenv("API_PORT", "8080"), 8080),
            api_debug=_bool(os.getenv("API_DEBUG", "false")),
            api_master_key=os.getenv("API_MASTER_KEY"),
            # CORS
            cors_origins=os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:3000,http://localhost:8080",
            ),
            cors_credentials=_bool(os.getenv("CORS_CREDENTIALS", "true")),
            # Execution Mode
            execution_mode=os.getenv("EXECUTION_MODE", "local"),
            pool_manager_url=os.getenv("POOL_MANAGER_URL", "http://localhost:8090"),
            pool_acquire_timeout=_int(os.getenv("POOL_ACQUIRE_TIMEOUT", "30"), 30),
            pool_request_timeout=_int(os.getenv("POOL_REQUEST_TIMEOUT", "300"), 300),
            # AI (legacy - use model config instead)
            ai_provider=os.getenv("AI_PROVIDER", "openai"),
            ai_model=os.getenv("AI_MODEL", "gpt-4.1-nano"),
            ai_temperature=_float(os.getenv("AI_TEMPERATURE", "0"), 0.0),
            reason_model=os.getenv("REASON_MODEL", "deepseek-reasoner"),
            reason_model_provider=os.getenv("REASON_MODEL_PROVIDER", "yyds"),
            response_model=os.getenv("RESPONSE_MODEL", "gpt-5-nano"),
            response_model_provider=os.getenv("RESPONSE_MODEL_PROVIDER", "openai"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            # Consul
            consul_host=os.getenv("CONSUL_HOST", "localhost"),
            consul_port=_int(os.getenv("CONSUL_PORT", "8500"), 8500),
            consul_enabled=_bool(os.getenv("CONSUL_ENABLED", "true")),
            # Database
            database_url=os.getenv("DATABASE_URL", ""),
            database_schema=os.getenv("DATABASE_SCHEMA", "dev"),
            # Load sub-configs
            logging=LoggingConfig.from_env(),
            infrastructure=InfraConfig.from_env(),
            services=ServiceConfig.from_env(),
            lightning=LightningConfig.from_env(),
            model=ModelConfig.from_env(),
            resources=AgentResourceConfig.from_env(),
        )

    @property
    def allowed_origins(self) -> list[str]:
        """Parse CORS origins"""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # Backward compatibility aliases
    @property
    def model_service_url(self) -> str:
        return self.services.model_service_url

    @property
    def mcp_service_url(self) -> str:
        return self.services.mcp_service_url

    @property
    def isa_api_url(self) -> str:
        return self.services.model_service_url.replace("localhost", "127.0.0.1")

    @property
    def resolved_isa_api_url(self) -> str:
        return self.isa_api_url

    @property
    def resolved_mcp_server_url(self) -> str:
        return self.services.mcp_service_url.replace("localhost", "127.0.0.1")

    @property
    def log_level(self) -> str:
        return self.logging.log_level

    @property
    def log_format(self) -> str:
        return self.logging.log_format

    @property
    def loki_enabled(self) -> bool:
        return self.infrastructure.loki_enabled

    @property
    def loki_host(self) -> str:
        return self.infrastructure.loki_host

    @property
    def loki_port(self) -> int:
        return self.infrastructure.loki_port

    @property
    def loki_url(self) -> str:
        return self.infrastructure.loki_url

    @property
    def redis_host(self) -> str:
        return self.infrastructure.redis_host

    @property
    def redis_port(self) -> int:
        return self.infrastructure.redis_port

    @property
    def postgres_grpc_host(self) -> str:
        return self.infrastructure.postgres_grpc_host

    @property
    def postgres_grpc_port(self) -> int:
        return self.infrastructure.postgres_grpc_port

    @property
    def auth_service_url(self) -> str:
        return self.services.authorization_service_url + "/api/v1/authorization"

    @property
    def account_service_url(self) -> str:
        return self.services.account_service_url

    @property
    def wallet_service_url(self) -> str:
        return self.services.wallet_service_url

    @property
    def session_service_url(self) -> str:
        return self.services.session_service_url

    @property
    def storage_service_url(self) -> str:
        return self.services.storage_service_url
