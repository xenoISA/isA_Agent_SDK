#!/usr/bin/env python3
"""Business services configuration

External service dependencies from isA_User and peer ISA services.
"""
import os
from dataclasses import dataclass

def _bool(val: str) -> bool:
    return val.lower() == "true"

@dataclass
class ServiceConfig:
    """Business service endpoints"""

    # ===========================================
    # Peer ISA Services
    # ===========================================
    # isA_Model - LLM/Embedding service
    model_service_url: str = "http://localhost:8082"

    # isA_MCP - MCP tools service
    mcp_service_url: str = "http://localhost:8081"

    # isA_Data - RAG/Analytics service
    data_service_url: str = "http://localhost:8084"

    # isA_OS - Web/Cloud OS services
    web_service_url: str = "http://localhost:8083"
    os_service_url: str = "http://localhost:8085"

    # ===========================================
    # isA_User - Identity & Access Services
    # ===========================================
    auth_service_url: str = "http://localhost:8201"
    account_service_url: str = "http://localhost:8202"
    session_service_url: str = "http://localhost:8203"
    authorization_service_url: str = "http://localhost:8204"

    # ===========================================
    # isA_User - Business Services
    # ===========================================
    audit_service_url: str = "http://localhost:8205"
    notification_service_url: str = "http://localhost:8206"
    payment_service_url: str = "http://localhost:8207"
    wallet_service_url: str = "http://localhost:8208"
    storage_service_url: str = "http://localhost:8209"

    # ===========================================
    # isA_User - AI/Memory
    # ===========================================
    memory_service_url: str = "http://localhost:8223"

    # ===========================================
    # Service Discovery
    # ===========================================
    use_consul_discovery: bool = True

    @classmethod
    def from_env(cls) -> 'ServiceConfig':
        """
        Load service URLs with priority: env -> consul -> localhost

        This implements the standard service discovery pattern:
        1. Check environment variables first (explicit configuration)
        2. Try Consul service discovery if enabled
        3. Fall back to localhost defaults
        """
        from .consul_config import consul_config

        def resolve_url(env_key: str, consul_service: str, fallback: str) -> str:
            """Resolve service URL using env -> consul -> localhost priority"""
            # Priority 1: Environment variable
            env_url = os.getenv(env_key)
            if env_url:
                return env_url

            # Priority 2: Consul service discovery
            consul_url = consul_config.discover_service(consul_service, fallback)
            return consul_url

        return cls(
            # Peer ISA Services
            model_service_url=resolve_url("ISA_API_URL", "model_service", "http://localhost:8082"),
            mcp_service_url=resolve_url("MCP_SERVER_URL", "mcp_service", "http://localhost:8081"),
            data_service_url=resolve_url("DATA_SERVICE_URL", "data_service", "http://localhost:8084"),
            web_service_url=resolve_url("WEB_SERVICE_URL", "web_service", "http://localhost:8083"),
            os_service_url=resolve_url("OS_SERVICE_URL", "os_service", "http://localhost:8085"),

            # isA_User Services
            auth_service_url=resolve_url("AUTH_SERVICE_URL", "auth_service", "http://localhost:8201"),
            account_service_url=resolve_url("ACCOUNT_SERVICE_URL", "account_service", "http://localhost:8202"),
            session_service_url=resolve_url("SESSION_SERVICE_URL", "session_service", "http://localhost:8203"),
            authorization_service_url=resolve_url("AUTHORIZATION_SERVICE_URL", "authorization_service", "http://localhost:8204"),
            audit_service_url=resolve_url("AUDIT_SERVICE_URL", "audit_service", "http://localhost:8205"),
            notification_service_url=resolve_url("NOTIFICATION_SERVICE_URL", "notification_service", "http://localhost:8206"),
            payment_service_url=resolve_url("PAYMENT_SERVICE_URL", "payment_service", "http://localhost:8207"),
            wallet_service_url=resolve_url("WALLET_SERVICE_URL", "wallet_service", "http://localhost:8208"),
            storage_service_url=resolve_url("STORAGE_SERVICE_URL", "storage_service", "http://localhost:8209"),
            memory_service_url=resolve_url("MEMORY_SERVICE_URL", "memory_service", "http://localhost:8223"),

            # Discovery
            use_consul_discovery=_bool(os.getenv("USE_CONSUL_DISCOVERY", "true")),
        )
