"""
Clients package for dual-mode service integrations (Local/Cloud)

This package contains client implementations that support both local and cloud modes:

Core Clients (dual-mode):
- session_client: Session management (Local: SQLite, Cloud: isa_user)
- storage_client: File storage & RAG (Local: file+SQLite, Cloud: storage_service)
- model_client: Model inference (Local: Ollama, Cloud: isa_model/API)

Service Clients (cloud-only):
- mcp_client: MCP (Model Context Protocol) JSON-RPC client
- pool_manager_client: Pool Manager client for isolated VM execution
- user_client: User service client (enterprise auth)

Base:
- base: Base client with mode switching (local/cloud/auto)
"""

# Base client
from .base import (
    Backend,
    BackendConfig,
    BaseClient,
    ClientMode,
)

# Session client (dual-mode)
from .session_client import (
    SessionClient,
    Session,
    Message,
    SessionBackend,
    LocalSessionBackend,
    CloudSessionBackend,
    get_session_client,
    close_session_client,
)

# Storage client (dual-mode)
from .storage_client import (
    StorageClient,
    FileInfo,
    SearchResult,
    RAGResponse,
    StorageBackend,
    LocalStorageBackend,
    CloudStorageBackend,
    get_storage_client,
    close_storage_client,
)

# Service clients (cloud)
from .mcp_client import MCPClient
from .pool_manager_client import PoolManagerClient, get_pool_manager_client

__all__ = [
    # Base
    'Backend',
    'BackendConfig',
    'BaseClient',
    'ClientMode',

    # Session
    'SessionClient',
    'Session',
    'Message',
    'SessionBackend',
    'LocalSessionBackend',
    'CloudSessionBackend',
    'get_session_client',
    'close_session_client',

    # Storage
    'StorageClient',
    'FileInfo',
    'SearchResult',
    'RAGResponse',
    'StorageBackend',
    'LocalStorageBackend',
    'CloudStorageBackend',
    'get_storage_client',
    'close_storage_client',

    # Service clients
    'MCPClient',
    'PoolManagerClient',
    'get_pool_manager_client',
]
