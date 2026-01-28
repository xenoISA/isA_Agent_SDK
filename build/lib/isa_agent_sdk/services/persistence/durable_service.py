"""
Durable Execution Service
Professional checkpointer management based on LangGraph best practices
"""

import os
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from isa_agent_sdk.core.config import settings
from isa_agent_sdk.utils.logger import api_logger

# Import checkpointer dependencies
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.memory import MemorySaver
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    POSTGRES_AVAILABLE = True
    api_logger.info("✅ PostgreSQL checkpointer dependencies available")
except ImportError as e:
    from langgraph.checkpoint.memory import MemorySaver
    PostgresSaver = None
    AsyncPostgresSaver = None
    psycopg = None
    dict_row = None
    AsyncConnectionPool = None
    POSTGRES_AVAILABLE = False
    api_logger.warning(f"⚠️ PostgreSQL not available: {e}")

# Import custom checkpointers
try:
    from .session_checkpointer import SessionServiceCheckpointer
    SESSION_CHECKPOINTER_AVAILABLE = True
    api_logger.info("✅ SessionServiceCheckpointer available")
except ImportError as e:
    SessionServiceCheckpointer = None
    SESSION_CHECKPOINTER_AVAILABLE = False
    api_logger.warning(f"⚠️ SessionServiceCheckpointer not available: {e}")


class DurableService:
    """
    Durable execution service providing persistent checkpointing.

    Architecture:
    - Development: MemorySaver (fast, no setup)
    - Production: SessionService or PostgreSQL (persistent, scalable)
    - Automatic fallback for reliability

    Supported Backends:
    - session_service: 使用 session_service 微服务存储 checkpoints（推荐）
    - postgres: PostgreSQL 直连（传统方式）
    - memory: 内存存储（开发/测试）

    Best Practice (from LangGraph docs):
    - Use global AsyncConnectionPool created in FastAPI lifespan
    - Initialize AsyncPostgresSaver once and share across requests
    - Call setup() during initialization
    """

    def __init__(
        self,
        force_postgres: bool = False,
        default_postgres: bool = False,  # 改为 False，优先使用 session_service
        checkpointer_backend: str = None
    ):
        self.force_postgres = force_postgres
        self.default_postgres = default_postgres

        # 确定 checkpointer 后端
        # 优先级：环境变量 > 参数 > 默认值
        self.checkpointer_backend = (
            os.getenv("CHECKPOINTER_BACKEND") or
            checkpointer_backend or
            ("postgres" if (force_postgres or default_postgres) else "session_service")
        )

        self.environment = os.getenv("ENVIRONMENT", "dev")
        self._database_url = None
        self._checkpointer_cache = None
        self._connection_pool = None  # Async connection pool (global, created in lifespan)

        api_logger.info(f"🔄 DurableService initialized for environment: {self.environment}")
        api_logger.info(f"📊 Checkpointer backend: {self.checkpointer_backend}")
    
    def get_checkpointer(self):
        """
        Get appropriate checkpointer based on backend configuration.

        Returns:
            Checkpointer instance (SessionServiceCheckpointer, PostgresSaver, or MemorySaver)
        """
        if self._checkpointer_cache:
            return self._checkpointer_cache

        # 根据配置选择 checkpointer
        backend = self.checkpointer_backend.lower()

        if backend == "session_service":
            checkpointer = self._create_session_service_checkpointer()
        elif backend == "postgres":
            checkpointer = self._create_postgres_checkpointer()
        elif backend == "memory":
            checkpointer = self._create_memory_checkpointer()
        else:
            # 默认使用 session_service，如果不可用则降级到 memory
            api_logger.warning(f"⚠️ Unknown backend '{backend}', trying session_service")
            checkpointer = self._create_session_service_checkpointer()

        self._checkpointer_cache = checkpointer
        return checkpointer
    
    def _should_use_postgres(self) -> bool:
        """Determine if PostgreSQL should be used"""
        # Enable PostgreSQL checkpointer now that we have proper setup
        # Using JsonPlusSerializer with pickle_fallback to handle LangGraph issue #5511
        
        # 如果明确禁用PostgreSQL
        if os.getenv("USE_POSTGRES_CHECKPOINTER", "").lower() == "false":
            return False
            
        return (
            self.force_postgres or
            self.default_postgres or  # 默认使用PostgreSQL
            self.environment in ["production", "staging"] or
            os.getenv("USE_POSTGRES_CHECKPOINTER", "true").lower() == "true"  # 默认值改为true
        )
    
    def _create_memory_checkpointer(self):
        """Create MemorySaver checkpointer"""
        api_logger.info(f"🧠 Using MemorySaver for {self.environment}")
        api_logger.info("   ✅ Fast, reliable, no setup required")
        api_logger.info("   📝 Note: State will not persist between restarts")
        return MemorySaver()

    def _create_session_service_checkpointer(self):
        """Create SessionService checkpointer with fallback"""
        if not SESSION_CHECKPOINTER_AVAILABLE:
            api_logger.warning("⚠️ SessionServiceCheckpointer not available, using MemorySaver")
            return self._create_memory_checkpointer()

        # 获取 session_service URL
        session_service_url = getattr(settings, 'session_service_url', None)
        if not session_service_url:
            api_logger.warning("⚠️ session_service_url not configured, using MemorySaver")
            return self._create_memory_checkpointer()

        try:
            api_logger.info(f"🔗 Creating SessionService checkpointer for {self.environment}")
            api_logger.info(f"   URL: {session_service_url}")

            checkpointer = SessionServiceCheckpointer(
                session_service_url=session_service_url,
                user_id="system",  # 使用 system 用户来存储 checkpoints
                timeout=30.0
            )

            api_logger.info("✅ SessionService checkpointer created successfully")
            api_logger.info("   📊 Graph state will persist in session_service")
            api_logger.info("   🔄 Using SessionServiceCheckpointer with messages API")
            api_logger.info("   ✨ Checkpoints stored as special messages (message_type='checkpoint')")
            return checkpointer

        except Exception as e:
            api_logger.error(f"❌ SessionService checkpointer creation failed: {e}")
            api_logger.info("⚠️ Falling back to MemorySaver")
            return self._create_memory_checkpointer()
    
    def _create_postgres_checkpointer(self):
        """Create PostgreSQL checkpointer with fallback"""
        if not POSTGRES_AVAILABLE:
            api_logger.warning("⚠️ PostgreSQL dependencies not available, using MemorySaver")
            return self._create_memory_checkpointer()
        
        database_url = self._get_database_url()
        if not database_url:
            api_logger.warning("⚠️ No database URL configured, using MemorySaver")
            return self._create_memory_checkpointer()
        
        try:
            api_logger.info(f"🔗 Creating PostgreSQL checkpointer for {self.environment}")
            
            # For production, create AsyncPostgresSaver with direct connection
            # This creates a persistent connection without context manager wrapper
            import psycopg
            from psycopg_pool import AsyncConnectionPool
            
            # Get schema from settings
            schema = getattr(settings, 'database_schema', 'agent')

            # Create a modified connection string with schema in options
            # Format: postgresql://user:pass@host:port/db?options=-csearch_path=schema
            if '?' in database_url:
                # URL already has parameters
                conn_str = f"{database_url}&options=-csearch_path={schema}"
            else:
                # URL doesn't have parameters
                conn_str = f"{database_url}?options=-csearch_path={schema}"

            api_logger.info(f"📊 Creating connection pool for schema: {schema}")

            # Create connection pool for async operations
            pool = AsyncConnectionPool(
                conn_str,
                min_size=1,
                max_size=5,
                max_waiting=10,  # Max number of requests waiting for a connection
                timeout=30.0,  # Connection timeout in seconds
            )

            # Store the pool for cleanup later
            self._connection_pool = pool

            # Create AsyncPostgresSaver with the connection pool
            checkpointer = AsyncPostgresSaver(
                pool,
                serde=None  # Use default serialization
            )

            api_logger.info("✅ PostgreSQL checkpointer created successfully")
            api_logger.info("   📊 Graph state will persist between restarts")
            api_logger.info("   🔄 Using AsyncPostgresSaver with connection pool")
            api_logger.info("   ⚠️ Using stable LangGraph 0.5.4 + langgraph-checkpoint-postgres 2.0.8")
            return checkpointer
            
        except Exception as e:
            api_logger.error(f"❌ PostgreSQL checkpointer creation failed: {e}")
            api_logger.info("⚠️ Falling back to MemorySaver")
            return self._create_memory_checkpointer()
    
    def setup_postgres_tables(self, database_url: Optional[str] = None) -> bool:
        """
        Setup PostgreSQL tables for checkpointer.
        
        Args:
            database_url: Optional database URL override
            
        Returns:
            True if setup successful or not needed, False if failed
        """
        if not self._should_use_postgres():
            api_logger.info(f"ℹ️ Using MemorySaver for {self.environment}, no setup needed")
            return True
        
        if not POSTGRES_AVAILABLE:
            api_logger.warning("⚠️ PostgreSQL not available, no setup needed")
            return True
        
        database_url = database_url or self._get_database_url()
        if not database_url:
            api_logger.warning("⚠️ No database URL configured, no setup needed")
            return True
        
        try:
            api_logger.info(f"🔧 Setting up PostgreSQL tables for {self.environment}...")

            # Get schema from settings
            schema = getattr(settings, 'database_schema', 'agent')

            # Create connection with required parameters for setup
            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            }

            with psycopg.connect(database_url, **connection_kwargs) as conn:
                # Set search_path to correct schema
                conn.execute(f"SET search_path TO {schema}")
                api_logger.info(f"📊 Using schema: {schema}")

                checkpointer = PostgresSaver(conn)
                checkpointer.setup()
                api_logger.info("✅ PostgreSQL checkpointer tables initialized")

            return True
            
        except Exception as e:
            api_logger.error(f"❌ Database setup failed: {e}")
            api_logger.warning("⚠️ Will use MemorySaver as fallback")
            return False
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test database connection and return status.
        
        Returns:
            Dict with connection status and details
        """
        if not self._should_use_postgres():
            return {
                "status": "success",
                "type": "memory",
                "message": "Using MemorySaver, no connection test needed"
            }
        
        if not POSTGRES_AVAILABLE:
            return {
                "status": "warning", 
                "type": "memory_fallback",
                "message": "PostgreSQL not available, using MemorySaver"
            }
        
        database_url = self._get_database_url()
        if not database_url:
            return {
                "status": "warning",
                "type": "memory_fallback", 
                "message": "No database URL configured, using MemorySaver"
            }
        
        try:
            # Test connection with proper parameters
            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            }
            
            conn = psycopg.connect(database_url, **connection_kwargs)
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()  # Test query execution
            
            conn.close()
            
            return {
                "status": "success",
                "type": "postgres",
                "message": "PostgreSQL connection successful",
                "database": self._mask_database_url(database_url)
            }
            
        except Exception as e:
            api_logger.error(f"❌ Database connection test failed: {e}")
            return {
                "status": "error",
                "type": "postgres_failed",
                "message": f"Connection failed: {str(e)[:100]}...",
                "will_fallback": True
            }
    
    def _get_database_url(self) -> str:
        """Get database URL from configuration"""
        if self._database_url:
            return self._database_url

        # Priority 1: Environment variable DATABASE_URL (highest priority)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            self._database_url = database_url
            api_logger.info(f"📊 Using DATABASE_URL from environment")
            return self._database_url

        # Priority 2: settings.database_url
        if settings.database_url:
            self._database_url = settings.database_url
            api_logger.info(f"📊 Using database_url from settings")
            return self._database_url

        # No database URL configured
        api_logger.warning("⚠️ No DATABASE_URL configured")
        return ""
    
    def _mask_database_url(self, url: str) -> str:
        """Mask sensitive information in database URL"""
        if "@" in url:
            return url.split("@")[1]  # Remove credentials
        return url
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get comprehensive service information.

        Returns:
            Dict with service configuration and status
        """
        backend = self.checkpointer_backend.lower()

        # 检查各后端的可用性
        session_service_available = SESSION_CHECKPOINTER_AVAILABLE and bool(getattr(settings, 'session_service_url', None))
        postgres_available = POSTGRES_AVAILABLE and bool(self._get_database_url())

        # 确定实际使用的 checkpointer 类型
        if backend == "session_service":
            actual_type = "session_service" if session_service_available else "memory"
        elif backend == "postgres":
            actual_type = "postgres" if postgres_available else "memory"
        elif backend == "memory":
            actual_type = "memory"
        else:
            actual_type = "session_service" if session_service_available else "memory"

        # 是否支持持久化
        persistent = actual_type in ["session_service", "postgres"]

        return {
            "environment": self.environment,
            "checkpointer_backend": self.checkpointer_backend,
            "actual_checkpointer": actual_type,
            "session_service_available": session_service_available,
            "session_service_url": getattr(settings, 'session_service_url', "not configured"),
            "postgres_available": postgres_available,
            "database_url_masked": self._mask_database_url(self._get_database_url()) if self._get_database_url() else "not configured",
            "features": {
                "durable_execution": persistent,
                "cross_restart_persistence": persistent,
                "multi_thread_support": True,
                "interrupt_resume": True,
                "checkpoint_in_session_service": actual_type == "session_service"
            }
        }
    
    def enable_postgres_for_testing(self):
        """Enable PostgreSQL for testing purposes"""
        self.force_postgres = True
        self._checkpointer_cache = None  # Clear cache
        api_logger.info("🧪 PostgreSQL checkpointer enabled for testing")
    
    def reset_checkpointer_cache(self):
        """Reset checkpointer cache (useful for testing)"""
        self._checkpointer_cache = None
        api_logger.info("🔄 Checkpointer cache reset")
    
    # ========== Durable Execution Helpers ==========
    
    def create_thread_config(self, thread_id: str) -> Dict[str, Any]:
        """
        Create thread configuration for durable execution.
        
        Args:
            thread_id: Unique thread identifier
            
        Returns:
            Thread configuration dict
        """
        return {
            "configurable": {
                "thread_id": thread_id
            }
        }
    
    def validate_thread_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate thread configuration.
        
        Args:
            config: Thread configuration
            
        Returns:
            True if valid, False otherwise
        """
        try:
            thread_id = config["configurable"]["thread_id"]
            return bool(thread_id)
        except (KeyError, TypeError):
            return False
    
    @asynccontextmanager
    async def durable_execution_context(self, thread_id: str):
        """
        Context manager for durable execution.
        
        Args:
            thread_id: Thread identifier for this execution
            
        Yields:
            Dict with checkpointer and config
        """
        checkpointer = self.get_checkpointer()
        config = self.create_thread_config(thread_id)
        
        api_logger.info(f"🚀 Starting durable execution - thread: {thread_id}")
        
        try:
            yield {
                "checkpointer": checkpointer,
                "config": config,
                "thread_id": thread_id
            }
        finally:
            api_logger.info(f"✅ Durable execution completed - thread: {thread_id}")

    async def cleanup(self):
        """
        Cleanup resources - close connection pool if exists
        """
        if self._connection_pool is not None:
            try:
                api_logger.info("🔌 Closing PostgreSQL connection pool...")
                await self._connection_pool.close()
                api_logger.info("✅ PostgreSQL connection pool closed")
            except Exception as e:
                api_logger.error(f"❌ Error closing connection pool: {e}")
            finally:
                self._connection_pool = None
                self._checkpointer_cache = None


# Global service instance
durable_service = DurableService()


def get_durable_service(force_postgres: bool = False) -> DurableService:
    """
    Get durable service instance.
    
    Args:
        force_postgres: Force PostgreSQL usage
        
    Returns:
        DurableService instance
    """
    if force_postgres:
        durable_service.enable_postgres_for_testing()
    
    return durable_service


def get_checkpointer():
    """
    Convenience function to get checkpointer directly.

    Returns:
        Checkpointer instance
    """
    return durable_service.get_checkpointer()


# ============================================================================
# Async Initialization (FastAPI Lifespan)
# ============================================================================

async def initialize_async_pool():
    """
    Initialize checkpointer based on configured backend in FastAPI lifespan.

    Supports multiple backends:
    - postgres: AsyncConnectionPool + AsyncPostgresSaver
    - session_service: SessionServiceCheckpointer (via User Service API)
    - memory: MemorySaver (fallback)

    This follows LangGraph best practices:
    - Create one global checkpointer instance
    - Initialize once during startup
    - Share checkpointer across all requests

    Returns:
        Tuple[AsyncConnectionPool | None, BaseCheckpointSaver | None]:
            (connection_pool, checkpointer) or (None, None) if using MemorySaver
    """
    # Determine which backend to use
    backend = durable_service.checkpointer_backend.lower()

    # Handle session_service backend
    if backend == "session_service":
        if not SESSION_CHECKPOINTER_AVAILABLE:
            api_logger.warning("⚠️ SessionServiceCheckpointer not available, using MemorySaver")
            return None, None

        try:
            from .session_checkpointer import SessionServiceCheckpointer

            # Get session service URL from settings
            session_service_url = os.getenv("SESSION_SERVICE_URL", settings.session_service_url)
            if not session_service_url:
                api_logger.warning("⚠️ SESSION_SERVICE_URL not configured, using MemorySaver")
                return None, None

            api_logger.info(f"🔗 Initializing SessionServiceCheckpointer with URL: {session_service_url}")
            checkpointer = SessionServiceCheckpointer(
                session_service_url=session_service_url,
                auth_token=None,  # Will use API key from headers if needed
                user_id="agent_service"
            )

            api_logger.info("✅ SessionServiceCheckpointer initialized and ready")
            api_logger.info("   📊 Graph state will persist in User Service messages API")

            # Store checkpointer reference
            durable_service._checkpointer_cache = checkpointer

            return None, checkpointer  # No connection pool needed for session service

        except Exception as e:
            api_logger.error(f"❌ Failed to initialize SessionServiceCheckpointer: {e}")
            api_logger.exception(e)
            return None, None

    # Handle postgres backend
    if backend != "postgres":
        api_logger.info(f"🧠 Using MemorySaver (backend={backend})")
        return None, None

    # Check if we should use PostgreSQL
    if not durable_service._should_use_postgres():
        api_logger.info("🧠 Using MemorySaver (PostgreSQL checkpointer disabled)")
        return None, None

    if not POSTGRES_AVAILABLE:
        api_logger.warning("⚠️ PostgreSQL dependencies not available, using MemorySaver")
        return None, None

    database_url = durable_service._get_database_url()
    if not database_url:
        api_logger.warning("⚠️ No database URL configured, using MemorySaver")
        return None, None

    try:
        # Get schema from settings
        schema = getattr(settings, 'database_schema', 'agent')

        api_logger.info(f"🔗 Initializing AsyncConnectionPool for schema: {schema}")

        # Configure callback must leave connection in clean state (no open transaction)
        async def configure_conn(conn):
            """Configure each connection to use the correct schema"""
            # Execute SET and commit to leave connection in clean state
            await conn.execute(f"SET search_path TO {schema}")
            await conn.commit()  # Critical: must commit to avoid INTRANS state

        # Create AsyncConnectionPool (recommended pattern from LangGraph docs)
        pool = AsyncConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=10,
            timeout=30.0,
            max_waiting=20,
            configure=configure_conn,  # Set search_path on each connection
            open=False,
        )

        # Open the pool
        await pool.open(wait=True, timeout=30.0)
        api_logger.info(f"✅ AsyncConnectionPool opened successfully with schema: {schema}")

        # Create AsyncPostgresSaver with the pool
        checkpointer = AsyncPostgresSaver(pool)

        # Setup tables (if not exists)
        # Note: setup() requires autocommit mode for CREATE INDEX CONCURRENTLY
        try:
            api_logger.info(f"🔧 Setting up PostgreSQL checkpointer tables in schema: {schema}")
            # Get a connection in autocommit mode for setup
            async with await pool.getconn() as conn:
                await conn.set_autocommit(True)
                temp_saver = AsyncPostgresSaver(conn)
                await temp_saver.setup()
            api_logger.info("✅ PostgreSQL checkpointer tables initialized")
        except Exception as e:
            # Tables might already exist, which is fine
            api_logger.warning(f"⚠️ Table setup note: {e}")
            api_logger.info("   Continuing with existing tables...")

        # Store pool reference for cleanup
        durable_service._connection_pool = pool
        durable_service._checkpointer_cache = checkpointer

        api_logger.info("✅ AsyncPostgresSaver initialized and ready")
        api_logger.info("   📊 Graph state will persist between restarts")
        api_logger.info("   🔄 Checkpointer shared across all requests")

        return pool, checkpointer

    except Exception as e:
        api_logger.error(f"❌ Failed to initialize AsyncConnectionPool: {e}")
        api_logger.warning("⚠️ Falling back to MemorySaver")
        import traceback
        api_logger.error(traceback.format_exc())
        return None, None