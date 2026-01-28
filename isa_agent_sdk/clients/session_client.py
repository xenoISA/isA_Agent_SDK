#!/usr/bin/env python3
"""
Session Client - Dual-mode session management (Local/Cloud)

Provides:
1. Session CRUD operations
2. Message history management
3. Session resume/fork (like Claude Agent SDK)
4. Context storage per session

Modes:
- LOCAL: File/SQLite based (default, zero dependencies)
- CLOUD: isa_user service (enterprise)
- AUTO: Try cloud, fallback to local
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Backend, BackendConfig, BaseClient, ClientMode

logger = logging.getLogger(__name__)


# ==================== Data Models ====================

@dataclass
class Message:
    """Chat message"""
    id: str
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            metadata=data.get("metadata", {})
        )


@dataclass
class Session:
    """Session data"""
    id: str
    user_id: str
    title: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            id=data["id"],
            user_id=data.get("user_id", "default"),
            title=data.get("title", "Untitled Session"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            context=data.get("context", {}),
            metadata=data.get("metadata", {})
        )


# ==================== Backend Protocol ====================

class SessionBackend(Backend, ABC):
    """Protocol for session backends"""

    @abstractmethod
    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new session"""
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        ...

    @abstractmethod
    async def update_session(self, session: Session) -> bool:
        """Update session"""
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        ...

    @abstractmethod
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Session]:
        """List sessions for a user"""
        ...

    @abstractmethod
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add message to session"""
        ...

    @abstractmethod
    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a session"""
        ...


# ==================== Local Backend (SQLite) ====================

class LocalSessionBackend(SessionBackend):
    """
    Local session backend using SQLite

    Features:
    - Zero external dependencies
    - File-based persistence
    - Full session history
    - Works offline
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.storage_path = Path(config.local_storage_path)
        self.db_path = self.storage_path / config.local_db_name
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> bool:
        """Initialize SQLite database"""
        try:
            # Create storage directory
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Connect to SQLite
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

            # Create tables
            self._create_tables()

            logger.info(f"LocalSessionBackend initialized: {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LocalSessionBackend: {e}")
            return False

    def _create_tables(self):
        """Create database tables"""
        cursor = self._conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                context TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)
        """)

        self._conn.commit()

    async def health_check(self) -> bool:
        """Check if database is accessible"""
        if self._conn is None:
            return False
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new session"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        session = Session(
            id=session_id,
            user_id=user_id,
            title=title or f"Session {session_id[:8]}",
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )

        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (id, user_id, title, created_at, updated_at, context, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.user_id,
            session.title,
            session.created_at.isoformat(),
            session.updated_at.isoformat(),
            json.dumps(session.context),
            json.dumps(session.metadata)
        ))
        self._conn.commit()

        logger.info(f"Created session: {session_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        # Get messages
        messages = await self.get_messages(session_id)

        return Session(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            context=json.loads(row["context"] or "{}"),
            metadata=json.loads(row["metadata"] or "{}"),
            messages=messages
        )

    async def update_session(self, session: Session) -> bool:
        """Update session"""
        session.updated_at = datetime.utcnow()

        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE sessions SET
                title = ?,
                updated_at = ?,
                context = ?,
                metadata = ?
            WHERE id = ?
        """, (
            session.title,
            session.updated_at.isoformat(),
            json.dumps(session.context),
            json.dumps(session.metadata),
            session.id
        ))
        self._conn.commit()

        return cursor.rowcount > 0

    async def delete_session(self, session_id: str) -> bool:
        """Delete session and its messages"""
        cursor = self._conn.cursor()

        # Delete messages first (foreign key)
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

        # Delete session
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

        return cursor.rowcount > 0

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Session]:
        """List sessions for a user"""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))

        sessions = []
        for row in cursor.fetchall():
            sessions.append(Session(
                id=row["id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                context=json.loads(row["context"] or "{}"),
                metadata=json.loads(row["metadata"] or "{}"),
                messages=[]  # Don't load messages for list
            ))

        return sessions

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add message to session"""
        message = Message(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            role=role,
            content=content,
            metadata=metadata or {}
        )

        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            message.id,
            session_id,
            message.role,
            message.content,
            message.timestamp.isoformat(),
            json.dumps(message.metadata)
        ))

        # Update session updated_at
        cursor.execute("""
            UPDATE sessions SET updated_at = ? WHERE id = ?
        """, (datetime.utcnow().isoformat(), session_id))

        self._conn.commit()
        return message

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a session"""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ? OFFSET ?
        """, (session_id, limit, offset))

        messages = []
        for row in cursor.fetchall():
            messages.append(Message(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"] or "{}")
            ))

        return messages


# ==================== Cloud Backend (isa_user) ====================

class CloudSessionBackend(SessionBackend):
    """
    Cloud session backend using isa_user service

    Features:
    - Distributed session storage
    - Multi-user support
    - SSO integration
    - Enterprise features
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.service_url = config.service_url
        self.auth_token = config.auth_token
        self._user_client = None

    async def initialize(self) -> bool:
        """Initialize connection to isa_user service"""
        try:
            from .user_client import user_client
            self._user_client = user_client

            # Set service URL if provided
            if self.service_url:
                self._user_client.base_url = self.service_url

            logger.info(f"CloudSessionBackend initialized: {self.service_url}")
            return True

        except ImportError:
            logger.error("user_client not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize CloudSessionBackend: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if isa_user service is accessible"""
        if self._user_client is None:
            return False
        try:
            result = self._user_client.health_check()
            return result.get("success", False)
        except Exception:
            return False

    async def close(self) -> None:
        """Close connection"""
        self._user_client = None

    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create session via isa_user API"""
        result = self._user_client.create_session(
            user_id=user_id,
            title=title or f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            metadata=metadata or {},
            auth_token=self.auth_token
        )

        if not result.get("success"):
            raise Exception(f"Failed to create session: {result.get('error')}")

        data = result.get("data", {})
        session_id = data.get("session_id") or data.get("id")

        return Session(
            id=session_id,
            user_id=user_id,
            title=title or f"Session {session_id[:8]}",
            metadata=metadata or {}
        )

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session from isa_user API"""
        # Note: isa_user doesn't have direct get_session, we use list and filter
        # This is a limitation of the current API
        result = self._user_client.get_session_messages(
            session_id=session_id,
            page=1,
            page_size=100,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            return None

        messages_data = result.get("data", [])
        messages = [
            Message(
                id=m.get("id", str(uuid.uuid4())),
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=datetime.fromisoformat(m["timestamp"]) if "timestamp" in m else datetime.utcnow(),
                metadata=m.get("metadata", {})
            )
            for m in messages_data
        ]

        return Session(
            id=session_id,
            user_id="unknown",  # API doesn't return this
            title=f"Session {session_id[:8]}",
            messages=messages
        )

    async def update_session(self, session: Session) -> bool:
        """Update session - limited support in isa_user"""
        # Current isa_user API doesn't support session updates
        logger.warning("CloudSessionBackend.update_session: Not fully supported")
        return True

    async def delete_session(self, session_id: str) -> bool:
        """Delete session via isa_user API"""
        # Current isa_user API doesn't have delete endpoint
        logger.warning("CloudSessionBackend.delete_session: Not fully supported")
        return True

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Session]:
        """List sessions from isa_user API"""
        result = self._user_client.get_user_sessions(
            user_id=user_id,
            page=(offset // limit) + 1,
            page_size=limit,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            return []

        sessions = []
        for data in result.get("data", []):
            sessions.append(Session(
                id=data.get("session_id") or data.get("id"),
                user_id=user_id,
                title=data.get("title", f"Session {data.get('session_id', '')[:8]}"),
                created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
                updated_at=datetime.fromisoformat(data["last_activity"]) if "last_activity" in data else datetime.utcnow(),
                metadata=data.get("metadata", {})
            ))

        return sessions

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add message via isa_user API"""
        result = self._user_client.add_message(
            session_id=session_id,
            role=role,
            content=content,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            raise Exception(f"Failed to add message: {result.get('error')}")

        return Message(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            role=role,
            content=content,
            metadata=metadata or {}
        )

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages from isa_user API"""
        result = self._user_client.get_session_messages(
            session_id=session_id,
            page=(offset // limit) + 1,
            page_size=limit,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            return []

        messages = []
        for m in result.get("data", []):
            messages.append(Message(
                id=m.get("id", str(uuid.uuid4())),
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=datetime.fromisoformat(m["timestamp"]) if "timestamp" in m else datetime.utcnow(),
                metadata=m.get("metadata", {})
            ))

        return messages


# ==================== Session Client ====================

class SessionClient(BaseClient[SessionBackend]):
    """
    Dual-mode session client

    Usage:
        # Local mode (default)
        client = SessionClient()
        await client.initialize()

        # Cloud mode
        client = SessionClient(config=BackendConfig(
            mode=ClientMode.CLOUD,
            service_url="http://isa-user:8080"
        ))
        await client.initialize()

        # Auto mode (try cloud, fallback to local)
        client = SessionClient(config=BackendConfig(
            mode=ClientMode.AUTO,
            service_url="http://isa-user:8080"
        ))
        await client.initialize()
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self._current_session_id: Optional[str] = None

    def _create_local_backend(self) -> SessionBackend:
        return LocalSessionBackend(self.config)

    def _create_cloud_backend(self) -> SessionBackend:
        return CloudSessionBackend(self.config)

    # ==================== Session Operations ====================

    async def create_session(
        self,
        user_id: str = "default",
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new session"""
        session = await self.backend.create_session(user_id, title, metadata)
        self._current_session_id = session.id
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return await self.backend.get_session(session_id)

    async def update_session(self, session: Session) -> bool:
        """Update session"""
        return await self.backend.update_session(session)

    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if self._current_session_id == session_id:
            self._current_session_id = None
        return await self.backend.delete_session(session_id)

    async def list_sessions(
        self,
        user_id: str = "default",
        limit: int = 50,
        offset: int = 0
    ) -> List[Session]:
        """List sessions for a user"""
        return await self.backend.list_sessions(user_id, limit, offset)

    # ==================== Message Operations ====================

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add message to session"""
        return await self.backend.add_message(session_id, role, content, metadata)

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a session"""
        return await self.backend.get_messages(session_id, limit, offset)

    # ==================== Session Convenience Methods ====================

    @property
    def current_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self._current_session_id

    def set_current_session(self, session_id: str):
        """Set current session ID"""
        self._current_session_id = session_id

    async def resume(self, session_id: str) -> Optional[Session]:
        """
        Resume a session (like Claude Agent SDK)

        Loads the session and sets it as current.
        """
        session = await self.get_session(session_id)
        if session:
            self._current_session_id = session_id
        return session

    async def fork(self, session_id: str) -> Session:
        """
        Fork a session (like Claude Agent SDK)

        Creates a new session with the same messages as the original.
        """
        original = await self.get_session(session_id)
        if original is None:
            raise ValueError(f"Session {session_id} not found")

        # Create new session
        new_session = await self.create_session(
            user_id=original.user_id,
            title=f"{original.title} (fork)",
            metadata={
                **original.metadata,
                "forked_from": session_id
            }
        )

        # Copy messages
        for msg in original.messages:
            await self.add_message(
                new_session.id,
                msg.role,
                msg.content,
                msg.metadata
            )

        return new_session

    async def set_context(self, session_id: str, key: str, value: Any):
        """Set context value for a session"""
        session = await self.get_session(session_id)
        if session:
            session.context[key] = value
            await self.update_session(session)

    async def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get context value from a session"""
        session = await self.get_session(session_id)
        if session:
            return session.context.get(key, default)
        return default


# ==================== Singleton / Factory ====================

_session_client: Optional[SessionClient] = None


async def get_session_client(config: Optional[BackendConfig] = None) -> SessionClient:
    """Get or create session client singleton"""
    global _session_client

    if _session_client is None:
        _session_client = SessionClient(config)
        await _session_client.initialize()

    return _session_client


async def close_session_client():
    """Close session client"""
    global _session_client

    if _session_client:
        await _session_client.close()
        _session_client = None
