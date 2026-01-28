#!/usr/bin/env python3
"""
Storage Client - Dual-mode storage management (Local/Cloud)

Provides:
1. File storage (upload, download, delete)
2. Semantic search (vector similarity)
3. RAG query support
4. File metadata management

Modes:
- LOCAL: File system + SQLite + local embeddings (default, minimal dependencies)
- CLOUD: storage_service (enterprise, full features)
- AUTO: Try cloud, fallback to local
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from .base import Backend, BackendConfig, BaseClient, ClientMode

logger = logging.getLogger(__name__)


# ==================== Data Models ====================

@dataclass
class FileInfo:
    """File metadata"""
    id: str
    filename: str
    content_type: str
    size: int
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    path: Optional[str] = None  # Local path or cloud URL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "path": self.path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileInfo":
        return cls(
            id=data["id"],
            filename=data["filename"],
            content_type=data.get("content_type", "application/octet-stream"),
            size=data.get("size", 0),
            user_id=data.get("user_id", "default"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            path=data.get("path")
        )


@dataclass
class SearchResult:
    """Search result with relevance score"""
    file_id: str
    filename: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class RAGResponse:
    """RAG query response"""
    answer: str
    sources: List[SearchResult]
    confidence: float
    query: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "query": self.query
        }


# ==================== Backend Protocol ====================

class StorageBackend(Backend, ABC):
    """Protocol for storage backends"""

    @abstractmethod
    async def upload_file(
        self,
        content: bytes,
        filename: str,
        user_id: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FileInfo:
        """Upload a file"""
        ...

    @abstractmethod
    async def download_file(self, file_id: str, user_id: str) -> Optional[bytes]:
        """Download file content"""
        ...

    @abstractmethod
    async def get_file_info(self, file_id: str, user_id: str) -> Optional[FileInfo]:
        """Get file metadata"""
        ...

    @abstractmethod
    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """Delete a file"""
        ...

    @abstractmethod
    async def list_files(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        tags: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """List files for a user"""
        ...

    @abstractmethod
    async def semantic_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Semantic search across files"""
        ...

    @abstractmethod
    async def rag_query(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        max_tokens: int = 500
    ) -> RAGResponse:
        """RAG query against files"""
        ...


# ==================== Local Backend (File + SQLite) ====================

class LocalStorageBackend(StorageBackend):
    """
    Local storage backend using file system and SQLite

    Features:
    - File system storage
    - SQLite metadata
    - Basic text search (FTS5)
    - Optional: sentence-transformers for embeddings

    Note: For full semantic search, install sentence-transformers:
        pip install sentence-transformers
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.storage_path = Path(config.local_storage_path)
        self.files_path = self.storage_path / "files"
        self.db_path = self.storage_path / config.local_db_name
        self._conn: Optional[sqlite3.Connection] = None
        self._embedder = None

    async def initialize(self) -> bool:
        """Initialize storage"""
        try:
            # Create directories
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.files_path.mkdir(parents=True, exist_ok=True)

            # Connect to SQLite
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

            # Create tables
            self._create_tables()

            # Try to load embedder (optional)
            self._load_embedder()

            logger.info(f"LocalStorageBackend initialized: {self.storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LocalStorageBackend: {e}")
            return False

    def _create_tables(self):
        """Create database tables"""
        cursor = self._conn.cursor()

        # Files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_type TEXT,
                size INTEGER,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                path TEXT,
                content_text TEXT
            )
        """)

        # Embeddings table (for semantic search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            )
        """)

        # Full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                filename, content_text, tags,
                content='files',
                content_rowid='rowid'
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_files_user_id ON files(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_file_id ON embeddings(file_id)
        """)

        self._conn.commit()

    def _load_embedder(self):
        """Load sentence-transformers model if available"""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers for semantic search")
        except ImportError:
            logger.info("sentence-transformers not available, using FTS5 for search")
            self._embedder = None

    async def health_check(self) -> bool:
        """Check if storage is accessible"""
        if self._conn is None:
            return False
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1")
            return self.files_path.exists()
        except Exception:
            return False

    async def close(self) -> None:
        """Close connections"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _get_file_path(self, file_id: str) -> Path:
        """Get file path for a file ID"""
        # Use subdirectories to avoid too many files in one directory
        return self.files_path / file_id[:2] / file_id

    def _extract_text(self, content: bytes, content_type: str) -> str:
        """Extract text from file content"""
        # Simple text extraction - can be extended
        if content_type.startswith("text/"):
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        # TODO: Add PDF, DOCX extraction with optional dependencies
        return ""

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        user_id: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FileInfo:
        """Upload a file"""
        file_id = f"file_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        # Detect content type if not provided
        if content_type is None:
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"

        # Save file
        file_path = self._get_file_path(file_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

        # Extract text for search
        content_text = self._extract_text(content, content_type)

        # Create file info
        file_info = FileInfo(
            id=file_id,
            filename=filename,
            content_type=content_type,
            size=len(content),
            user_id=user_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            tags=tags or [],
            path=str(file_path)
        )

        # Save to database
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO files (id, filename, content_type, size, user_id, created_at, updated_at, metadata, tags, path, content_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_info.id,
            file_info.filename,
            file_info.content_type,
            file_info.size,
            file_info.user_id,
            file_info.created_at.isoformat(),
            file_info.updated_at.isoformat(),
            json.dumps(file_info.metadata),
            json.dumps(file_info.tags),
            file_info.path,
            content_text
        ))

        # Update FTS index
        cursor.execute("""
            INSERT INTO files_fts (rowid, filename, content_text, tags)
            SELECT rowid, filename, content_text, tags FROM files WHERE id = ?
        """, (file_id,))

        # Generate embeddings if available
        if self._embedder and content_text:
            await self._generate_embeddings(file_id, content_text)

        self._conn.commit()
        logger.info(f"Uploaded file: {filename} -> {file_id}")
        return file_info

    async def _generate_embeddings(self, file_id: str, text: str):
        """Generate and store embeddings for text"""
        if not self._embedder:
            return

        # Chunk text (simple approach)
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        cursor = self._conn.cursor()
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = self._embedder.encode(chunk)
                cursor.execute("""
                    INSERT INTO embeddings (id, file_id, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"emb_{uuid.uuid4().hex[:12]}",
                    file_id,
                    i,
                    chunk,
                    embedding.tobytes()
                ))

    async def download_file(self, file_id: str, user_id: str) -> Optional[bytes]:
        """Download file content"""
        file_info = await self.get_file_info(file_id, user_id)
        if file_info is None:
            return None

        file_path = Path(file_info.path)
        if not file_path.exists():
            return None

        return file_path.read_bytes()

    async def get_file_info(self, file_id: str, user_id: str) -> Optional[FileInfo]:
        """Get file metadata"""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM files WHERE id = ? AND user_id = ?
        """, (file_id, user_id))
        row = cursor.fetchone()

        if row is None:
            return None

        return FileInfo(
            id=row["id"],
            filename=row["filename"],
            content_type=row["content_type"],
            size=row["size"],
            user_id=row["user_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
            tags=json.loads(row["tags"] or "[]"),
            path=row["path"]
        )

    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """Delete a file"""
        file_info = await self.get_file_info(file_id, user_id)
        if file_info is None:
            return False

        # Delete physical file
        file_path = Path(file_info.path)
        if file_path.exists():
            file_path.unlink()

        # Delete from database
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM embeddings WHERE file_id = ?", (file_id,))
        cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self._conn.commit()

        return True

    async def list_files(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        tags: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """List files for a user"""
        cursor = self._conn.cursor()

        if tags:
            # Filter by tags (simple approach)
            cursor.execute("""
                SELECT * FROM files
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM files
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset))

        files = []
        for row in cursor.fetchall():
            file_tags = json.loads(row["tags"] or "[]")
            # Filter by tags if provided
            if tags and not any(t in file_tags for t in tags):
                continue
            files.append(FileInfo(
                id=row["id"],
                filename=row["filename"],
                content_type=row["content_type"],
                size=row["size"],
                user_id=row["user_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"] or "{}"),
                tags=file_tags,
                path=row["path"]
            ))

        return files

    async def semantic_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Semantic search across files"""
        if self._embedder:
            return await self._vector_search(user_id, query, top_k, min_score, tags)
        else:
            return await self._fts_search(user_id, query, top_k, tags)

    async def _vector_search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        min_score: float,
        tags: Optional[List[str]]
    ) -> List[SearchResult]:
        """Vector similarity search"""
        import numpy as np

        query_embedding = self._embedder.encode(query)

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT e.*, f.filename, f.user_id, f.tags
            FROM embeddings e
            JOIN files f ON e.file_id = f.id
            WHERE f.user_id = ?
        """, (user_id,))

        results = []
        for row in cursor.fetchall():
            # Filter by tags if provided
            if tags:
                file_tags = json.loads(row["tags"] or "[]")
                if not any(t in file_tags for t in tags):
                    continue

            # Calculate cosine similarity
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            score = float(np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            ))

            if score >= min_score:
                results.append(SearchResult(
                    file_id=row["file_id"],
                    filename=row["filename"],
                    content=row["content"],
                    score=score
                ))

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def _fts_search(
        self,
        user_id: str,
        query: str,
        top_k: int,
        tags: Optional[List[str]]
    ) -> List[SearchResult]:
        """Full-text search fallback"""
        cursor = self._conn.cursor()

        # Use FTS5 for search
        cursor.execute("""
            SELECT f.*, bm25(files_fts) as score
            FROM files f
            JOIN files_fts ON f.rowid = files_fts.rowid
            WHERE files_fts MATCH ? AND f.user_id = ?
            ORDER BY score
            LIMIT ?
        """, (query, user_id, top_k))

        results = []
        for row in cursor.fetchall():
            file_tags = json.loads(row["tags"] or "[]")
            if tags and not any(t in file_tags for t in tags):
                continue
            results.append(SearchResult(
                file_id=row["id"],
                filename=row["filename"],
                content=row["content_text"][:500] if row["content_text"] else "",
                score=abs(row["score"])  # BM25 returns negative scores
            ))

        return results

    async def rag_query(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        max_tokens: int = 500
    ) -> RAGResponse:
        """RAG query - simple implementation"""
        # Get relevant documents
        sources = await self.semantic_search(user_id, query, top_k)

        if not sources:
            return RAGResponse(
                answer="No relevant documents found.",
                sources=[],
                confidence=0.0,
                query=query
            )

        # Build context from sources
        context = "\n\n".join([
            f"[{s.filename}]: {s.content[:500]}"
            for s in sources
        ])

        # Simple answer generation (no LLM, just return context)
        # In production, this would call an LLM
        answer = f"Based on the following sources:\n\n{context}"

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=sources[0].score if sources else 0.0,
            query=query
        )


# ==================== Cloud Backend (storage_service) ====================

class CloudStorageBackend(StorageBackend):
    """
    Cloud storage backend using storage_service

    Features:
    - Full RAG capabilities
    - Advanced semantic search
    - File indexing and processing
    - Enterprise features
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.service_url = config.service_url
        self.auth_token = config.auth_token
        self._storage_service = None

    async def initialize(self) -> bool:
        """Initialize connection to storage_service"""
        try:
            from ..components.storage_service import StorageService
            self._storage_service = StorageService(self.service_url)
            logger.info(f"CloudStorageBackend initialized: {self.service_url}")
            return True
        except ImportError:
            logger.error("StorageService not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize CloudStorageBackend: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if storage_service is accessible"""
        if self._storage_service is None:
            return False
        result = self._storage_service.health_check()
        return result.get("success", False)

    async def close(self) -> None:
        """Close connection"""
        if self._storage_service:
            self._storage_service.close()
            self._storage_service = None

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        user_id: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FileInfo:
        """Upload via storage_service"""
        result = self._storage_service.upload_file(
            file_content=content,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            raise Exception(f"Upload failed: {result.get('error')}")

        return FileInfo(
            id=result.get("file_id"),
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size=len(content),
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
            path=result.get("download_url")
        )

    async def download_file(self, file_id: str, user_id: str) -> Optional[bytes]:
        """Download from storage_service"""
        # Note: Current storage_service doesn't have direct download
        # This would need to be implemented
        logger.warning("CloudStorageBackend.download_file: Not fully implemented")
        return None

    async def get_file_info(self, file_id: str, user_id: str) -> Optional[FileInfo]:
        """Get file info from storage_service"""
        result = self._storage_service.get_file_info(
            file_id=file_id,
            user_id=user_id,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            return None

        data = result.get("file_info", {})
        return FileInfo.from_dict(data)

    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """Delete via storage_service"""
        result = self._storage_service.delete_file(
            file_id=file_id,
            user_id=user_id,
            auth_token=self.auth_token
        )
        return result.get("success", False)

    async def list_files(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        tags: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """List files from storage_service"""
        result = self._storage_service.list_files(
            user_id=user_id,
            limit=limit,
            offset=offset,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            return []

        return [FileInfo.from_dict(f) for f in result.get("files", [])]

    async def semantic_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Semantic search via storage_service"""
        result = self._storage_service.semantic_search(
            user_id=user_id,
            query=query,
            top_k=top_k,
            min_score=min_score,
            tags=tags,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            return []

        return [
            SearchResult(
                file_id=r.get("file_id", ""),
                filename=r.get("filename", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {})
            )
            for r in result.get("results", [])
        ]

    async def rag_query(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        max_tokens: int = 500
    ) -> RAGResponse:
        """RAG query via storage_service"""
        result = self._storage_service.rag_query(
            user_id=user_id,
            query=query,
            top_k=top_k,
            max_tokens=max_tokens,
            auth_token=self.auth_token
        )

        if not result.get("success"):
            return RAGResponse(
                answer=f"Error: {result.get('error')}",
                sources=[],
                confidence=0.0,
                query=query
            )

        sources = [
            SearchResult(
                file_id=s.get("file_id", ""),
                filename=s.get("filename", ""),
                content=s.get("content", ""),
                score=s.get("score", 0.0)
            )
            for s in result.get("sources", [])
        ]

        return RAGResponse(
            answer=result.get("answer", ""),
            sources=sources,
            confidence=result.get("confidence", 0.0),
            query=query
        )


# ==================== Storage Client ====================

class StorageClient(BaseClient[StorageBackend]):
    """
    Dual-mode storage client

    Usage:
        # Local mode (default)
        client = StorageClient()
        await client.initialize()

        # Upload file
        file_info = await client.upload_file(
            content=b"Hello, world!",
            filename="hello.txt",
            user_id="user123"
        )

        # Search
        results = await client.semantic_search("user123", "hello")

        # RAG query
        response = await client.rag_query("user123", "What is the content?")
    """

    def _create_local_backend(self) -> StorageBackend:
        return LocalStorageBackend(self.config)

    def _create_cloud_backend(self) -> StorageBackend:
        return CloudStorageBackend(self.config)

    # ==================== File Operations ====================

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        user_id: str = "default",
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FileInfo:
        """Upload a file"""
        return await self.backend.upload_file(
            content, filename, user_id, content_type, metadata, tags
        )

    async def upload_from_path(
        self,
        path: Union[str, Path],
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FileInfo:
        """Upload a file from path"""
        path = Path(path)
        content = path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(path))
        return await self.upload_file(
            content=content,
            filename=path.name,
            user_id=user_id,
            content_type=content_type,
            metadata=metadata,
            tags=tags
        )

    async def download_file(self, file_id: str, user_id: str = "default") -> Optional[bytes]:
        """Download file content"""
        return await self.backend.download_file(file_id, user_id)

    async def get_file_info(self, file_id: str, user_id: str = "default") -> Optional[FileInfo]:
        """Get file metadata"""
        return await self.backend.get_file_info(file_id, user_id)

    async def delete_file(self, file_id: str, user_id: str = "default") -> bool:
        """Delete a file"""
        return await self.backend.delete_file(file_id, user_id)

    async def list_files(
        self,
        user_id: str = "default",
        limit: int = 100,
        offset: int = 0,
        tags: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """List files for a user"""
        return await self.backend.list_files(user_id, limit, offset, tags)

    # ==================== Search Operations ====================

    async def semantic_search(
        self,
        user_id: str = "default",
        query: str = "",
        top_k: int = 5,
        min_score: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Semantic search across files"""
        return await self.backend.semantic_search(user_id, query, top_k, min_score, tags)

    async def rag_query(
        self,
        user_id: str = "default",
        query: str = "",
        top_k: int = 3,
        max_tokens: int = 500
    ) -> RAGResponse:
        """RAG query against files"""
        return await self.backend.rag_query(user_id, query, top_k, max_tokens)


# ==================== Singleton / Factory ====================

_storage_client: Optional[StorageClient] = None


async def get_storage_client(config: Optional[BackendConfig] = None) -> StorageClient:
    """Get or create storage client singleton"""
    global _storage_client

    if _storage_client is None:
        _storage_client = StorageClient(config)
        await _storage_client.initialize()

    return _storage_client


async def close_storage_client():
    """Close storage client"""
    global _storage_client

    if _storage_client:
        await _storage_client.close()
        _storage_client = None
