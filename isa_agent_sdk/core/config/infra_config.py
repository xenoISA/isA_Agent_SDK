#!/usr/bin/env python3
"""Infrastructure services configuration (from isA_Cloud)

All infrastructure service endpoints using native drivers.
"""
import os
from dataclasses import dataclass
from typing import Optional

def _bool(val: str) -> bool:
    return val.lower() == "true"

def _int(val: str, default: int) -> int:
    try:
        return int(val) if val else default
    except ValueError:
        return default

@dataclass
class InfraConfig:
    """Infrastructure service endpoints from isA_Cloud"""

    # ===========================================
    # PostgreSQL (native asyncpg - port 5432)
    # ===========================================
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Legacy gRPC aliases (deprecated)
    postgres_grpc_host: str = "localhost"
    postgres_grpc_port: int = 5432

    # ===========================================
    # Redis (native - port 6379)
    # ===========================================
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # ===========================================
    # Qdrant (native HTTP - port 6333)
    # ===========================================
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    # ===========================================
    # MinIO (native S3 - port 9000)
    # ===========================================
    minio_host: str = "localhost"
    minio_port: int = 9000
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    # ===========================================
    # Neo4j (native bolt - port 7687)
    # ===========================================
    neo4j_host: str = "localhost"
    neo4j_port: int = 7687
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"

    # ===========================================
    # NATS (native - port 4222)
    # ===========================================
    nats_host: str = "localhost"
    nats_port: int = 4222
    nats_url: Optional[str] = None

    # ===========================================
    # MQTT (native - port 1883)
    # ===========================================
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883

    # ===========================================
    # DuckDB (embedded - no network)
    # ===========================================
    duckdb_path: str = ":memory:"

    # ===========================================
    # Loki (logging)
    # ===========================================
    loki_enabled: bool = False
    loki_host: str = "localhost"
    loki_port: int = 3100
    loki_url: str = "http://localhost:3100"

    @classmethod
    def from_env(cls) -> 'InfraConfig':
        """Load infrastructure config from environment"""
        return cls(
            # PostgreSQL
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=_int(os.getenv("POSTGRES_PORT", "5432"), 5432),
            postgres_db=os.getenv("POSTGRES_DB", "postgres"),
            postgres_user=os.getenv("POSTGRES_USER", "postgres"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            postgres_grpc_host=os.getenv("POSTGRES_GRPC_HOST") or os.getenv("POSTGRES_HOST", "localhost"),
            postgres_grpc_port=_int(os.getenv("POSTGRES_GRPC_PORT") or os.getenv("POSTGRES_PORT", "5432"), 5432),

            # Redis
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=_int(os.getenv("REDIS_PORT", "6379"), 6379),
            redis_db=_int(os.getenv("REDIS_DB", "0"), 0),
            redis_password=os.getenv("REDIS_PASSWORD"),

            # Qdrant
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=_int(os.getenv("QDRANT_PORT", "6333"), 6333),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),

            # MinIO
            minio_host=os.getenv("MINIO_HOST", "localhost"),
            minio_port=_int(os.getenv("MINIO_PORT", "9000"), 9000),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            minio_secure=_bool(os.getenv("MINIO_SECURE", "false")),

            # Neo4j
            neo4j_host=os.getenv("NEO4J_HOST", "localhost"),
            neo4j_port=_int(os.getenv("NEO4J_PORT", "7687"), 7687),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4j"),

            # NATS
            nats_host=os.getenv("NATS_HOST", "localhost"),
            nats_port=_int(os.getenv("NATS_PORT", "4222"), 4222),
            nats_url=os.getenv("NATS_URL"),

            # MQTT
            mqtt_host=os.getenv("MQTT_HOST", "localhost"),
            mqtt_port=_int(os.getenv("MQTT_PORT", "1883"), 1883),

            # DuckDB
            duckdb_path=os.getenv("DUCKDB_PATH", ":memory:"),

            # Loki
            loki_enabled=_bool(os.getenv("LOKI_ENABLED", "false")),
            loki_host=os.getenv("LOKI_HOST", "localhost"),
            loki_port=_int(os.getenv("LOKI_PORT", "3100"), 3100),
            loki_url=os.getenv("LOKI_URL", "http://localhost:3100"),
        )
