"""
Centralized Logging Configuration with Loki Integration via gRPC

This module provides a centralized logging setup that sends logs to:
1. Console (stdout) - For local development and debugging
2. Loki (gRPC) - For centralized log aggregation via isa_common.LokiClient

The Loki service is discovered via Consul service discovery.
"""
import logging
import sys
import os
from typing import Optional

from isa_agent_sdk.core.config import settings


def setup_logger(
    name: str,
    level: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with centralized Loki integration via gRPC
    
    Logs are sent to:
    1. Console (stdout) - For local development and debugging
    2. Loki (gRPC) - For centralized log aggregation (if enabled)
    
    The Loki service is auto-discovered via Consul service discovery.
    If Consul discovery fails, it falls back to the configured loki_host and loki_port.
    
    Args:
        name: Logger name (e.g., "isA_Agent.API")
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Log format string (optional)
    
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("isA_Agent.API")
        >>> logger.info("API started successfully")
        # Logs to both console and Loki (if enabled)
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set log level
    log_level = (level or settings.log_level).upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Disable propagation to prevent duplicate logs
    logger.propagate = False

    # Log format
    formatter = logging.Formatter(format_str or settings.log_format)

    # 1. Console Handler (for local development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Loki gRPC Handler (for centralized logging)
    if settings.loki_enabled:
        try:
            from isa_common.loki_client import LokiClient

            # Custom Loki handler that uses isa_common.LokiClient
            class LokiGrpcHandler(logging.Handler):
                """
                Custom Loki handler that uses isa_common.LokiClient via gRPC
                
                This handler automatically pushes logs to Loki using the LokiClient
                from isa_common, which handles gRPC connection management, retries,
                and efficient batching.
                """

                def __init__(self, loki_host: str, loki_port: int, user_id: str, service_labels: dict):
                    super().__init__()
                    self.loki_host = loki_host
                    self.loki_port = loki_port
                    self.user_id = user_id
                    self.service_labels = service_labels
                    self.client = None
                    self._error_count = 0
                    self._success_count = 0

                def emit(self, record):
                    """Send log record to Loki via gRPC"""
                    try:
                        # Lazy initialization of LokiClient (only when needed)
                        if self.client is None:
                            self.client = LokiClient(
                                host=self.loki_host,
                                port=self.loki_port,
                                user_id=self.user_id
                            )
                            self.client.__enter__()  # Open connection
                        
                        # Format the log message
                        log_entry = self.format(record)
                        
                        # Build labels with level
                        labels = self.service_labels.copy()
                        labels['level'] = record.levelname.lower()
                        
                        # Push to Loki using simple API
                        self.client.push_log(
                            message=log_entry,
                            labels=labels
                        )
                        
                        self._success_count += 1
                        
                        # Debug: log first few successful pushes
                        if self._success_count <= 2:
                            print(f"[LOKI_DEBUG] Successfully pushed log to Loki: {log_entry[:80]}", flush=True)
                            
                    except Exception as e:
                        # Graceful degradation - don't fail the application
                        self._error_count += 1
                        if self._error_count <= 3:
                            print(f"[LOKI_ERROR] Failed to push log to Loki: {e}", flush=True)
                            import traceback
                            traceback.print_exc()

                def close(self):
                    """Clean up Loki client connection"""
                    if self.client:
                        try:
                            self.client.__exit__(None, None, None)
                        except:
                            pass
                    super().close()

            # Discover Loki service via Consul
            loki_host = settings.loki_host
            loki_port = settings.loki_port

            # Try Consul discovery first
            if settings.consul_enabled:
                try:
                    from isa_common.consul_client import ConsulRegistry

                    consul = ConsulRegistry(
                        consul_host=settings.consul_host,
                        consul_port=settings.consul_port
                    )

                    # Discover Loki gRPC service (registered as "loki-grpc-service")
                    loki_url = consul.get_service_address(
                        "loki-grpc-service",
                        fallback_url=f"http://{loki_host}:{loki_port}",
                        max_retries=1
                    )
                    
                    # Parse host and port from discovered URL
                    if loki_url and loki_url.startswith("http://"):
                        loki_url = loki_url.replace("http://", "")
                        if ":" in loki_url:
                            loki_host, port_str = loki_url.split(":")
                            loki_port = int(port_str)
                        else:
                            loki_host = loki_url
                    
                    if name == "isA_Agent":
                        print(f"[LOKI] Discovered Loki service via Consul: {loki_host}:{loki_port}", flush=True)
                        
                except Exception as e:
                    # Consul discovery failed, use configured values
                    if name == "isA_Agent":
                        print(f"[LOKI] Consul discovery failed, using configured Loki: {loki_host}:{loki_port}", flush=True)
                        print(f"[LOKI] Discovery error: {e}", flush=True)

            # Extract service name and logger component
            # e.g., "isA_Agent.API" -> service="agent", logger="API"
            service_name = "agent_service"
            logger_component = name.replace("isA_Agent.", "").replace("isA_Agent", "main")

            # Labels for Loki (used for filtering and searching)
            service_labels = {
                "service": service_name,
                "logger": logger_component,
                "environment": os.getenv("ENVIRONMENT", "development"),
                "job": "agent_service"
            }

            # Create Loki gRPC handler
            loki_handler = LokiGrpcHandler(
                loki_host=loki_host,
                loki_port=loki_port,
                user_id=service_name,
                service_labels=service_labels
            )

            # Set formatter
            loki_handler.setFormatter(formatter)

            # Only send INFO and above to Loki (reduce network traffic)
            loki_handler.setLevel(logging.INFO)

            logger.addHandler(loki_handler)

            # Log successful Loki integration (only once)
            if name == "isA_Agent":
                logger.info(f"✅ Centralized logging enabled | loki={loki_host}:{loki_port} via gRPC")

        except ImportError as e:
            # isa_common not installed
            if name == "isA_Agent":
                logger.warning(f"⚠️  Could not setup Loki handler: {e}")
                logger.warning(f"⚠️  Please install isa-common: pip install isa-common")
        except Exception as e:
            # Loki unavailable or other error - don't fail the app
            if name == "isA_Agent":
                logger.warning(f"⚠️  Could not connect to Loki at {settings.loki_host}:{settings.loki_port}: {e}")
                logger.info("📝 Logging to console only")

    return logger


# Create default application loggers
app_logger = setup_logger("isA_Agent")
api_logger = setup_logger("isA_Agent.API")
agent_logger = setup_logger("isA_Agent.SmartAgent")
tracer_logger = setup_logger("isA_Agent.Tracer")
