#!/usr/bin/env python3
"""
Timeout configuration for production stability
统一超时配置管理
"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TimeoutConfig:
    """Centralized timeout configuration"""
    
    # ISA Client timeouts (seconds)
    isa_client_reasoning: float = 60.0      # Reasoning node timeout
    isa_client_response: float = 30.0       # Response formatting timeout
    isa_client_default: float = 45.0        # Default ISA client timeout
    
    # API timeouts (seconds)
    api_request: float = 120.0              # Total API request timeout
    sse_stream: float = 300.0               # SSE stream timeout (5 minutes)
    
    # Database timeouts (seconds)
    db_query: float = 10.0                  # Database query timeout
    db_connection: float = 5.0              # Database connection timeout
    
    # MCP timeouts (seconds)
    mcp_tool_call: float = 30.0             # MCP tool execution timeout
    mcp_resource: float = 10.0              # MCP resource retrieval timeout
    mcp_initialization: float = 15.0        # MCP initialization timeout
    
    # Human-in-the-loop timeouts (seconds)
    human_approval: float = 300.0           # Human approval timeout (5 minutes)
    
    # Retry configuration
    max_retries: int = 3                    # Maximum retry attempts
    retry_delay_base: float = 1.0           # Base delay for exponential backoff
    retry_delay_max: float = 10.0           # Maximum retry delay
    
    @classmethod
    def from_env(cls) -> 'TimeoutConfig':
        """Load timeout configuration from environment variables"""
        return cls(
            isa_client_reasoning=float(os.getenv('ISA_TIMEOUT_REASONING', '60.0')),
            isa_client_response=float(os.getenv('ISA_TIMEOUT_RESPONSE', '30.0')),
            isa_client_default=float(os.getenv('ISA_TIMEOUT_DEFAULT', '45.0')),
            
            api_request=float(os.getenv('API_TIMEOUT_REQUEST', '120.0')),
            sse_stream=float(os.getenv('API_TIMEOUT_SSE', '300.0')),
            
            db_query=float(os.getenv('DB_TIMEOUT_QUERY', '10.0')),
            db_connection=float(os.getenv('DB_TIMEOUT_CONNECTION', '5.0')),
            
            mcp_tool_call=float(os.getenv('MCP_TIMEOUT_TOOL', '30.0')),
            mcp_resource=float(os.getenv('MCP_TIMEOUT_RESOURCE', '10.0')),
            mcp_initialization=float(os.getenv('MCP_TIMEOUT_INIT', '15.0')),
            
            human_approval=float(os.getenv('HUMAN_TIMEOUT_APPROVAL', '300.0')),
            
            max_retries=int(os.getenv('RETRY_MAX_ATTEMPTS', '3')),
            retry_delay_base=float(os.getenv('RETRY_DELAY_BASE', '1.0')),
            retry_delay_max=float(os.getenv('RETRY_DELAY_MAX', '10.0'))
        )


# Global timeout configuration instance
timeout_config = TimeoutConfig.from_env()


class TimeoutManager:
    """Centralized timeout and retry management"""
    
    @staticmethod
    async def with_timeout_and_retry(
        operation,
        timeout: float,
        operation_name: str,
        max_retries: Optional[int] = None,
        retry_on_timeout: bool = True
    ):
        """Execute operation with timeout and retry logic"""
        import asyncio
        import logging
        
        logger = logging.getLogger(__name__)
        retries = max_retries if max_retries is not None else timeout_config.max_retries
        
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(operation, timeout=timeout)
                
            except asyncio.TimeoutError:
                if attempt < retries and retry_on_timeout:
                    delay = min(
                        timeout_config.retry_delay_base * (2 ** attempt),
                        timeout_config.retry_delay_max
                    )
                    logger.warning(
                        f"{operation_name} timeout (attempt {attempt + 1}/{retries + 1}), "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"{operation_name} failed after {retries + 1} attempts due to timeout")
                    raise asyncio.TimeoutError(f"{operation_name} timeout after {timeout}s")
                    
            except Exception as e:
                if attempt < retries:
                    delay = min(
                        timeout_config.retry_delay_base * (2 ** attempt),
                        timeout_config.retry_delay_max
                    )
                    logger.warning(
                        f"{operation_name} error (attempt {attempt + 1}/{retries + 1}): {e}, "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"{operation_name} failed after {retries + 1} attempts: {e}")
                    raise