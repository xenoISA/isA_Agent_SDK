"""
Circuit Breaker Pattern Implementation
Provides fault tolerance for external service calls
"""
import asyncio
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from isa_agent_sdk.utils.logger import api_logger


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    request_timeout: float = 30.0   # seconds
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        api_logger.info(f"🔌 Circuit breaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        self.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                self.failed_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next attempt in {self.next_attempt_time - time.time():.1f}s"
                )
            else:
                # Try half-open state
                self.state = CircuitState.HALF_OPEN
                api_logger.info(f"🔌 Circuit breaker '{self.name}' entering HALF_OPEN state")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.request_timeout
            )
            
            # Success - reset failure count
            await self._on_success()
            return result
            
        except asyncio.TimeoutError as e:
            await self._on_failure(f"Timeout after {self.config.request_timeout}s")
            raise CircuitBreakerTimeoutError(f"Request timeout: {e}")
            
        except self.config.expected_exception as e:
            await self._on_failure(str(e))
            raise
        
        except Exception as e:
            await self._on_failure(f"Unexpected error: {e}")
            raise
    
    async def _on_success(self):
        """Handle successful request"""
        self.successful_requests += 1
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            api_logger.info(f"🔌 Circuit breaker '{self.name}' returned to CLOSED state")
    
    async def _on_failure(self, error_msg: str):
        """Handle failed request"""
        self.failed_requests += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        api_logger.warning(f"🔌 Circuit breaker '{self.name}' failure #{self.failure_count}: {error_msg}")
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            api_logger.error(
                f"🔌 Circuit breaker '{self.name}' OPENED after {self.failure_count} failures. "
                f"Will retry in {self.config.recovery_timeout}s"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": self.next_attempt_time if self.state == CircuitState.OPEN else None
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.next_attempt_time = 0
        api_logger.info(f"🔌 Circuit breaker '{self.name}' manually reset")


class CircuitBreakerError(Exception):
    """Base circuit breaker exception"""
    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is open"""
    pass


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Request timed out"""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()


# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()


# Service-specific circuit breakers
def get_mcp_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for MCP server"""
    config = CircuitBreakerConfig(
        failure_threshold=10,  # More lenient - allow more failures before opening
        recovery_timeout=10.0,  # Faster recovery - retry after 10s instead of 30s
        request_timeout=15.0    # Longer timeout - 15s for MCP tools
    )
    return circuit_manager.get_breaker("mcp_server", config)


def get_database_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for database"""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        request_timeout=15.0
    )
    return circuit_manager.get_breaker("database", config)


def get_isa_model_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for ISA model API"""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=45.0,
        request_timeout=20.0
    )
    return circuit_manager.get_breaker("isa_model", config)