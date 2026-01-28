"""
Billing Service - Official LangChain Custom Events Integration

Uses official LangChain custom event dispatching for tracking non-LangChain components:
- adispatch_custom_event() for custom model/tool call tracking
- BaseCallbackHandler.on_custom_event() for event handling
- 1 model call = 1 credit, 1 tool call = 2 credits

Based on official LangChain patterns for custom operations.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import aiohttp
from functools import wraps

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    jwt = None
    JWT_AVAILABLE = False
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import adispatch_custom_event

from isa_agent_sdk.utils.logger import api_logger


@dataclass
class BillingResult:
    """Result of a billing operation"""
    success: bool
    model_calls: int
    tool_calls: int
    total_credits: float
    credits_remaining: float
    error_message: Optional[str] = None
    execution_details: Optional[Dict[str, Any]] = None


class BillingCallbackHandler(BaseCallbackHandler):
    """
    Official LangChain callback handler for custom billing events
    
    Listens for custom events dispatched by instrumented model/tool calls:
    - "custom_model_call" events = 1 credit each
    - "custom_tool_call" events = 2 credits each
    """
    
    def __init__(self, user_id: str, session_id: str, auth_token: Optional[str] = None):
        super().__init__()
        self.user_id = user_id
        self.session_id = session_id
        self.auth_token = auth_token
        
        # Simple pricing
        self.model_call_cost = 1.0
        self.tool_call_cost = 2.0
        
        # Execution tracking
        self.model_calls = 0
        self.tool_calls = 0
        self.execution_log = []
        self.start_time = time.time()
    
    def on_custom_event(self, name: str, data: Dict[str, Any], **kwargs) -> None:
        """Handle custom billing events using official LangChain pattern"""
        
        if name == "custom_model_call":
            self.model_calls += 1
            execution_record = {
                'type': 'model_call',
                'event_name': name,
                'data': data,
                'timestamp': time.time(),
                'call_number': self.model_calls,
                'credits': self.model_call_cost
            }
            self.execution_log.append(execution_record)
            api_logger.debug(f"💰 Model call #{self.model_calls} detected: {data.get('node_name', 'unknown')}")
            
        elif name == "custom_tool_call":
            self.tool_calls += 1
            execution_record = {
                'type': 'tool_call',
                'event_name': name,
                'data': data,
                'timestamp': time.time(),
                'call_number': self.tool_calls,
                'credits': self.tool_call_cost
            }
            self.execution_log.append(execution_record)
            api_logger.debug(f"💰 Tool call #{self.tool_calls} detected: {data.get('tool_name', 'unknown')}")
    
    def calculate_total_credits(self) -> float:
        """Calculate total credits based on execution counts"""
        total = self.model_calls * self.model_call_cost + self.tool_calls * self.tool_call_cost
        return max(total, 1.0)  # Minimum 1 credit
    
    def get_billing_summary(self) -> Dict[str, Any]:
        """Get comprehensive billing summary"""
        total_credits = self.calculate_total_credits()
        execution_time = time.time() - self.start_time
        
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "total_credits": total_credits,
            "execution_time_seconds": round(execution_time, 2),
            "breakdown": {
                "model_cost": self.model_calls * self.model_call_cost,
                "tool_cost": self.tool_calls * self.tool_call_cost
            },
            "pricing": {
                "model_call_cost": self.model_call_cost,
                "tool_call_cost": self.tool_call_cost
            },
            "execution_log": self.execution_log
        }


class BillingService:
    """
    Billing service using official LangChain custom events
    
    Usage:
    1. Create billing handler
    2. Add handler to graph config
    3. Use instrumentation decorators on your custom functions
    4. Events are automatically tracked via callbacks
    """
    
    def __init__(self):
        from isa_agent_sdk.core.config import settings
        self.user_service = None
        self.user_service_url = settings.account_service_url
        self.wallet_service_url = settings.wallet_service_url
        self.jwt_secret = "your-secret-key"  # Should be in config/env
        self.service_status = "unknown"
        self.wallet_service_available = False
        self._init_user_service()
        self._check_wallet_service()
    
    def _init_user_service(self):
        """Initialize user service with health check"""
        try:
            api_logger.info("Initializing UserServiceMicroservices...")
            from .user_service import UserServiceMicroservices
            from .user_service import MicroserviceConfig

            # Create config with Consul service discovery
            config = MicroserviceConfig()
            api_logger.info(f"Created MicroserviceConfig with Consul discovery:")
            api_logger.info(f"  account_service: {config.account_service_url}")
            api_logger.info(f"  wallet_service: {config.wallet_service_url}")
            api_logger.info(f"  session_service: {config.session_service_url}")
            api_logger.info(f"  storage_service: {config.storage_service_url}")

            self.user_service = UserServiceMicroservices(config=config)
            api_logger.info("UserServiceMicroservices instance created")
            
            # Test User Service health
            health_result = self.user_service.health_check()
            if health_result.get("status") == "error":
                api_logger.warning(f"User Service health check failed: {health_result.get('error', 'unknown')}")
                self.user_service = None
                self.service_status = "unhealthy"
            else:
                api_logger.info("User service initialized and healthy")
                self.service_status = "healthy"
        except Exception as e:
            api_logger.error(f"User service initialization failed: {e}", exc_info=True)
            self.user_service = None
            self.service_status = "unavailable"
    
    def _check_wallet_service(self):
        """Check if wallet service is available"""
        try:
            import requests
            response = requests.get(f"{self.wallet_service_url}/health", timeout=2)
            if response.status_code == 200:
                self.wallet_service_available = True
                api_logger.info("Wallet service available at port 8209")
            else:
                self.wallet_service_available = False
                api_logger.warning(f"Wallet service health check failed with status: {response.status_code}")
        except Exception as e:
            self.wallet_service_available = False
            api_logger.warning(f"Wallet service not available at port 8209: {e}")
    
    async def _generate_jwt_token(self, user_id: str) -> Optional[str]:
        """
        Generate JWT token for User Service authentication
        
        Args:
            user_id: User ID to generate token for
            
        Returns:
            JWT token string or None if generation fails
        """
        try:
            # Try to get token from User Service /auth/dev-token endpoint
            async with aiohttp.ClientSession() as session:
                payload = {
                    "user_id": user_id,
                    "role": "user"
                }
                
                try:
                    async with session.post(
                        f"{self.user_service_url}/auth/dev-token",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            token = data.get("access_token")
                            if token:
                                api_logger.info(f"JWT token generated successfully for user: {user_id}")
                                return token
                        else:
                            api_logger.warning(f"Token generation failed with status: {response.status}")
                            
                except aiohttp.ClientError as e:
                    api_logger.warning(f"User Service connection error: {e}")
                
                # Fallback: Generate token locally (for development/testing)
                if not JWT_AVAILABLE:
                    api_logger.warning("PyJWT not installed, cannot generate fallback token")
                    return None

                api_logger.info("Generating fallback JWT token locally")
                payload = {
                    "user_id": user_id,
                    "role": "user",
                    "exp": int(time.time()) + 3600,  # 1 hour expiration
                    "iat": int(time.time())
                }

                token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
                return token
                
        except Exception as e:
            api_logger.error(f"JWT token generation error: {e}")
            return None
    
    async def _consume_credits_via_wallet(
        self,
        user_id: str,
        amount: float,
        description: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Consume credits via wallet service"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"
                
                payload = {
                    "amount": amount,
                    "service": "isA_agent",
                    "reason": description,
                    "usage_record_id": int(time.time())  # Optional integer ID
                }
                
                async with session.post(
                    f"{self.wallet_service_url}/api/v1/users/{user_id}/credits/consume",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        # Get remaining balance from response
                        wallet_balance = data.get("balance")  # Direct balance field
                        if wallet_balance is None:
                            wallet_balance = data.get("data", {}).get("remaining_balance", 0)
                        
                        api_logger.info(f"Wallet service: Consumed {amount} credits, remaining: {wallet_balance}")
                        return {
                            "success": True,
                            "remaining_credits": float(wallet_balance) if wallet_balance else 0,
                            "transaction_id": data.get("transaction_id")
                        }
                    else:
                        error_msg = data.get("detail", "Unknown error")
                        api_logger.error(f"Wallet service error: {error_msg}")
                        return {
                            "success": False,
                            "error": error_msg
                        }
                        
        except Exception as e:
            api_logger.error(f"Wallet service connection error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_billing_handler(
        self, 
        user_id: str, 
        session_id: str,
        auth_token: Optional[str] = None
    ) -> BillingCallbackHandler:
        """Create a billing callback handler"""
        return BillingCallbackHandler(user_id, session_id, auth_token)
    
    async def finalize_billing(
        self, 
        billing_handler: BillingCallbackHandler
    ) -> BillingResult:
        """Finalize billing after execution completion"""
        try:
            billing_summary = billing_handler.get_billing_summary()
            total_credits = billing_summary["total_credits"]
            model_calls = billing_summary["model_calls"]
            tool_calls = billing_summary["tool_calls"]
            user_id = billing_summary["user_id"]
            session_id = billing_summary["session_id"]
            
            # Prepare usage data for User Service with all required fields
            usage_data = {
                "user_id": user_id,
                "session_id": session_id,
                "endpoint": "/api/chat",
                "event_type": "ai_chat",
                "credits_charged": total_credits,
                "cost_usd": total_credits * 0.002,  # Assume 1 credit = $0.002
                "calculation_method": "simple_credit_billing",
                "tokens_used": billing_summary.get("total_tokens", model_calls + tool_calls),
                "prompt_tokens": billing_summary.get("prompt_tokens", model_calls),
                "completion_tokens": billing_summary.get("completion_tokens", tool_calls),
                "model_name": "gpt-4",  # Default model
                "provider": "internal",
                "tool_name": "isA_agent",
                "operation_name": "chat_completion",
                "billing_metadata": {
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "execution_time": billing_summary["execution_time_seconds"],
                    "billing_method": "langchain_custom_events"
                },
                "request_data": {
                    "billing_method": "langchain_custom_events",
                    "pricing": billing_summary["pricing"]
                },
                "response_data": {
                    "execution_log": billing_summary["execution_log"]
                }
            }
            
            remaining_credits = None  # Will be set after successful service call
            
            # Try wallet service first (if available)
            if self.wallet_service_available:
                try:
                    auth_token = billing_handler.auth_token
                    if not auth_token:
                        api_logger.warning("No auth token provided - trying without authentication")
                    
                    wallet_result = await self._consume_credits_via_wallet(
                        user_id=user_id,
                        amount=total_credits,
                        description=f"AI Chat: {model_calls} model calls, {tool_calls} tool calls",
                        auth_token=auth_token
                    )
                    
                    if wallet_result.get("success"):
                        remaining_credits = wallet_result.get("remaining_credits", 0)
                        api_logger.info(f"💰 Wallet service billing successful: {total_credits} credits consumed, {remaining_credits} remaining")
                        
                        # Return successful billing result
                        return BillingResult(
                            success=True,
                            model_calls=model_calls,
                            tool_calls=tool_calls,
                            total_credits=total_credits,
                            credits_remaining=remaining_credits,
                            execution_details=billing_summary
                        )
                    else:
                        api_logger.warning(f"Wallet service billing failed: {wallet_result.get('error')}, falling back to User Service")
                        
                except Exception as e:
                    api_logger.warning(f"Wallet service error: {e}, falling back to User Service")
            
            # Fallback to User Service if wallet service failed or unavailable
            if self.user_service and remaining_credits is None:
                try:
                    # Use auth token from frontend/API request
                    auth_token = billing_handler.auth_token
                    if not auth_token:
                        api_logger.error("No auth token provided from frontend - cannot determine real credit balance")
                        raise Exception("Authentication required for accurate credit billing")
                    
                    api_logger.debug(f"Billing with auth token: {auth_token[:20]}... (length: {len(auth_token)})")
                    
                    # Record usage with frontend-provided authentication
                    usage_result = self.user_service.record_usage(
                        user_id=user_id,
                        session_id=session_id,
                        usage_data=usage_data,
                        auth_token=auth_token
                    )
                    
                    # Consume credits if usage was recorded successfully
                    if usage_result.get("success", False):
                        usage_record_id = usage_result.get("data", {}).get("id")
                        api_logger.debug(f"Usage recorded successfully with ID: {usage_record_id}")
                        
                        credit_result = self.user_service.consume_credits(
                            user_id=user_id,
                            amount=total_credits,
                            description=f"AI Chat: {model_calls} model calls, {tool_calls} tool calls",
                            auth_token=auth_token
                        )

                        if credit_result.get("success", False):
                            # Handle different response formats from wallet service
                            # Format 1: direct remaining_credits field
                            remaining_credits = credit_result.get("remaining_credits")

                            # Format 2: nested in data field
                            if remaining_credits is None and "data" in credit_result:
                                data = credit_result.get("data", {})
                                remaining_credits = data.get("remaining_credits") or data.get("balance")

                            # Format 3: nested in wallet/account info
                            if remaining_credits is None and "wallet" in credit_result:
                                wallet = credit_result.get("wallet", {})
                                remaining_credits = wallet.get("balance") or wallet.get("credits")

                            if remaining_credits is not None:
                                api_logger.info(f"💰 User Service billing completed: {total_credits} credits consumed, {remaining_credits} remaining")
                            else:
                                api_logger.warning(f"User Service returned success but no remaining_credits value. Response: {credit_result}")
                                # Don't fail - use 0 as fallback and mark as warning
                                remaining_credits = 0.0
                        else:
                            error_msg = credit_result.get('error', 'unknown')
                            api_logger.error(f"Credit consumption failed: {error_msg}")
                            raise Exception(f"Credit billing failed: {error_msg}")
                    else:
                        error_msg = usage_result.get('error', 'unknown')
                        api_logger.error(f"Usage recording failed: {error_msg}")
                        raise Exception(f"Usage recording failed: {error_msg}")
                        
                except Exception as e:
                    api_logger.error(f"User Service billing error (non-critical): {e}")
                    # Return billing failure but don't crash the service
                    return BillingResult(
                        success=False,
                        model_calls=billing_handler.model_calls,
                        tool_calls=billing_handler.tool_calls,
                        total_credits=billing_handler.calculate_total_credits(),
                        credits_remaining=0.0,
                        error_message=f"Billing service temporarily unavailable: {str(e)}"
                    )
            else:
                api_logger.error("User Service not available - cannot process billing")
                return BillingResult(
                    success=False,
                    model_calls=billing_handler.model_calls,
                    tool_calls=billing_handler.tool_calls,
                    total_credits=billing_handler.calculate_total_credits(),
                    credits_remaining=0.0,
                    error_message="Billing service not configured"
                )
            
            # Verify we have a valid remaining_credits value
            if remaining_credits is None:
                api_logger.error("No valid remaining_credits obtained from User Service")
                return BillingResult(
                    success=False,
                    model_calls=billing_handler.model_calls,
                    tool_calls=billing_handler.tool_calls,
                    total_credits=billing_handler.calculate_total_credits(),
                    credits_remaining=0.0,
                    error_message="Unable to determine credit balance"
                )
            
            api_logger.info(f"💰 Billed {total_credits} credits: {model_calls} model calls, {tool_calls} tool calls (user: {user_id})")
            
            return BillingResult(
                success=True,
                model_calls=model_calls,
                tool_calls=tool_calls,
                total_credits=total_credits,
                credits_remaining=remaining_credits,
                execution_details=billing_summary
            )
            
        except Exception as e:
            api_logger.error(f"Billing finalization error: {e}")
            return BillingResult(
                success=False,
                model_calls=0,
                tool_calls=0,
                total_credits=0.0,
                credits_remaining=0.0,
                error_message=str(e)
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get billing service status"""
        return {
            "user_service_status": self.service_status,
            "user_service_url": self.user_service_url,
            "user_service_available": self.user_service is not None,
            "wallet_service_url": self.wallet_service_url,
            "wallet_service_available": self.wallet_service_available,
            "billing_method": "langchain_custom_events",
            "primary_billing": "wallet_service" if self.wallet_service_available else "user_service"
        }


# Global billing service
billing_service = BillingService()


# === Official LangChain Instrumentation Decorators ===

def track_model_call(node_name: str = "unknown"):
    """
    Official LangChain decorator for tracking custom model calls
    
    Usage:
        @track_model_call("ReasonNode")
        async def call_model(self, messages, tools=None):
            # Your custom model implementation
            return response
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract config if available
            config = None
            if 'config' in kwargs:
                config = kwargs['config']
            elif len(args) > 0 and hasattr(args[0], 'config'):
                config = getattr(args[0], 'config', None)
            
            try:
                # Dispatch custom event using official LangChain method
                await adispatch_custom_event(
                    "custom_model_call",
                    {
                        "node_name": node_name,
                        "function": func.__name__,
                        "timestamp": time.time(),
                        "status": "started"
                    },
                    config=config
                )
                
                # Execute original function
                result = await func(*args, **kwargs)
                
                # Dispatch completion event
                await adispatch_custom_event(
                    "custom_model_call_complete",
                    {
                        "node_name": node_name,
                        "function": func.__name__,
                        "status": "completed"
                    },
                    config=config
                )
                
                return result
                
            except Exception as e:
                # Dispatch error event
                await adispatch_custom_event(
                    "custom_model_call_error",
                    {
                        "node_name": node_name,
                        "function": func.__name__,
                        "error": str(e)
                    },
                    config=config
                )
                raise
        
        return wrapper
    return decorator


def track_tool_call(tool_name: str = "unknown"):
    """
    Official LangChain decorator for tracking custom tool calls
    
    Usage:
        @track_tool_call("web_search")
        async def mcp_call_tool(self, tool_name, tool_args, config):
            # Your custom tool implementation
            return result
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract config if available
            config = None
            if 'config' in kwargs:
                config = kwargs['config']
            elif len(args) > 0 and hasattr(args[0], 'config'):
                config = getattr(args[0], 'config', None)
            
            try:
                # Dispatch custom event
                await adispatch_custom_event(
                    "custom_tool_call",
                    {
                        "tool_name": tool_name,
                        "function": func.__name__,
                        "timestamp": time.time(),
                        "status": "started"
                    },
                    config=config
                )
                
                # Execute original function
                result = await func(*args, **kwargs)
                
                # Dispatch completion event
                await adispatch_custom_event(
                    "custom_tool_call_complete",
                    {
                        "tool_name": tool_name,
                        "function": func.__name__,
                        "status": "completed"
                    },
                    config=config
                )
                
                return result
                
            except Exception as e:
                # Dispatch error event
                await adispatch_custom_event(
                    "custom_tool_call_error",
                    {
                        "tool_name": tool_name,
                        "function": func.__name__,
                        "error": str(e)
                    },
                    config=config
                )
                raise
        
        return wrapper
    return decorator


# === Integration Helper Functions ===

def create_billing_handler(
    user_id: str, 
    session_id: str,
    auth_token: Optional[str] = None
) -> BillingCallbackHandler:
    """Create a billing handler"""
    return billing_service.create_billing_handler(user_id, session_id, auth_token)


async def stream_with_billing(
    graph,
    inputs: Dict[str, Any],
    user_id: str,
    session_id: str,
    trace_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    stream_mode: list = ["updates", "messages", "custom"]
):
    """
    Stream graph execution with automatic billing tracking
    
    Usage in your chat service:
        billing_handler = create_billing_handler(user_id, session_id, trace_id)
        config = {"callbacks": [billing_handler]}
        
        async for chunk in stream_with_billing(graph, inputs, user_id, session_id, config=config):
            # Handle normal streaming
            if chunk[0] == "billing":
                billing_result = chunk[1]  # Final billing result
            else:
                # Normal stream processing
                yield chunk
    """
    # Create billing handler
    billing_handler = create_billing_handler(user_id, session_id, trace_id)
    
    # Prepare config with callbacks
    execution_config = config.copy() if config else {}
    callbacks = execution_config.get("callbacks", [])
    callbacks.append(billing_handler)
    execution_config["callbacks"] = callbacks
    
    try:
        # Stream with billing tracking
        async for chunk in graph.astream(inputs, config=execution_config, stream_mode=stream_mode):
            yield chunk
        
        # Finalize billing after stream completion
        billing_result = await billing_service.finalize_billing(billing_handler)
        
        # Yield billing result as final item
        yield ("billing", billing_result)
        
    except Exception as e:
        api_logger.error(f"Stream execution with billing error: {e}")
        billing_result = BillingResult(
            success=False,
            model_calls=billing_handler.model_calls,
            tool_calls=billing_handler.tool_calls,
            total_credits=billing_handler.calculate_total_credits(),
            credits_remaining=0.0,
            error_message=str(e)
        )
        yield ("billing", billing_result)


# === Usage Instructions ===
"""
To integrate with your existing code:

1. Add decorators to your base_node methods:

   In base_node.py:
   
   @track_model_call("BaseNode")
   async def call_model(self, messages, tools=None):
       # Your existing model implementation
       return response
   
   @track_tool_call("MCP")  
   async def mcp_call_tool(self, tool_name, tool_args, config):
       # Your existing tool implementation
       return result

2. Use billing-enabled streaming in chat_service.py:

   async def stream_chat():
       billing_handler = create_billing_handler(user_id, session_id, trace_id)
       config = {"callbacks": [billing_handler], ...}
       
       async for chunk in graph.astream(state, config=config, stream_mode=["updates", "messages", "custom"]):
           # Normal streaming
           yield chunk
       
       # Get final billing
       billing_result = await billing_service.finalize_billing(billing_handler)

That's it! The official LangChain custom events will automatically track your calls.
"""