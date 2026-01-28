"""
User Service Microservices Client

Handles all communication with microservices for:
- Account management (account_service)
- Credit/wallet management (wallet_service) 
- Session management (session_service)
- File storage management (storage_service)

Replaces the monolithic User Service with distributed microservices.
"""

import requests
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from isa_agent_sdk.utils.logger import api_logger
from isa_agent_sdk.core.resilience.system_protection import SystemProtectionMiddleware


class MicroserviceConfig:
    """Configuration for microservices endpoints with dynamic Consul discovery"""
    
    def __init__(self):
        self.timeout = 30.0
    
    @property
    def account_service_url(self) -> str:
        """Get account service URL via Consul discovery"""
        from isa_agent_sdk.core.config import settings
        return settings.account_service_url
    
    @property
    def wallet_service_url(self) -> str:
        """Get wallet service URL via Consul discovery"""
        from isa_agent_sdk.core.config import settings
        return settings.wallet_service_url
    
    @property
    def session_service_url(self) -> str:
        """Get session service URL via Consul discovery"""
        from isa_agent_sdk.core.config import settings
        return settings.session_service_url
    
    @property
    def storage_service_url(self) -> str:
        """Get storage service URL via Consul discovery"""
        from isa_agent_sdk.core.config import settings
        return settings.storage_service_url


class UserServiceMicroservices:
    """Client for microservices-based user operations"""
    
    def __init__(self, config: Optional[MicroserviceConfig] = None):
        self.config = config or MicroserviceConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        # System protection can be added later if needed
        # self.protection = SystemProtectionMiddleware()
    
    def close(self):
        """Close HTTP session"""
        if self.session:
            self.session.close()
            self.session = None
    
    def _get_headers(self, auth_token: Optional[str] = None) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make protected HTTP request with circuit breaker"""
        try:
            # Ensure timeout is set if not provided
            if 'timeout' not in kwargs:
                kwargs['timeout'] = self.config.timeout
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            api_logger.error(f"Request failed {method} {url}: {e}")
            return {"success": False, "error": str(e), "status": "error"}
    
    # Health check methods
    
    def health_check(self) -> Dict[str, Any]:
        """Check all microservices health"""
        services = {
            "account_service": f"{self.config.account_service_url}/health",
            "wallet_service": f"{self.config.wallet_service_url}/health", 
            "session_service": f"{self.config.session_service_url}/health",
            "storage_service": f"{self.config.storage_service_url}/health"
        }
        
        results = {}
        overall_healthy = True
        
        for service_name, health_url in services.items():
            try:
                response = self.session.get(health_url, timeout=5.0)
                response.raise_for_status()
                results[service_name] = response.json()
            except Exception as e:
                api_logger.error(f"{service_name} health check failed: {e}")
                results[service_name] = {"status": "error", "error": str(e)}
                overall_healthy = False
        
        return {
            "status": "healthy" if overall_healthy else "error",
            "services": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Account management methods (account_service)
    
    def ensure_user(
        self,
        user_id: str,
        email: str,
        name: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ensure user exists via account_service"""
        headers = self._get_headers(auth_token)
        
        payload = {
            "user_id": user_id,
            "email": email,
            "name": name,
            "is_active": True
        }
        
        return self._make_request(
            "POST",
            f"{self.config.account_service_url}/api/v1/accounts/ensure",
            headers=headers,
            json=payload
        )
    
    def get_user_profile(
        self,
        user_id: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user profile via account_service"""
        headers = self._get_headers(auth_token)
        
        return self._make_request(
            "GET",
            f"{self.config.account_service_url}/api/v1/accounts/profile/{user_id}",
            headers=headers
        )
    
    # Credit/wallet management methods (wallet_service)
    
    def get_credit_balance(
        self,
        user_id: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user credit balance via wallet_service"""
        headers = self._get_headers(auth_token)
        
        # Use backward compatibility endpoint
        return self._make_request(
            "GET",
            f"{self.config.wallet_service_url}/api/v1/users/{user_id}/credits/balance",
            headers=headers
        )
    
    def consume_credits(
        self,
        user_id: str,
        amount: float,
        description: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Consume user credits via wallet_service - Best Practice: Wallet as single source of truth"""
        headers = self._get_headers(auth_token)

        # Step 1: Get user's wallet from wallet_service (single source of truth)
        balance_result = self.get_credit_balance(user_id, auth_token)

        if not balance_result.get("success"):
            return {
                "success": False,
                "error": f"Failed to get wallet: {balance_result.get('error', 'unknown')}"
            }

        wallet_id = balance_result.get("wallet_id")
        if not wallet_id:
            return {
                "success": False,
                "error": "No wallet found for user"
            }

        # Step 2: Consume credits from the wallet using proper wallet_service endpoint
        payload = {
            "amount": amount,
            "description": description,
            "service": "isA_agent",
            "reason": "AI chat usage"
        }

        # Use correct wallet_service endpoint: /api/v1/wallets/{wallet_id}/consume
        result = self._make_request(
            "POST",
            f"{self.config.wallet_service_url}/api/v1/wallets/{wallet_id}/consume",
            headers=headers,
            json=payload
        )

        # Parse response and extract remaining credits
        if result.get("success"):
            # Wallet service returns balance in the response
            remaining_balance = result.get("balance") or result.get("data", {}).get("balance_after")
            return {
                "success": True,
                "remaining_credits": remaining_balance,
                "wallet_id": wallet_id,
                "transaction_id": result.get("transaction_id")
            }

        return result
    
    def record_usage(
        self,
        user_id: str,
        session_id: str,
        usage_data: Dict[str, Any],
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record usage via wallet_service (consume credits)"""
        credits_charged = usage_data.get("credits_charged", 0.0)
        
        if credits_charged > 0:
            description = (
                f"AI Chat - {usage_data.get('model_name', 'unknown')} - "
                f"Tokens: {usage_data.get('tokens_used', 0)}"
            )
            return self.consume_credits(user_id, credits_charged, description, auth_token)
        
        return {"success": True, "message": "No credits charged"}
    
    # Session management methods (session_service)
    
    def create_session(
        self,
        user_id: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create new session via session_service"""
        headers = self._get_headers(auth_token)
        
        payload = {
            "user_id": user_id,
            "title": title,
            "metadata": metadata or {},
            "is_active": True
        }
        
        return self._make_request(
            "POST",
            f"{self.config.session_service_url}/api/v1/sessions",
            headers=headers,
            json=payload
        )
    
    def get_session(
        self,
        session_id: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get session details via session_service"""
        headers = self._get_headers(auth_token)
        
        return self._make_request(
            "GET",
            f"{self.config.session_service_url}/api/v1/sessions/{session_id}",
            headers=headers
        )
    
    def get_user_sessions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user sessions via session_service"""
        headers = self._get_headers(auth_token)
        
        page = (offset // limit) + 1
        page_size = limit
        
        return self._make_request(
            "GET",
            f"{self.config.session_service_url}/api/v1/users/{user_id}/sessions",
            headers=headers,
            params={"page": page, "page_size": page_size}
        )
    
    def add_session_message(
        self,
        session_id: str,
        role: str,
        content: Any,
        message_type: str = "chat",
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add message to session via session_service"""
        headers = self._get_headers(auth_token)
        
        payload = {
            "role": role,
            "content": str(content),
            "message_type": message_type,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "metadata": metadata or {}
        }
        
        return self._make_request(
            "POST",
            f"{self.config.session_service_url}/api/v1/sessions/{session_id}/messages",
            headers=headers,
            json=payload
        )
    
    def get_session_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        role: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get session messages via session_service"""
        headers = self._get_headers(auth_token)
        
        page = (offset // limit) + 1
        params = {"page": page, "page_size": limit}
        
        return self._make_request(
            "GET",
            f"{self.config.session_service_url}/api/v1/sessions/{session_id}/messages",
            headers=headers,
            params=params
        )
    
    def update_session_activity(
        self,
        session_id: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update session activity via session_service"""
        headers = self._get_headers(auth_token)
        
        payload = {
            "last_activity": datetime.utcnow().isoformat()
        }
        
        return self._make_request(
            "PUT",
            f"{self.config.session_service_url}/api/v1/sessions/{session_id}",
            headers=headers,
            json=payload
        )
    
    def delete_session(
        self,
        session_id: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete session via session_service"""
        headers = self._get_headers(auth_token)
        
        return self._make_request(
            "DELETE",
            f"{self.config.session_service_url}/api/v1/sessions/{session_id}",
            headers=headers
        )
    
    # File storage methods (storage_service)
    
    def upload_file(
        self,
        user_id: str,
        file_path: Union[str, Path],
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload file via storage_service"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            with open(file_path, "rb") as file:
                files = {"file": (file_path.name, file, "application/octet-stream")}
                
                response = self.session.post(
                    f"{self.config.storage_service_url}/api/v1/users/{user_id}/files/upload",
                    headers=headers,
                    files=files
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            api_logger.error(f"Failed to upload file for user {user_id}: {e}")
            return {"success": False, "error": str(e), "status": "error"}
    
    def upload_file_content(
        self,
        user_id: str,
        file_content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload file content via storage_service"""
        try:
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            files = {"file": (filename, file_content, content_type)}
            
            response = self.session.post(
                f"{self.config.storage_service_url}/api/v1/users/{user_id}/files/upload",
                headers=headers,
                files=files
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            api_logger.error(f"Failed to upload file content for user {user_id}: {e}")
            return {"success": False, "error": str(e), "status": "error"}
    
    def get_user_files(
        self,
        user_id: str,
        prefix: str = "",
        limit: int = 100,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user files list via storage_service"""
        headers = self._get_headers(auth_token)
        
        params = {"prefix": prefix, "limit": limit}
        
        return self._make_request(
            "GET",
            f"{self.config.storage_service_url}/api/v1/users/{user_id}/files",
            headers=headers,
            params=params
        )
    
    def get_file_info(
        self,
        user_id: str,
        file_path: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get file information via storage_service"""
        headers = self._get_headers(auth_token)
        
        params = {"file_path": file_path}
        
        return self._make_request(
            "GET",
            f"{self.config.storage_service_url}/api/v1/users/{user_id}/files/info",
            headers=headers,
            params=params
        )
    
    def delete_file(
        self,
        user_id: str,
        file_path: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete user file via storage_service"""
        headers = self._get_headers(auth_token)
        
        params = {"file_path": file_path}
        
        return self._make_request(
            "DELETE",
            f"{self.config.storage_service_url}/api/v1/users/{user_id}/files",
            headers=headers,
            params=params
        )


# Global microservices-based user service instance
user_service = UserServiceMicroservices()


def cleanup_user_service():
    """Cleanup function for graceful shutdown"""
    user_service.close()