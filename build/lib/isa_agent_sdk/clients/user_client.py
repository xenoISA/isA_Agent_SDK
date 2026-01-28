"""
User Client - Simplified microservices client for agent service

Directly connects to user-staging container microservices without Consul discovery.
This client replaces app/components/user_service.py with a cleaner, more maintainable approach.

Services available in user-staging container:
- account_service: port 8202
- session_service: port 8203
- wallet_service: port 8208
- storage_service: port 8209
"""

import requests
from typing import Dict, Any, Optional
from datetime import datetime

from isa_agent_sdk.utils.logger import api_logger


class UserClient:
    """
    Simplified client for user-staging microservices

    Uses direct container-to-container communication without Consul discovery.
    All services run in the user-staging Docker container.
    """

    def __init__(self, base_host: str = "user-staging", timeout: float = 5.0):
        """
        Initialize user client

        Args:
            base_host: Base hostname for microservices (default: user-staging)
            timeout: Request timeout in seconds (default: 5.0)
        """
        self.base_host = base_host
        self.timeout = timeout
        self.session = requests.Session()

        # Service ports
        self.ports = {
            'account': 8202,
            'session': 8203,
            'wallet': 8208,
            'storage': 8209
        }

        api_logger.info(f"UserClient initialized with base_host={base_host}, timeout={timeout}s")

    def close(self):
        """Close HTTP session"""
        if self.session:
            self.session.close()
            self.session = None

    def _get_url(self, service: str, path: str) -> str:
        """
        Build service URL

        Args:
            service: Service name (account, session, wallet, storage)
            path: API path

        Returns:
            Full URL
        """
        port = self.ports.get(service)
        if not port:
            raise ValueError(f"Unknown service: {service}")

        # Ensure path starts with /
        if not path.startswith('/'):
            path = f'/{path}'

        return f"http://{self.base_host}:{port}{path}"

    def _request(
        self,
        method: str,
        service: str,
        path: str,
        auth_token: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request to microservice

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            service: Service name
            path: API path
            auth_token: Optional auth token
            **kwargs: Additional request parameters

        Returns:
            Response JSON or error dict
        """
        url = self._get_url(service, path)

        # Set headers
        headers = kwargs.pop('headers', {})
        headers['Content-Type'] = 'application/json'
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'

        # Ensure timeout is set
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            api_logger.error(f"Request timeout: {method} {url} (timeout={self.timeout}s)")
            return {"success": False, "error": "Request timeout", "status": "timeout"}

        except requests.exceptions.ConnectionError as e:
            api_logger.error(f"Connection error: {method} {url}: {e}")
            return {"success": False, "error": f"Connection error: {str(e)}", "status": "connection_error"}

        except requests.exceptions.HTTPError as e:
            api_logger.error(f"HTTP error: {method} {url}: {e}")
            return {"success": False, "error": f"HTTP error: {str(e)}", "status": "http_error"}

        except Exception as e:
            api_logger.error(f"Request failed: {method} {url}: {e}")
            return {"success": False, "error": str(e), "status": "error"}

    # ==================== Health Check ====================

    def health_check(self) -> Dict[str, Any]:
        """Check all microservices health"""
        results = {}
        overall_healthy = True

        for service_name, port in self.ports.items():
            try:
                url = f"http://{self.base_host}:{port}/health"
                response = self.session.get(url, timeout=2.0)
                response.raise_for_status()
                results[service_name] = response.json()
            except Exception as e:
                api_logger.warning(f"{service_name}_service health check failed: {e}")
                results[service_name] = {"status": "error", "error": str(e)}
                overall_healthy = False

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "services": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    # ==================== Account Service ====================

    def ensure_user(
        self,
        user_id: str,
        email: str,
        name: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ensure user account exists (create if not exists)

        Args:
            user_id: User ID
            email: User email
            name: User name
            auth_token: Optional auth token

        Returns:
            Response from account_service
        """
        return self._request(
            'POST',
            'account',
            '/api/v1/accounts/ensure',
            auth_token=auth_token,
            json={
                'user_id': user_id,
                'email': email,
                'name': name,
                'is_active': True
            }
        )

    def get_user_profile(
        self,
        user_id: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user profile from account_service"""
        return self._request(
            'GET',
            'account',
            f'/api/v1/accounts/profile/{user_id}',
            auth_token=auth_token
        )

    # ==================== Session Service ====================

    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create new session

        Args:
            user_id: User ID
            title: Session title
            metadata: Session metadata
            auth_token: Optional auth token

        Returns:
            Response from session_service
        """
        return self._request(
            'POST',
            'session',
            '/api/v1/sessions',
            auth_token=auth_token,
            json={
                'user_id': user_id,
                'title': title or f'Session {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}',
                'metadata': metadata or {}
            }
        )

    def get_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get session details"""
        params = {'user_id': user_id} if user_id else {}
        return self._request(
            'GET',
            'session',
            f'/api/v1/sessions/{session_id}',
            auth_token=auth_token,
            params=params
        )

    def get_user_sessions(
        self,
        user_id: str,
        active_only: bool = False,
        page: int = 1,
        page_size: int = 50,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user sessions"""
        return self._request(
            'GET',
            'session',
            f'/api/v1/users/{user_id}/sessions',
            auth_token=auth_token,
            params={
                'active_only': active_only,
                'page': page,
                'page_size': page_size
            }
        )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add message to session"""
        params = {'user_id': user_id} if user_id else {}
        return self._request(
            'POST',
            'session',
            f'/api/v1/sessions/{session_id}/messages',
            auth_token=auth_token,
            params=params,
            json={
                'role': role,
                'content': content
            }
        )

    def get_session_messages(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get session messages"""
        params = {
            'page': page,
            'page_size': page_size
        }
        if user_id:
            params['user_id'] = user_id

        return self._request(
            'GET',
            'session',
            f'/api/v1/sessions/{session_id}/messages',
            auth_token=auth_token,
            params=params
        )

    # ==================== Wallet Service ====================

    def get_credit_balance(
        self,
        user_id: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user credit balance"""
        return self._request(
            'GET',
            'wallet',
            f'/api/v1/users/{user_id}/credits/balance',
            auth_token=auth_token
        )

    def consume_credits(
        self,
        user_id: str,
        amount: float,
        description: str,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Consume user credits

        Args:
            user_id: User ID
            amount: Credit amount to consume
            description: Transaction description
            auth_token: Optional auth token

        Returns:
            Response from wallet_service
        """
        # First get wallet_id
        balance_result = self.get_credit_balance(user_id, auth_token)

        if not balance_result.get("success"):
            return {
                "success": False,
                "error": f"Failed to get wallet: {balance_result.get('error', 'unknown')}"
            }

        wallet_id = balance_result.get("wallet_id")
        if not wallet_id:
            return {"success": False, "error": "No wallet found for user"}

        # Consume credits
        return self._request(
            'POST',
            'wallet',
            f'/api/v1/wallets/{wallet_id}/transactions/consume',
            auth_token=auth_token,
            json={
                'amount': amount,
                'description': description
            }
        )

    # ==================== Storage Service ====================

    def get_storage_stats(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get storage statistics"""
        params = {}
        if user_id:
            params['user_id'] = user_id
        if organization_id:
            params['organization_id'] = organization_id

        return self._request(
            'GET',
            'storage',
            '/api/v1/storage/stats',
            auth_token=auth_token,
            params=params
        )

    def get_storage_quota(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get storage quota"""
        params = {}
        if user_id:
            params['user_id'] = user_id
        if organization_id:
            params['organization_id'] = organization_id

        return self._request(
            'GET',
            'storage',
            '/api/v1/storage/quota',
            auth_token=auth_token,
            params=params
        )


# Global singleton instance
_user_client = None


def get_user_client() -> UserClient:
    """
    Get or create global UserClient instance

    Returns:
        UserClient singleton
    """
    global _user_client
    if _user_client is None:
        _user_client = UserClient()
    return _user_client


# Alias for backward compatibility
user_client = get_user_client()
