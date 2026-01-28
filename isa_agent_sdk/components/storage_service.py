"""
Storage Service Client

Client for communicating with the storage_service microservice
Handles file uploads, RAG queries, and semantic search
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from fastapi import UploadFile

from isa_agent_sdk.utils.logger import api_logger
from .consul_discovery import discover_service_url

logger = logging.getLogger(__name__)


class StorageService:
    """Client for interacting with storage_service microservice"""
    
    def __init__(self, storage_service_url: Optional[str] = None):
        """Initialize storage client with Consul service discovery
        
        Args:
            storage_service_url: URL of the storage service (if None, will use Consul discovery)
        """
        # Use Consul service discovery with fallback to configured storage service
        if storage_service_url is None:
            from isa_agent_sdk.core.config import settings
            discovered_url = discover_service_url("storage_service", settings.storage_service_url)
            self.base_url = discovered_url.rstrip('/')
        else:
            self.base_url = storage_service_url.rstrip('/')
            
        self.timeout = 60.0
        self.session = requests.Session()
        self.session.timeout = self.timeout
        
        api_logger.info(f"Storage service initialized with URL: {self.base_url}")
    
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
        """Make protected HTTP request"""
        try:
            # Ensure timeout is set if not provided
            if 'timeout' not in kwargs:
                kwargs['timeout'] = self.timeout
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            api_logger.error(f"Storage service request failed {method} {url}: {e}")
            return {"success": False, "error": str(e), "status": "error"}
        
    def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        user_id: str,
        organization_id: Optional[str] = None,
        access_level: str = "private",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
        enable_indexing: bool = True
    ) -> Dict[str, Any]:
        """Upload file to storage service with automatic indexing
        
        Args:
            file_content: File content bytes
            filename: File name
            content_type: File content type
            user_id: User ID
            organization_id: Organization ID (optional)
            access_level: Access level (private/public/restricted/shared)
            metadata: File metadata
            tags: File tags
            auth_token: Auth token (optional)
            
        Returns:
            Upload response from storage service
        """
        try:
            # Prepare headers
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            # Prepare form data
            files = {
                'file': (filename, file_content, content_type)
            }
            
            data = {
                'user_id': user_id,
                'access_level': access_level,
                'enable_indexing': str(enable_indexing).lower()
            }
            
            if organization_id:
                data['organization_id'] = organization_id
            if metadata:
                data['metadata'] = json.dumps(metadata)
            if tags:
                data['tags'] = ','.join(tags)
            
            response = self.session.post(
                f"{self.base_url}/api/v1/files/upload",
                files=files,
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                api_logger.info(f"File uploaded successfully: {filename} -> {result.get('file_id')}")
                return {
                    'success': True,
                    'file_id': result.get('file_id'),
                    'download_url': result.get('download_url'),
                    'file_size': result.get('file_size'),
                    'content_type': result.get('content_type'),
                    'uploaded_at': result.get('uploaded_at')
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                api_logger.error(f"File upload failed: {response.status_code} - {error_detail}")
                return {
                    'success': False,
                    'error': f"Upload failed: {error_detail}"
                }
                
        except Exception as e:
            api_logger.error(f"Storage service upload error: {e}")
            return {
                'success': False,
                'error': f"Storage service error: {str(e)}"
            }
    
    def semantic_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        enable_rerank: bool = False,
        min_score: float = 0.0,
        file_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Semantic search across user's files
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            enable_rerank: Enable result reranking
            min_score: Minimum relevance score
            file_types: Filter by file types
            tags: Filter by tags
            auth_token: Auth token (optional)
            
        Returns:
            Search results
        """
        try:
            search_request = {
                "user_id": user_id,
                "query": query,
                "top_k": top_k,
                "enable_rerank": enable_rerank,
                "min_score": min_score
            }
            
            if file_types:
                search_request["file_types"] = file_types
            if tags:
                search_request["tags"] = tags
            
            headers = self._get_headers(auth_token)
            
            response = self.session.post(
                f"{self.base_url}/api/v1/files/search",
                json=search_request,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                api_logger.info(f"Semantic search completed: {len(result.get('results', []))} results")
                return {
                    'success': True,
                    'query': result.get('query'),
                    'results': result.get('results', []),
                    'results_count': result.get('results_count', 0),
                    'latency_ms': result.get('latency_ms', 0)
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                api_logger.error(f"Semantic search failed: {response.status_code} - {error_detail}")
                return {
                    'success': False,
                    'error': f"Search failed: {error_detail}"
                }
                
        except Exception as e:
            api_logger.error(f"Storage service search error: {e}")
            return {
                'success': False,
                'error': f"Search error: {str(e)}"
            }
    
    def rag_query(
        self,
        user_id: str,
        query: str,
        rag_mode: str = "simple",
        session_id: Optional[str] = None,
        top_k: int = 3,
        enable_citations: bool = True,
        max_tokens: int = 500,
        temperature: float = 0.7,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """RAG query against user's files
        
        Args:
            user_id: User ID
            query: User question
            rag_mode: RAG mode (simple, raptor, self_rag, crag, plan_rag, hm_rag)
            session_id: Session ID for multi-turn conversation
            top_k: Number of documents to retrieve
            enable_citations: Enable citation extraction
            max_tokens: Maximum response length
            temperature: Generation temperature
            auth_token: Auth token (optional)
            
        Returns:
            RAG response with answer and sources
        """
        try:
            rag_request = {
                "user_id": user_id,
                "query": query,
                "rag_mode": rag_mode,
                "top_k": top_k,
                "enable_citations": enable_citations,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            if session_id:
                rag_request["session_id"] = session_id
            
            headers = self._get_headers(auth_token)
            
            response = self.session.post(
                f"{self.base_url}/api/v1/files/ask",
                json=rag_request,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                rag_answer = result.get('rag_answer', {})
                api_logger.info(f"RAG query completed: {len(rag_answer.get('answer', ''))} chars")
                return {
                    'success': True,
                    'query': result.get('query'),
                    'answer': rag_answer.get('answer', ''),
                    'confidence': rag_answer.get('confidence', 0.0),
                    'sources': rag_answer.get('sources', []),
                    'citations': rag_answer.get('citations', []),
                    'latency_ms': result.get('latency_ms', 0)
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                api_logger.error(f"RAG query failed: {response.status_code} - {error_detail}")
                return {
                    'success': False,
                    'error': f"RAG query failed: {error_detail}"
                }
                
        except Exception as e:
            api_logger.error(f"Storage service RAG error: {e}")
            return {
                'success': False,
                'error': f"RAG error: {str(e)}"
            }
    
    def get_file_info(self, file_id: str, user_id: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Get file information
        
        Args:
            file_id: File ID
            user_id: User ID
            
        Returns:
            File information
        """
        try:
            headers = self._get_headers(auth_token)
            params = {"user_id": user_id}
            
            response = self.session.get(
                f"{self.base_url}/api/v1/files/{file_id}",
                params=params,
                headers=headers
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'file_info': response.json()
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                return {
                    'success': False,
                    'error': f"Get file info failed: {error_detail}"
                }
                
        except Exception as e:
            api_logger.error(f"Storage service get file info error: {e}")
            return {
                'success': False,
                'error': f"Get file info error: {str(e)}"
            }
    
    def list_files(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List user files
        
        Args:
            user_id: User ID
            organization_id: Organization ID (optional)
            limit: Number of files to return
            offset: Offset for pagination
            auth_token: Auth token (optional)
            
        Returns:
            List of user files
        """
        try:
            params = {
                "user_id": user_id,
                "limit": limit,
                "offset": offset
            }
            
            if organization_id:
                params["organization_id"] = organization_id
            
            headers = self._get_headers(auth_token)
            
            response = self.session.get(
                f"{self.base_url}/api/v1/files",
                params=params,
                headers=headers
            )
            
            if response.status_code == 200:
                files = response.json()
                return {
                    'success': True,
                    'files': files,
                    'count': len(files)
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                return {
                    'success': False,
                    'error': f"List files failed: {error_detail}"
                }
                
        except Exception as e:
            api_logger.error(f"Storage service list files error: {e}")
            return {
                'success': False,
                'error': f"List files error: {str(e)}"
            }
    
    def delete_file(self, file_id: str, user_id: str, permanent: bool = False, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Delete file
        
        Args:
            file_id: File ID
            user_id: User ID
            permanent: Whether to permanently delete
            auth_token: Auth token (optional)
            
        Returns:
            Deletion result
        """
        try:
            params = {
                "user_id": user_id,
                "permanent": permanent
            }
            
            headers = self._get_headers(auth_token)
            
            response = self.session.delete(
                f"{self.base_url}/api/v1/files/{file_id}",
                params=params,
                headers=headers
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'message': response.json().get('message', 'File deleted')
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                return {
                    'success': False,
                    'error': f"Delete file failed: {error_detail}"
                }
                
        except Exception as e:
            api_logger.error(f"Storage service delete file error: {e}")
            return {
                'success': False,
                'error': f"Delete file error: {str(e)}"
            }
    
    def get_storage_stats(self, user_id: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Get storage statistics
        
        Args:
            user_id: User ID
            auth_token: Auth token (optional)
            
        Returns:
            Storage statistics
        """
        try:
            headers = self._get_headers(auth_token)
            params = {"user_id": user_id}
            
            response = self.session.get(
                f"{self.base_url}/api/v1/storage/stats",
                params=params,
                headers=headers
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'stats': response.json()
                }
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
                return {
                    'success': False,
                    'error': f"Get stats failed: {error_detail}"
                }
                
        except Exception as e:
            api_logger.error(f"Storage service get stats error: {e}")
            return {
                'success': False,
                'error': f"Get stats error: {str(e)}"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check storage service health
        
        Returns:
            Health status
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5.0)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'status': 'healthy',
                    'details': response.json()
                }
            else:
                return {
                    'success': False,
                    'status': 'unhealthy',
                    'error': f"Health check failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': 'unreachable',
                'error': f"Health check error: {str(e)}"
            }


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service(storage_service_url: Optional[str] = None) -> StorageService:
    """Get storage service singleton instance
    
    Args:
        storage_service_url: URL of the storage service
        
    Returns:
        StorageService instance
    """
    global _storage_service
    
    if _storage_service is None:
        _storage_service = StorageService(storage_service_url)
        
        # Test connection
        health = _storage_service.health_check()
        if health.get('success'):
            api_logger.info(f"Storage service connected: {storage_service_url}")
        else:
            api_logger.warning(f"Storage service health check failed: {health.get('error')}")
    
    return _storage_service


def cleanup_storage_service():
    """Cleanup function for graceful shutdown"""
    global _storage_service
    if _storage_service:
        _storage_service.close()
        _storage_service = None