"""
Session service with User Service API integration
Replaces direct database operations with User Service API calls for better service architecture
"""
import uuid
import datetime
from typing import Dict, Optional, Any, List

from isa_agent_sdk.clients.user_client import user_client
from isa_agent_sdk.utils.logger import api_logger


class SessionService:
    """Session service using User Service API for all persistence operations"""
    
    def __init__(self, auth_token: Optional[str] = None):
        self.sessions: Dict[str, Dict] = {}  # In-memory cache
        self.current_session_id: Optional[str] = None
        self.auth_token = auth_token  # JWT token for User Service API
        
        api_logger.info("🔗 SessionService initialized with User Service API integration")
    
    def set_auth_token(self, token: str):
        """Set authentication token for User Service API"""
        self.auth_token = token
        api_logger.info("🔐 Auth token set for SessionService")
    
    async def create_session(self, session_id: Optional[str] = None, user_id: str = "anonymous", title: Optional[str] = None) -> str:
        """Create new session via User Service API"""
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        if title is None:
            title = f"Session {session_id[:8]}"
        
        # Ensure user exists first (required by User Service API)
        if user_id != "anonymous" and self.auth_token:
            try:
                ensure_result = user_client.ensure_user(
                    user_id=user_id,
                    email=f"{user_id.split('|')[-1]}@temp.com",  # Fallback email
                    name=f"User {user_id.split('|')[-1]}",  # Fallback name
                    auth_token=self.auth_token
                )
                if not ensure_result.get("success"):
                    api_logger.warning(f"⚠️ Failed to ensure user exists: {ensure_result.get('error')}")
            except Exception as e:
                api_logger.warning(f"⚠️ Error ensuring user exists: {e}")

        # Create session via User Service API
        try:
            result = user_client.create_session(
                user_id=user_id,
                title=title,
                metadata={
                    "source": "session_service",
                    "client_type": "isA_agent",
                    "created_via": "api"
                },
                auth_token=self.auth_token
            )
            
            if result.get("success"):
                # Extract session_id from API response
                api_session_data = result.get("data", {})
                api_session_id = api_session_data.get("session_id") or api_session_data.get("id")
                
                if api_session_id:
                    session_id = api_session_id
                
                # Create in-memory cache entry
                session_data = {
                    "id": session_id,
                    "user_id": user_id,
                    "title": title,
                    "created_at": datetime.datetime.now(),
                    "last_activity": datetime.datetime.now(),
                    "message_count": 0,
                    "summary": "",
                    "metadata": {
                        "source": "session_service",
                        "client_type": "isA_agent",
                        "created_via": "api"
                    },
                    "context": {},
                    "preferences": {},
                    "task_history": []
                }
                
                self.sessions[session_id] = session_data
                self.current_session_id = session_id
                
                api_logger.info(f"🔗 Created session via API: {session_id}")
                return session_id
            else:
                api_logger.error(f"⚠️ Failed to create session via API: {result.get('error')}")
                raise Exception(f"User Service API error: {result.get('error')}")
                
        except Exception as e:
            api_logger.error(f"⚠️ Session creation failed: {e}")
            # Fallback: create local session only
            session_data = {
                "id": session_id,
                "user_id": user_id,
                "title": title,
                "created_at": datetime.datetime.now(),
                "last_activity": datetime.datetime.now(),
                "message_count": 0,
                "summary": "",
                "metadata": {},  # Add empty metadata to prevent KeyError
                "context": {},
                "preferences": {},
                "task_history": []
            }
            
            self.sessions[session_id] = session_data
            self.current_session_id = session_id
            
            api_logger.warning(f"🔄 Created fallback local session: {session_id}")
            return session_id
    
    async def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """Get session data (try memory first, then User Service API via user sessions)"""
        # Try memory cache first
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # If no user_id provided, we can't fetch from User Service API
        # because there's no direct session endpoint - only user session lists
        if not user_id:
            api_logger.warning(f"⚠️ Cannot fetch session {session_id} from API without user_id")
            return None
        
        # Try loading from User Service API via user sessions
        try:
            result = user_client.get_user_sessions(
                user_id=user_id,
                page=1,
                page_size=100,  # Get enough sessions to find ours
                auth_token=self.auth_token
            )
            
            if result.get("success"):
                sessions_data = result.get("data", [])
                
                # Find the specific session
                for api_session in sessions_data:
                    if api_session.get("session_id") == session_id:
                        # Convert API response to our format and cache
                        formatted_session = {
                            "id": api_session.get("session_id"),
                            "user_id": api_session.get("user_id"),
                            "title": api_session.get("conversation_data", {}).get("topic", f"Session {session_id[:8]}"),
                            "created_at": self._parse_datetime(api_session.get("created_at")),
                            "last_activity": self._parse_datetime(api_session.get("last_activity")),
                            "message_count": api_session.get("message_count", 0),
                            "summary": api_session.get("session_summary", ""),
                            "metadata": api_session.get("metadata", {}),
                            "context": {},
                            "preferences": {},
                            "task_history": []
                        }
                        
                        self.sessions[session_id] = formatted_session
                        return formatted_session
                
        except Exception as e:
            api_logger.error(f"⚠️ Failed to load session from User Service API: {e}")
        
        return None
    
    async def update_session_activity(self, session_id: str, user_id: str = "anonymous"):
        """Update session last activity via User Service API"""
        # Update in-memory cache
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.datetime.now()
            self.sessions[session_id]["message_count"] += 1
        
        # Update via User Service API
        # Note: user_client doesn't have update_session_activity endpoint
        # Session activity is updated automatically when messages are added
        try:
            pass  # Activity updated via message operations
        except Exception as e:
            api_logger.warning(f"⚠️ Failed to update session activity via API: {e}")
    
    async def add_message_to_session(
        self,
        session_id: str,
        role: str,
        content: Any,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add message to session via User Service API"""
        try:
            result = user_client.add_message(
                session_id=session_id,
                role=role,
                content=str(content),
                auth_token=self.auth_token
            )
            
            if result.get("success"):
                # Update local cache
                await self.update_session_activity(session_id)
                api_logger.info(f"💬 Added message to session {session_id}")
                return result
            else:
                api_logger.error(f"⚠️ Failed to add message: {result.get('error')}")
                return result
                
        except Exception as e:
            api_logger.error(f"⚠️ Error adding message to session: {e}")
            return {"success": False, "error": str(e)}
    
    def set_session_context(self, session_id: str, key: str, value: Any):
        """Set session context (local only for now)"""
        if session_id in self.sessions:
            self.sessions[session_id]["context"][key] = value
    
    def get_session_context(self, session_id: str, key: str, default=None):
        """Get session context (local only for now)"""
        if session_id in self.sessions:
            return self.sessions[session_id]["context"].get(key, default)
        return default
    
    async def get_session_with_history(self, session_id: str) -> Optional[Dict]:
        """Get session data with conversation history from User Service API"""
        session_data = await self.get_session(session_id)
        
        if session_data:
            try:
                # Get conversation history from User Service API
                messages_result = user_client.get_session_messages(
                    session_id=session_id,
                    page=1,
                    page_size=100,  # Get recent messages
                    auth_token=self.auth_token
                )
                
                if messages_result.get("success"):
                    session_data['conversation_history'] = messages_result.get("data", [])
                    
                    # Add basic stats
                    session_data['stats'] = {
                        "message_count": len(messages_result.get("data", [])),
                        "last_updated": datetime.datetime.now().isoformat()
                    }
                    
            except Exception as e:
                api_logger.warning(f"⚠️ Failed to get session history from API: {e}")
                session_data['conversation_history'] = []
                session_data['stats'] = {}
        
        return session_data
    
    async def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user from User Service API"""
        try:
            result = user_client.get_user_sessions(
                user_id=user_id,
                page=1,
                page_size=50,
                auth_token=self.auth_token
            )
            
            if result.get("success"):
                return result.get("data", [])
            else:
                api_logger.error(f"⚠️ Failed to get user sessions: {result.get('error')}")
                
        except Exception as e:
            api_logger.error(f"⚠️ Error getting user sessions: {e}")
        
        return []
    
    async def clear_session(self, session_id: str):
        """Clear/delete session via User Service API"""
        # Remove from memory cache
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Delete via User Service API - Note: endpoint not yet implemented in user_client
        # Session deletion handled locally for now
        try:
            # TODO: Implement session deletion in user_client when endpoint is available
            api_logger.info(f"🗑️ Session {session_id} deleted locally (API deletion pending)")
        except Exception as e:
            api_logger.warning(f"⚠️ Error deleting session via API: {e}")
        
        api_logger.info(f"🗑️ Session {session_id} cleared from cache")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs from cache"""
        return list(self.sessions.keys())
    
    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self.current_session_id
    
    def set_current_session_id(self, session_id: str):
        """Set current session ID"""
        self.current_session_id = session_id
    
    def _parse_datetime(self, dt_str: Any) -> datetime.datetime:
        """Parse datetime string from API response"""
        if isinstance(dt_str, str):
            try:
                # Try ISO format first
                return datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            except:
                try:
                    # Try parsing common formats
                    return datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
                except:
                    pass
        
        # Fallback to current time
        return datetime.datetime.now()